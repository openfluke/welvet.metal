#include "welvet/parallel.hpp"

#include "welvet/activate.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace welvet::parallel {
namespace {

struct Layout {
  int batch = 0;
  int rows = 0; // seq or 1
  int dim = 0;
  std::vector<int64_t> shape;
};

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: empty input");
  }
  lay->dim = cfg.dim;
  lay->shape = input.shape;
  if (cfg.seq_len > 0) {
    if (input.shape.size() != 3 || static_cast<int>(input.shape[1]) != cfg.seq_len ||
        static_cast<int>(input.shape[2]) != cfg.dim) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: need [batch,seq,dim]");
    }
    lay->batch = static_cast<int>(input.shape[0]);
    lay->rows = cfg.seq_len;
  } else if (input.shape.size() == 2 && static_cast<int>(input.shape[1]) == cfg.dim) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->rows = 1;
  } else if (input.shape.size() == 3 && static_cast<int>(input.shape[2]) == cfg.dim) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->rows = static_cast<int>(input.shape[1]);
  } else {
    return Status::Err(StatusCode::InvalidArgument, "parallel: unsupported shape");
  }
  if (lay->batch <= 0 || lay->rows <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: invalid batch/rows");
  }
  const int want = lay->batch * lay->rows * lay->dim;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: data short");
  }
  return Status::Ok();
}

TensorF32 flatten(const TensorF32& input, const Layout& lay) {
  TensorF32 out;
  const int n = lay.batch * lay.rows;
  out.shape = {n, lay.dim};
  out.data.assign(input.data.begin(),
                  input.data.begin() + static_cast<std::ptrdiff_t>(n * lay.dim));
  return out;
}

TensorF32 unflatten(const TensorF32& flat, const Layout& lay, int out_dim) {
  TensorF32 out;
  if (lay.rows == 1 && lay.shape.size() == 2) {
    out.shape = {lay.batch, out_dim};
  } else {
    out.shape = {lay.batch, lay.rows, out_dim};
  }
  const int n = lay.batch * lay.rows * out_dim;
  out.data.assign(flat.data.begin(), flat.data.begin() + static_cast<std::ptrdiff_t>(n));
  return out;
}

std::vector<double> softmax_f64(const std::vector<double>& logits) {
  double max_z = logits[0];
  for (double z : logits) {
    if (z > max_z) max_z = z;
  }
  std::vector<double> out(logits.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double e = std::exp(logits[i] - max_z);
    out[i] = e;
    sum += e;
  }
  if (sum == 0.0) {
    const double inv = 1.0 / static_cast<double>(out.size());
    for (double& v : out) v = inv;
    return out;
  }
  for (double& v : out) v /= sum;
  return out;
}

Status combine_equal_width(const Config& cfg, const std::vector<TensorF32>& branch_out,
                           TensorF32* out) {
  if (!out || branch_out.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: combineEqualWidth empty");
  }
  const int nb = static_cast<int>(branch_out.size());
  const int n = static_cast<int>(branch_out[0].data.size());
  for (const auto& o : branch_out) {
    if (static_cast<int>(o.data.size()) != n) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: unequal branch widths");
    }
  }
  out->shape = branch_out[0].shape;
  out->data.assign(static_cast<std::size_t>(n), 0.0f);

  switch (cfg.combine) {
    case Combine::Add:
      for (const auto& o : branch_out) {
        for (int j = 0; j < n; ++j) {
          out->data[static_cast<std::size_t>(j)] += o.data[static_cast<std::size_t>(j)];
        }
      }
      break;
    case Combine::Avg: {
      const float inv = static_cast<float>(1.0 / static_cast<double>(nb));
      for (const auto& o : branch_out) {
        for (int j = 0; j < n; ++j) {
          out->data[static_cast<std::size_t>(j)] += o.data[static_cast<std::size_t>(j)];
        }
      }
      for (int j = 0; j < n; ++j) {
        out->data[static_cast<std::size_t>(j)] *= inv;
      }
      break;
    }
    case Combine::Max:
      for (int j = 0; j < n; ++j) {
        double best = static_cast<double>(branch_out[0].data[static_cast<std::size_t>(j)]);
        for (int i = 1; i < nb; ++i) {
          const double v = static_cast<double>(
              branch_out[static_cast<std::size_t>(i)].data[static_cast<std::size_t>(j)]);
          if (v > best) best = v;
        }
        out->data[static_cast<std::size_t>(j)] = static_cast<float>(best);
      }
      break;
    case Combine::SparseK: {
      int k = sparse_k_eff(cfg);
      if (k > nb) k = nb;
      struct Scored {
        int i;
        double nrm;
      };
      std::vector<Scored> scores(static_cast<std::size_t>(nb));
      for (int i = 0; i < nb; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
          const double v = static_cast<double>(
              branch_out[static_cast<std::size_t>(i)].data[static_cast<std::size_t>(j)]);
          s += v * v;
        }
        scores[static_cast<std::size_t>(i)] = {i, std::sqrt(s)};
      }
      std::sort(scores.begin(), scores.end(), [](const Scored& a, const Scored& b) {
        if (a.nrm != b.nrm) return a.nrm > b.nrm;
        return a.i < b.i;
      });
      const float inv = static_cast<float>(1.0 / static_cast<double>(k));
      for (int t = 0; t < k; ++t) {
        const auto& o = branch_out[static_cast<std::size_t>(scores[static_cast<std::size_t>(t)].i)];
        for (int j = 0; j < n; ++j) {
          out->data[static_cast<std::size_t>(j)] += o.data[static_cast<std::size_t>(j)];
        }
      }
      for (int j = 0; j < n; ++j) {
        out->data[static_cast<std::size_t>(j)] *= inv;
      }
      break;
    }
    case Combine::Disagree: {
      const double beta = disagree_beta_eff(cfg);
      const double inv = 1.0 / static_cast<double>(nb);
      std::vector<double> mean(static_cast<std::size_t>(n), 0.0);
      for (const auto& o : branch_out) {
        for (int j = 0; j < n; ++j) {
          mean[static_cast<std::size_t>(j)] +=
              static_cast<double>(o.data[static_cast<std::size_t>(j)]) * inv;
        }
      }
      if (nb == 2) {
        for (int j = 0; j < n; ++j) {
          const double a = static_cast<double>(branch_out[0].data[static_cast<std::size_t>(j)]);
          const double b = static_cast<double>(branch_out[1].data[static_cast<std::size_t>(j)]);
          out->data[static_cast<std::size_t>(j)] =
              static_cast<float>(mean[static_cast<std::size_t>(j)] + beta * (a - b));
        }
      } else {
        for (int j = 0; j < n; ++j) {
          const double a0 = static_cast<double>(branch_out[0].data[static_cast<std::size_t>(j)]);
          out->data[static_cast<std::size_t>(j)] = static_cast<float>(
              mean[static_cast<std::size_t>(j)] + beta * (a0 - mean[static_cast<std::size_t>(j)]));
        }
      }
      break;
    }
    default:
      return Status::Err(StatusCode::InvalidArgument, "parallel: combineEqualWidth unsupported");
  }
  return Status::Ok();
}

} // namespace

int out_dim(const Config& cfg) {
  if (cfg.out_feat <= 0) return 0;
  if (cfg.combine == Combine::Concat) return cfg.branches * cfg.out_feat;
  return cfg.out_feat;
}

int weight_count(const Config& cfg) {
  const int branch_w = cfg.branches * cfg.out_feat * cfg.dim;
  const int gate_w = cfg.combine == Combine::Filter ? cfg.branches * cfg.dim : 0;
  return branch_w + gate_w;
}

int sparse_k_eff(const Config& cfg) {
  int k = cfg.sparse_k;
  if (k <= 0) k = 2;
  if (k > cfg.branches) k = cfg.branches;
  if (k < 1) k = 1;
  return k;
}

double disagree_beta_eff(const Config& cfg) {
  if (cfg.disagree_beta == 0.0) return 1.0;
  return cfg.disagree_beta;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "parallel: nil config");
  if (cfg->dim <= 0 || cfg->branches <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: need positive Dim/Branches");
  }
  if (cfg->out_feat < 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: OutFeat < 0");
  }
  switch (cfg->combine) {
    case Combine::Concat:
    case Combine::Add:
    case Combine::Avg:
    case Combine::Filter:
    case Combine::Max:
    case Combine::SparseK:
    case Combine::Disagree:
      break;
    default:
      return Status::Err(StatusCode::InvalidArgument, "parallel: unknown Combine");
  }
  if (cfg->seq_len < 0 || cfg->sparse_k < 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: SeqLen/SparseK < 0");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "parallel: nil out");
  Config cfg = cfg_in;
  if (cfg.out_feat <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: NewConfigured requires OutFeat > 0");
  }
  if (auto st = validate(&cfg); !st) return st;

  const int child_n = cfg.out_feat * cfg.dim;
  Layer layer;
  layer.cfg = cfg;
  layer.branches.resize(static_cast<std::size_t>(cfg.branches));
  for (int i = 0; i < cfg.branches; ++i) {
    const float* init = nullptr;
    if (packed_or_null) init = packed_or_null + i * child_n;
    if (auto st = dense::new_layer(cfg.dim, cfg.out_feat, Activation::Linear, dt, init,
                                   &layer.branches[static_cast<std::size_t>(i)]);
        !st) {
      return Status::Err(st.code, std::string("parallel branch ") + std::to_string(i) + ": " +
                                      st.message);
    }
  }
  if (cfg.combine == Combine::Filter) {
    dense::Layer gate;
    const float* ginit = nullptr;
    if (packed_or_null) ginit = packed_or_null + cfg.branches * child_n;
    if (auto st = dense::new_layer(cfg.dim, cfg.branches, Activation::Linear, dt, ginit, &gate);
        !st) {
      return Status::Err(st.code, std::string("parallel gate: ") + st.message);
    }
    layer.gate = std::move(gate);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  if (layer.branches.empty()) {
    *pre = input;
    *post = input;
    return Status::Ok();
  }

  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st) return st;
  TensorF32 flat = flatten(input, lay);
  const int n = lay.batch * lay.rows;
  const int nb = static_cast<int>(layer.branches.size());

  std::vector<TensorF32> branch_out(static_cast<std::size_t>(nb));
  std::vector<int> feats(static_cast<std::size_t>(nb));
  for (int i = 0; i < nb; ++i) {
    TensorF32 p, o;
    if (auto st = dense::forward_f32(layer.branches[static_cast<std::size_t>(i)], flat, &p, &o);
        !st) {
      return Status::Err(st.code, std::string("parallel fwd branch ") + std::to_string(i) + ": " +
                                      st.message);
    }
    if (static_cast<int>(o.data.size()) % n != 0) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: branch out not divisible by n");
    }
    feats[static_cast<std::size_t>(i)] = static_cast<int>(o.data.size()) / n;
    branch_out[static_cast<std::size_t>(i)] = std::move(o);
  }

  int od = 0;
  if (cfg.combine == Combine::Concat) {
    od = std::accumulate(feats.begin(), feats.end(), 0);
  } else {
    od = feats[0];
    for (int i = 1; i < nb; ++i) {
      if (feats[static_cast<std::size_t>(i)] != od) {
        return Status::Err(StatusCode::InvalidArgument, "parallel: unequal branch widths");
      }
    }
  }
  if (cfg.out_feat > 0) {
    const int expected = out_dim(cfg);
    if (expected > 0 && expected != od) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: measured out != cfg.OutDim");
    }
  }

  TensorF32 combined;
  combined.shape = {n, od};
  combined.data.assign(static_cast<std::size_t>(n * od), 0.0f);

  switch (cfg.combine) {
    case Combine::Add:
    case Combine::Avg:
    case Combine::Max:
    case Combine::SparseK:
    case Combine::Disagree: {
      TensorF32 merged;
      if (auto st = combine_equal_width(cfg, branch_out, &merged); !st) return st;
      combined.data = std::move(merged.data);
      break;
    }
    case Combine::Filter: {
      if (!layer.gate) {
        return Status::Err(StatusCode::InvalidArgument, "parallel: filter mode requires Gate");
      }
      TensorF32 gp, go;
      if (auto st = dense::forward_f32(*layer.gate, flat, &gp, &go); !st) {
        return Status::Err(st.code, std::string("parallel gate fwd: ") + st.message);
      }
      const int feat = feats[0];
      for (int r = 0; r < n; ++r) {
        std::vector<double> logits(static_cast<std::size_t>(nb));
        for (int i = 0; i < nb; ++i) {
          logits[static_cast<std::size_t>(i)] =
              static_cast<double>(go.data[static_cast<std::size_t>(r * nb + i)]);
        }
        const auto w = softmax_f64(logits);
        const int base = r * feat;
        for (int i = 0; i < nb; ++i) {
          const double wi = w[static_cast<std::size_t>(i)];
          const auto& o = branch_out[static_cast<std::size_t>(i)];
          for (int j = 0; j < feat; ++j) {
            combined.data[static_cast<std::size_t>(base + j)] += static_cast<float>(
                static_cast<double>(o.data[static_cast<std::size_t>(base + j)]) * wi);
          }
        }
      }
      break;
    }
    case Combine::Concat:
    default:
      for (int r = 0; r < n; ++r) {
        int off = r * od;
        for (int i = 0; i < nb; ++i) {
          const int f = feats[static_cast<std::size_t>(i)];
          const int src = r * f;
          const auto& o = branch_out[static_cast<std::size_t>(i)];
          for (int j = 0; j < f; ++j) {
            combined.data[static_cast<std::size_t>(off + j)] =
                o.data[static_cast<std::size_t>(src + j)];
          }
          off += f;
        }
      }
      break;
  }

  *pre = input;
  *post = unflatten(combined, lay, od);
  return Status::Ok();
}

} // namespace welvet::parallel
