#include "welvet/metacognition.hpp"

#include "welvet/weights.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::metacognition {
namespace {

struct Stats {
  double avg = 0, std = 0, max = 0, min = 0;
  int active = 0, total = 0;
};

struct Layout {
  int batch = 0;
  int seq = 0;
  int dim = 0;
};

Status parse_layout(int dim, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: empty input");
  }
  lay->dim = dim;
  if (input.shape.size() == 2) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = 1;
    if (static_cast<int>(input.shape[1]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "metacognition: width != Dim");
    }
  } else if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "metacognition: last dim != Dim");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument,
                       "metacognition: need [batch,dim] or [batch,seq,dim]");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: invalid batch/seq");
  }
  const int want = lay->batch * lay->seq * lay->dim;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: data short");
  }
  return Status::Ok();
}

TensorF32 flatten(const TensorF32& input, const Layout& lay) {
  TensorF32 out;
  const int n = lay.batch * lay.seq;
  out.shape = {n, lay.dim};
  out.data.assign(input.data.begin(),
                  input.data.begin() + static_cast<std::ptrdiff_t>(n * lay.dim));
  return out;
}

TensorF32 unflatten(const TensorF32& flat, const Layout& lay) {
  TensorF32 out;
  if (lay.seq == 1) {
    out.shape = {lay.batch, lay.dim};
  } else {
    out.shape = {lay.batch, lay.seq, lay.dim};
  }
  const int n = lay.batch * lay.seq * lay.dim;
  out.data.assign(flat.data.begin(), flat.data.begin() + static_cast<std::ptrdiff_t>(n));
  return out;
}

Stats compute_stats(const TensorF32& t) {
  Stats s;
  if (t.data.empty()) return s;
  const int n = static_cast<int>(t.data.size());
  s.total = n;
  double sum = 0, sum_sq = 0;
  s.max = static_cast<double>(t.data[0]);
  s.min = s.max;
  for (int i = 0; i < n; ++i) {
    const double v = static_cast<double>(t.data[static_cast<std::size_t>(i)]);
    sum += v;
    sum_sq += v * v;
    if (v > s.max) s.max = v;
    if (v < s.min) s.min = v;
    if (std::fabs(v) > 1e-6) s.active++;
  }
  s.avg = sum / static_cast<double>(n);
  if (n > 1) {
    s.std = std::sqrt(std::max(0.0, sum_sq / static_cast<double>(n) - s.avg * s.avg));
  }
  return s;
}

bool eval_rule(const Rule& r, const Stats& in_s, const Stats& out_s) {
  switch (r.condition) {
    case Condition::StdAbove:
      return out_s.std > r.threshold;
    case Condition::StdBelow:
      return out_s.std < r.threshold && out_s.std >= 0;
    case Condition::AvgAbove:
      return std::fabs(out_s.avg) > r.threshold;
    case Condition::AvgBelow:
      return std::fabs(out_s.avg) < r.threshold;
    case Condition::MaxAbove:
      return std::fabs(out_s.max) > r.threshold;
    case Condition::ActiveBelow:
      if (out_s.total == 0) return false;
      return static_cast<double>(out_s.active) / static_cast<double>(out_s.total) < r.threshold;
    case Condition::GainDrift: {
      const double in_mag = std::fabs(in_s.avg);
      if (in_mag < 1e-6) return false;
      const double gain = std::fabs(out_s.avg) / in_mag;
      return std::fabs(gain - 1.0) > r.threshold;
    }
    default:
      return false;
  }
}

Status reset_identity(dense::Layer& obs) {
  const int rows = obs.weights.rows;
  const int cols = obs.weights.cols;
  if (rows != cols) return Status::Ok();
  std::vector<float> w(static_cast<std::size_t>(rows * cols), 0.0f);
  for (int i = 0; i < rows; ++i) {
    w[static_cast<std::size_t>(i * cols + i)] = 1.0f;
  }
  return WeightStore::make(rows, cols, obs.weights.dtype, w.data(), &obs.weights);
}

} // namespace

int weight_count(const Config& cfg) { return cfg.dim * cfg.dim; }

std::vector<Rule> default_stability_rules() {
  return {
      {Condition::GainDrift, 0.5, Effect::Identity, 5, 0},
      {Condition::StdAbove, 50.0, Effect::Gate, 3, 0},
      {Condition::ActiveBelow, 0.01, Effect::Identity, 10, 0},
  };
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "metacognition: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: need positive Dim");
  }
  if (cfg->seq_len < 0) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: SeqLen < 0");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "metacognition: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st =
          dense::new_layer(cfg.dim, cfg.dim, Activation::Linear, dt, packed_or_null, &layer.observed);
      !st) {
    return Status::Err(st.code, std::string("metacognition observed: ") + st.message);
  }
  if (cfg.rules.empty()) {
    layer.rules = default_stability_rules();
  } else {
    layer.rules = cfg.rules;
  }
  layer.last_gate = 1.0;
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(cfg.dim, input, &lay); !st) return st;

  TensorF32 flat = flatten(input, lay);
  const Stats in_stats = compute_stats(flat);
  layer.last_gate = 1.0;

  TensorF32 pre_f, post_f;
  if (auto st = dense::forward_f32(layer.observed, flat, &pre_f, &post_f); !st) {
    return Status::Err(st.code, std::string("metacognition observed: ") + st.message);
  }
  const Stats out_stats = compute_stats(post_f);

  Effect effect = Effect::None;
  for (auto& r : layer.rules) {
    if (r.cooldown > 0 && r.fire_count < r.cooldown) {
      r.fire_count++;
      continue;
    }
    if (eval_rule(r, in_stats, out_stats)) {
      effect = r.effect;
      r.fire_count = 0;
    } else {
      r.fire_count++;
    }
  }

  switch (effect) {
    case Effect::Identity: {
      if (auto st = reset_identity(layer.observed); !st) return st;
      if (auto st = dense::forward_f32(layer.observed, flat, &pre_f, &post_f); !st) {
        return Status::Err(st.code, std::string("metacognition identity fwd: ") + st.message);
      }
      break;
    }
    case Effect::Gate: {
      layer.last_gate = 0.5;
      for (std::size_t i = 0; i < post_f.data.size(); ++i) {
        post_f.data[i] =
            static_cast<float>(static_cast<double>(post_f.data[i]) * layer.last_gate);
      }
      break;
    }
    case Effect::Passthrough: {
      pre_f = flat;
      post_f = flat;
      break;
    }
    default:
      break;
  }

  *pre = unflatten(pre_f, lay);
  *post = unflatten(post_f, lay);
  return Status::Ok();
}

int grad_w_size(const Layer& layer) { return layer.cfg.dim * layer.cfg.dim; }

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: nil grad out");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(cfg.dim, input, &lay); !st) return st;
  const int want = lay.batch * lay.seq * lay.dim;
  if (static_cast<int>(grad_out.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: gradOut short");
  }

  TensorF32 flat = flatten(input, lay);
  TensorF32 gy = flatten(grad_out, lay);
  if (layer.last_gate != 1.0 && layer.last_gate != 0.0) {
    for (std::size_t i = 0; i < gy.data.size(); ++i) {
      gy.data[i] =
          static_cast<float>(static_cast<double>(gy.data[i]) * layer.last_gate);
    }
  }

  TensorF32 pre_f;
  bool have_pre = false;
  if (!pre.data.empty() && static_cast<int>(pre.data.size()) >= want) {
    Layout pre_lay;
    if (parse_layout(cfg.dim, pre, &pre_lay)) {
      pre_f = flatten(pre, pre_lay);
      have_pre = true;
    }
  }
  if (!have_pre) {
    TensorF32 post_unused;
    if (auto st = dense::forward_f32(layer.observed, flat, &pre_f, &post_unused); !st) {
      return Status::Err(st.code, std::string("metacognition pre recompute: ") + st.message);
    }
  }

  TensorF32 gx, dw;
  if (auto st = dense::backward_f32(layer.observed, gy, flat, pre_f, &gx, &dw); !st) {
    return Status::Err(st.code, std::string("metacognition observed bwd: ") + st.message);
  }
  *grad_in = unflatten(gx, lay);
  *grad_w = std::move(dw);
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: ApplyGradSGD nil");
  }
  return dense::apply_grad_sgd(&layer->observed, dW, lr);
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: nil dst");
  }
  return layer.observed.weights.export_f32(dst);
}

} // namespace welvet::metacognition
