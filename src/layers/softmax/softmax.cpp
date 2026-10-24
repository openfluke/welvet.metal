#include "welvet/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace welvet::softmax {
namespace {

struct Layout {
  std::vector<int64_t> shape;
  int n = 0;
  int rows = 0;
  int cols = 0;
  double temp = 1.0;
};

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: empty input");
  }
  lay->shape = input.shape;
  lay->n = static_cast<int>(input.data.size());
  lay->temp = cfg.temperature == 0.0 ? 1.0 : cfg.temperature;

  switch (cfg.kind) {
  case Kind::Grid: {
    lay->cols = cfg.cols > 0 ? cfg.cols : cfg.dim;
    lay->rows = cfg.rows;
    if (lay->rows <= 0) {
      if (lay->cols <= 0 || lay->n % lay->cols != 0) {
        return Status::Err(StatusCode::InvalidArgument, "softmax: grid n not divisible by cols");
      }
      lay->rows = lay->n / lay->cols;
    }
    if (lay->rows * lay->cols > lay->n) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: grid rows×cols > n");
    }
    break;
  }
  case Kind::Hierarchical: {
    if (!cfg.hierarchy_levels.empty()) {
      lay->cols = cfg.hierarchy_levels.back();
    } else if (cfg.cols > 0) {
      lay->cols = cfg.cols;
    } else {
      lay->cols = cfg.dim;
    }
    lay->rows = cfg.rows;
    if (lay->rows <= 0) {
      if (lay->cols <= 0 || lay->n % lay->cols != 0) {
        return Status::Err(StatusCode::InvalidArgument,
                           "softmax: hierarchical n not divisible by cols");
      }
      lay->rows = lay->n / lay->cols;
    }
    if (lay->rows * lay->cols > lay->n) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: hierarchical rows×cols > n");
    }
    break;
  }
  default: {
    if (input.shape.empty()) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: scalar input");
    }
    lay->cols = static_cast<int>(input.shape.back());
    if (cfg.dim > 0 && lay->cols != cfg.dim) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: last dim != Dim");
    }
    if (lay->cols <= 0 || lay->n % lay->cols != 0) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: n not divisible by cols");
    }
    lay->rows = lay->n / lay->cols;
    break;
  }
  }
  return Status::Ok();
}

std::vector<float> stable_softmax(const std::vector<float>& logits) {
  if (logits.empty()) return {};
  float max_logit = logits[0];
  for (float v : logits) {
    if (v > max_logit) max_logit = v;
  }
  std::vector<float> out(logits.size());
  float sum_exp = 0.0f;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const float e = static_cast<float>(std::exp(static_cast<double>(logits[i] - max_logit)));
    out[i] = e;
    sum_exp += e;
  }
  for (float& v : out) {
    v /= sum_exp;
  }
  return out;
}

std::vector<float> sparsemax(const std::vector<float>& logits) {
  const int n = static_cast<int>(logits.size());
  std::vector<float> sorted(logits);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (sorted[static_cast<std::size_t>(i)] < sorted[static_cast<std::size_t>(j)]) {
        std::swap(sorted[static_cast<std::size_t>(i)], sorted[static_cast<std::size_t>(j)]);
      }
    }
  }
  float cum_sum = 0.0f;
  int k = 0;
  for (int i = 0; i < n; ++i) {
    cum_sum += sorted[static_cast<std::size_t>(i)];
    if (sorted[static_cast<std::size_t>(i)] - (cum_sum - 1.0f) / static_cast<float>(i + 1) > 0) {
      k = i + 1;
    } else {
      break;
    }
  }
  float tau = 0.0f;
  if (k > 0) {
    cum_sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      cum_sum += sorted[static_cast<std::size_t>(i)];
    }
    tau = (cum_sum - 1.0f) / static_cast<float>(k);
  }
  std::vector<float> result(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    result[static_cast<std::size_t>(i)] = static_cast<float>(
        std::max(0.0, static_cast<double>(logits[static_cast<std::size_t>(i)] - tau)));
  }
  return result;
}

std::vector<float> entmax(const std::vector<float>& logits, float alpha) {
  if (alpha <= 1.0f) return stable_softmax(logits);
  if (alpha >= 2.0f) return sparsemax(logits);
  const float weight = alpha - 1.0f;
  auto s1 = stable_softmax(logits);
  auto s2 = sparsemax(logits);
  std::vector<float> res(logits.size());
  float sum = 0.0f;
  for (std::size_t i = 0; i < res.size(); ++i) {
    res[i] = (1.0f - weight) * s1[i] + weight * s2[i];
    sum += res[i];
  }
  for (float& v : res) {
    v /= sum;
  }
  return res;
}

void apply_mask(std::vector<float>& logits, const std::vector<bool>& mask, int global_start) {
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const int gi = global_start + static_cast<int>(i);
    if (gi < static_cast<int>(mask.size()) && !mask[static_cast<std::size_t>(gi)]) {
      logits[i] = -1e9f;
    }
  }
}

float lcg_float(uint32_t* state) {
  *state = (*state) * 1664525u + 1013904223u;
  return static_cast<float>((*state) >> 8) * (1.0f / 16777216.0f);
}

std::vector<float> gumbel_noise(const std::vector<float>& logits, uint32_t* rng) {
  std::vector<float> noisy(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    float u = lcg_float(rng);
    if (u < 1e-10f) u = 1e-10f;
    const float gumbel = -static_cast<float>(std::log(-std::log(static_cast<double>(u))));
    noisy[i] = logits[i] + gumbel;
  }
  return noisy;
}

std::vector<float> probs_for_kind(const Config& cfg, std::vector<float> logits, int global_start,
                                  uint32_t* gumbel_rng) {
  switch (cfg.kind) {
  case Kind::Gumbel:
    return stable_softmax(gumbel_noise(logits, gumbel_rng));
  case Kind::Masked: {
    apply_mask(logits, cfg.mask, global_start);
    return stable_softmax(logits);
  }
  case Kind::Sparse:
    return sparsemax(logits);
  case Kind::Entmax: {
    float alpha = static_cast<float>(cfg.entmax_alpha);
    if (alpha == 0.0f) alpha = 1.5f;
    return entmax(logits, alpha);
  }
  default:
    return stable_softmax(logits);
  }
}

} // namespace

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "softmax: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: need positive Dim");
  }
  if (cfg->seq_len <= 0) cfg->seq_len = 1;
  if (cfg->temperature == 0.0) cfg->temperature = 1.0;
  if (cfg->temperature < 0.0) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: Temperature < 0");
  }
  switch (cfg->kind) {
  case Kind::Grid:
    if (cfg->cols <= 0) cfg->cols = cfg->dim;
    if (cfg->rows <= 0) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: KindGrid needs positive Rows");
    }
    break;
  case Kind::Hierarchical:
    if (cfg->hierarchy_levels.empty()) {
      return Status::Err(StatusCode::InvalidArgument,
                         "softmax: KindHierarchical needs HierarchyLevels");
    }
    if (cfg->cols <= 0) {
      cfg->cols = cfg->hierarchy_levels.back();
    }
    break;
  default:
    break;
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "softmax: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;
  Layer layer;
  layer.cfg = std::move(cfg);
  layer.dtype = dt;
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st) return st;

  const double inv_t = 1.0 / lay.temp;
  std::vector<float> out(static_cast<std::size_t>(lay.n), 0.0f);
  uint32_t gumbel_rng = cfg.gumbel_seed == 0 ? 0xC0FFEEu : cfg.gumbel_seed;

  for (int r = 0; r < lay.rows; ++r) {
    int start = r * lay.cols;
    int end = start + lay.cols;
    if (end > lay.n) end = lay.n;
    const int cols = end - start;
    std::vector<float> logits(static_cast<std::size_t>(cols));
    for (int i = 0; i < cols; ++i) {
      logits[static_cast<std::size_t>(i)] = static_cast<float>(
          static_cast<double>(input.data[static_cast<std::size_t>(start + i)]) * inv_t);
    }
    auto probs = probs_for_kind(cfg, std::move(logits), start, &gumbel_rng);
    double sum = 0.0;
    for (float p : probs) sum += static_cast<double>(p);
    if (sum == 0.0) {
      return Status::Err(StatusCode::InvalidArgument,
                         std::string("softmax: zero sum at row ") + std::to_string(r));
    }
    for (int i = 0; i < cols; ++i) {
      out[static_cast<std::size_t>(start + i)] = probs[static_cast<std::size_t>(i)];
    }
  }

  pre->shape = lay.shape;
  pre->data = out;
  *post = *pre;
  return Status::Ok();
}

} // namespace welvet::softmax
