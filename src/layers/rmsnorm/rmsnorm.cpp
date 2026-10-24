#include "welvet/rmsnorm.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::rmsnorm {
namespace {

struct Layout {
  int batch = 1;
  int seq = 1;
  int dim = 0;
};

Status parse_layout(int dim, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "rmsnorm: empty input");
  }
  lay->dim = dim;
  if (input.shape.size() == 2) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = 1;
    if (static_cast<int>(input.shape[1]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "rmsnorm: width != Dim");
    }
  } else if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "rmsnorm: last dim != Dim");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument,
                       "rmsnorm: need [batch,dim] or [batch,seq,dim]");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "rmsnorm: bad batch/seq");
  }
  const int want = lay->batch * lay->seq * dim;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "rmsnorm: short data");
  }
  return Status::Ok();
}

} // namespace

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "rmsnorm: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "rmsnorm: need Dim>0");
  }
  if (cfg->eps <= 0.0) {
    cfg->eps = 1e-6;
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* gamma_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "rmsnorm: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = WeightStore::make(1, cfg.dim, dt, gamma_init, &layer.gamma); !st) {
    return Status::Err(st.code, std::string("rmsnorm gamma: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "rmsnorm: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  layer.cfg = cfg;

  Layout lay;
  if (auto st = parse_layout(cfg.dim, input, &lay); !st) return st;

  const int dim = cfg.dim;
  const double eps = cfg.eps;
  const int n_tok = lay.batch * lay.seq;

  std::vector<double> g(static_cast<std::size_t>(dim));
  if (auto st = layer.gamma.decode_row_f64(0, g.data()); !st) {
    return Status::Err(st.code, std::string("rmsnorm gamma: ") + st.message);
  }

  if (lay.seq == 1) {
    pre->shape = {lay.batch, dim};
    post->shape = {lay.batch, dim};
  } else {
    pre->shape = {lay.batch, lay.seq, dim};
    post->shape = {lay.batch, lay.seq, dim};
  }
  pre->data.resize(static_cast<std::size_t>(n_tok * dim));
  post->data.resize(static_cast<std::size_t>(n_tok * dim));

  // Match Go CPU path: x̂ in f64, then FromFloat64; y = FromFloat64(x̂_f64 * g).
  for (int t = 0; t < n_tok; ++t) {
    const int base = t * dim;
    double sum_sq = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = static_cast<double>(input.data[static_cast<std::size_t>(base + i)]);
      sum_sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + eps);
    for (int i = 0; i < dim; ++i) {
      const double xh =
          static_cast<double>(input.data[static_cast<std::size_t>(base + i)]) * inv;
      pre->data[static_cast<std::size_t>(base + i)] = static_cast<float>(xh);
      post->data[static_cast<std::size_t>(base + i)] =
          static_cast<float>(xh * g[static_cast<std::size_t>(i)]);
    }
  }
  return Status::Ok();
}

} // namespace welvet::rmsnorm
