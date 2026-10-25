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

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "rmsnorm: nil grad out");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(cfg.dim, input, &lay); !st) return st;
  const int dim = cfg.dim;
  const double eps = cfg.eps;
  const int n_tok = lay.batch * lay.seq;
  if (static_cast<int>(pre.data.size()) < n_tok * dim ||
      static_cast<int>(grad_out.data.size()) < n_tok * dim) {
    return Status::Err(StatusCode::InvalidArgument, "rmsnorm: pre (x̂) or gradOut short");
  }

  std::vector<double> g(static_cast<std::size_t>(dim));
  if (auto st = layer.gamma.decode_row_f64(0, g.data()); !st) {
    return Status::Err(st.code, std::string("rmsnorm gamma: ") + st.message);
  }

  std::vector<double> d_gamma(static_cast<std::size_t>(dim), 0.0);
  if (lay.seq == 1) {
    grad_in->shape = {lay.batch, dim};
  } else {
    grad_in->shape = {lay.batch, lay.seq, dim};
  }
  grad_in->data.assign(static_cast<std::size_t>(n_tok * dim), 0.0f);
  *grad_w = TensorF32::zeros(1, dim);

  for (int t = 0; t < n_tok; ++t) {
    const int base = t * dim;
    double sum_sq = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = static_cast<double>(input.data[static_cast<std::size_t>(base + i)]);
      sum_sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + eps);

    double mean = 0.0;
    std::vector<double> u(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) {
      const double xh = static_cast<double>(pre.data[static_cast<std::size_t>(base + i)]);
      const double d = static_cast<double>(grad_out.data[static_cast<std::size_t>(base + i)]);
      d_gamma[static_cast<std::size_t>(i)] += d * xh;
      u[static_cast<std::size_t>(i)] = g[static_cast<std::size_t>(i)] * d;
      mean += xh * u[static_cast<std::size_t>(i)];
    }
    mean /= static_cast<double>(dim);
    for (int i = 0; i < dim; ++i) {
      const double xh = static_cast<double>(pre.data[static_cast<std::size_t>(base + i)]);
      grad_in->data[static_cast<std::size_t>(base + i)] =
          static_cast<float>(inv * (u[static_cast<std::size_t>(i)] - xh * mean));
    }
  }
  for (int i = 0; i < dim; ++i) {
    grad_w->data[static_cast<std::size_t>(i)] =
        static_cast<float>(d_gamma[static_cast<std::size_t>(i)]);
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "rmsnorm: ApplyGradSGD nil");
  }
  const int n = layer->gamma.rows * layer->gamma.cols;
  if (static_cast<int>(dW.data.size()) < n) {
    return Status::Err(StatusCode::InvalidArgument, "rmsnorm: dW short");
  }
  std::vector<double> buf(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    buf[static_cast<std::size_t>(i)] = static_cast<double>(dW.data[static_cast<std::size_t>(i)]);
  }
  return layer->gamma.apply_sgd(buf.data(), lr);
}

} // namespace welvet::rmsnorm
