#include "welvet/layernorm.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::layernorm {
namespace {

struct Layout {
  int batch = 1;
  int seq = 1;
  int dim = 0;
};

Status parse_layout(int dim, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: empty input");
  }
  lay->dim = dim;
  if (input.shape.size() == 2) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = 1;
    if (static_cast<int>(input.shape[1]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "layernorm: width != Dim");
    }
  } else if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "layernorm: last dim != Dim");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument,
                       "layernorm: need [batch,dim] or [batch,seq,dim]");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: bad batch/seq");
  }
  const int want = lay->batch * lay->seq * dim;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: short data");
  }
  return Status::Ok();
}

} // namespace

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "layernorm: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: need Dim>0");
  }
  if (cfg->eps <= 0.0) {
    cfg->eps = 1e-5;
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* gamma_init,
                      const float* beta_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "layernorm: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = WeightStore::make(1, cfg.dim, dt, gamma_init, &layer.gamma); !st) {
    return Status::Err(st.code, std::string("layernorm gamma: ") + st.message);
  }
  if (auto st = WeightStore::make(1, cfg.dim, dt, beta_init, &layer.beta); !st) {
    return Status::Err(st.code, std::string("layernorm beta: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: nil pre/post");
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
  std::vector<double> b(static_cast<std::size_t>(dim));
  if (auto st = layer.gamma.decode_row_f64(0, g.data()); !st) {
    return Status::Err(st.code, std::string("layernorm gamma: ") + st.message);
  }
  if (auto st = layer.beta.decode_row_f64(0, b.data()); !st) {
    return Status::Err(st.code, std::string("layernorm beta: ") + st.message);
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

  // Match Go CPU: mean/var in f64; x̂ then y = x̂*g+b before FromFloat64.
  for (int t = 0; t < n_tok; ++t) {
    const int base = t * dim;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = static_cast<double>(input.data[static_cast<std::size_t>(base + i)]);
      sum += v;
      sum_sq += v * v;
    }
    const double mean = sum / static_cast<double>(dim);
    const double var_ = sum_sq / static_cast<double>(dim) - mean * mean;
    const double inv = 1.0 / std::sqrt(var_ + eps);
    for (int i = 0; i < dim; ++i) {
      const double xh =
          (static_cast<double>(input.data[static_cast<std::size_t>(base + i)]) - mean) * inv;
      pre->data[static_cast<std::size_t>(base + i)] = static_cast<float>(xh);
      post->data[static_cast<std::size_t>(base + i)] = static_cast<float>(
          xh * g[static_cast<std::size_t>(i)] + b[static_cast<std::size_t>(i)]);
    }
  }
  return Status::Ok();
}

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: nil grad out");
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
    return Status::Err(StatusCode::InvalidArgument, "layernorm: pre (x̂) or gradOut short");
  }

  std::vector<double> g(static_cast<std::size_t>(dim));
  if (auto st = layer.gamma.decode_row_f64(0, g.data()); !st) {
    return Status::Err(st.code, std::string("layernorm gamma: ") + st.message);
  }

  std::vector<double> d_gamma(static_cast<std::size_t>(dim), 0.0);
  std::vector<double> d_beta(static_cast<std::size_t>(dim), 0.0);
  if (lay.seq == 1) {
    grad_in->shape = {lay.batch, dim};
  } else {
    grad_in->shape = {lay.batch, lay.seq, dim};
  }
  grad_in->data.assign(static_cast<std::size_t>(n_tok * dim), 0.0f);
  *grad_w = TensorF32::zeros(1, 2 * dim);

  for (int t = 0; t < n_tok; ++t) {
    const int base = t * dim;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = static_cast<double>(input.data[static_cast<std::size_t>(base + i)]);
      sum += v;
      sum_sq += v * v;
    }
    const double mean = sum / static_cast<double>(dim);
    const double var_ = sum_sq / static_cast<double>(dim) - mean * mean;
    const double inv = 1.0 / std::sqrt(var_ + eps);

    std::vector<double> dxh(static_cast<std::size_t>(dim));
    double sum_dxh = 0.0;
    double sum_dxh_xh = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double xh = static_cast<double>(pre.data[static_cast<std::size_t>(base + i)]);
      const double d = static_cast<double>(grad_out.data[static_cast<std::size_t>(base + i)]);
      d_gamma[static_cast<std::size_t>(i)] += d * xh;
      d_beta[static_cast<std::size_t>(i)] += d;
      dxh[static_cast<std::size_t>(i)] = g[static_cast<std::size_t>(i)] * d;
      sum_dxh += dxh[static_cast<std::size_t>(i)];
      sum_dxh_xh += dxh[static_cast<std::size_t>(i)] * xh;
    }
    const double n = static_cast<double>(dim);
    for (int i = 0; i < dim; ++i) {
      const double xh = static_cast<double>(pre.data[static_cast<std::size_t>(base + i)]);
      grad_in->data[static_cast<std::size_t>(base + i)] = static_cast<float>(
          inv / n * (n * dxh[static_cast<std::size_t>(i)] - sum_dxh - xh * sum_dxh_xh));
    }
  }
  for (int i = 0; i < dim; ++i) {
    grad_w->data[static_cast<std::size_t>(i)] =
        static_cast<float>(d_gamma[static_cast<std::size_t>(i)]);
    grad_w->data[static_cast<std::size_t>(dim + i)] =
        static_cast<float>(d_beta[static_cast<std::size_t>(i)]);
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: ApplyGradSGD nil");
  }
  const int dim = layer->cfg.dim;
  if (static_cast<int>(dW.data.size()) < 2 * dim) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: dW short");
  }
  std::vector<double> g_buf(static_cast<std::size_t>(dim));
  std::vector<double> b_buf(static_cast<std::size_t>(dim));
  for (int i = 0; i < dim; ++i) {
    g_buf[static_cast<std::size_t>(i)] = static_cast<double>(dW.data[static_cast<std::size_t>(i)]);
    b_buf[static_cast<std::size_t>(i)] =
        static_cast<double>(dW.data[static_cast<std::size_t>(dim + i)]);
  }
  if (auto st = layer->gamma.apply_sgd(g_buf.data(), lr); !st) return st;
  return layer->beta.apply_sgd(b_buf.data(), lr);
}

} // namespace welvet::layernorm
