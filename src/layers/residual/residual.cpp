#include "welvet/residual.hpp"

#include <string>

namespace welvet::residual {
namespace {

struct Layout {
  int batch = 0;
  int seq = 0;
  int in = 0;
};

} // namespace

int weight_count(const Config& cfg) {
  int d = cfg.depth;
  if (d <= 0) d = 1;
  return d * cfg.dim * cfg.dim;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "residual: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "residual: need positive Dim");
  }
  if (cfg->seq_len <= 0) cfg->seq_len = 1;
  if (cfg->depth <= 0) cfg->depth = 1;
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, Activation child_act,
                      const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "residual: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  const int dim = cfg.dim;
  const int depth = cfg.depth;
  const int child_n = dim * dim;
  Layer layer;
  layer.cfg = cfg;
  layer.children.resize(static_cast<std::size_t>(depth));
  for (int i = 0; i < depth; ++i) {
    const float* init = nullptr;
    if (packed_or_null) {
      init = packed_or_null + i * child_n;
    }
    if (auto st = dense::new_layer(dim, dim, child_act, dt, init,
                                   &layer.children[static_cast<std::size_t>(i)]);
        !st) {
      return Status::Err(st.code, std::string("residual F child ") + std::to_string(i) + ": " +
                                      st.message);
    }
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "residual.forward: Metal kernel not implemented; CPU execution is forbidden");
}

int grad_w_size(const Layer& layer) {
  int n = 0;
  for (const auto& ch : layer.children) {
    n += ch.weights.rows * ch.weights.cols;
  }
  return n;
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& /*pre*/, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "residual.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "residual.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "residual: nil dst");
  }
  int off = 0;
  for (const auto& ch : layer.children) {
    const int cn = ch.weights.rows * ch.weights.cols;
    if (auto st = ch.weights.export_f32(dst + off); !st) return st;
    off += cn;
  }
  return Status::Ok();
}

} // namespace welvet::residual
