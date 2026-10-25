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
  return Status::Err(StatusCode::Unsupported, "rmsnorm.forward: Metal kernel not implemented; CPU execution is forbidden");
}

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "rmsnorm.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "rmsnorm.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

} // namespace welvet::rmsnorm
