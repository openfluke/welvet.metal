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
  return Status::Err(StatusCode::Unsupported, "softmax.forward: Metal kernel not implemented; CPU execution is forbidden");
}

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& /*input*/,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "softmax.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& /*dW*/, double /*lr*/) {
  return Status::Err(StatusCode::Unsupported, "softmax.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

} // namespace welvet::softmax
