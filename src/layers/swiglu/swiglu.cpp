#include "welvet/swiglu.hpp"

#include "welvet/activate.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace welvet::swiglu {
namespace {

struct Layout {
  int batch = 1;
  int seq = 1;
  int in = 0;
};

} // namespace

Status validate(const Config& cfg) {
  if (cfg.input_dim <= 0 || cfg.intermediate_dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: need InputDim>0 and IntermediateDim>0");
  }
  return Status::Ok();
}

Config default_ffn(int d_model) {
  int inter = (d_model * 8) / 3;
  inter = (inter + 7) / 8 * 8;
  if (inter < d_model) inter = d_model * 2;
  return Config{d_model, inter};
}

Status new_configured(const Config& cfg, DType dt, const float* gate_init, const float* up_init,
                      const float* down_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "swiglu: nil out");
  if (auto st = validate(cfg); !st) return st;

  const int in = cfg.input_dim;
  const int inter = cfg.intermediate_dim;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(in, inter, Activation::Linear, dt, gate_init, &layer.Gate); !st) {
    return Status::Err(st.code, std::string("swiglu Gate: ") + st.message);
  }
  if (auto st = dense::new_layer(in, inter, Activation::Linear, dt, up_init, &layer.Up); !st) {
    return Status::Err(st.code, std::string("swiglu Up: ") + st.message);
  }
  if (auto st = dense::new_layer(inter, in, Activation::Linear, dt, down_init, &layer.Down); !st) {
    return Status::Err(st.code, std::string("swiglu Down: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "swiglu.forward: Metal kernel not implemented; CPU execution is forbidden");
}

int grad_w_size(const Layer& layer) {
  return layer.Gate.weights.rows * layer.Gate.weights.cols +
         layer.Up.weights.rows * layer.Up.weights.cols +
         layer.Down.weights.rows * layer.Down.weights.cols;
}

Status export_weights_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: nil export");
  }
  int off = 0;
  if (auto st = layer.Gate.weights.export_f32(dst + off); !st) return st;
  off += layer.Gate.weights.rows * layer.Gate.weights.cols;
  if (auto st = layer.Up.weights.export_f32(dst + off); !st) return st;
  off += layer.Up.weights.rows * layer.Up.weights.cols;
  return layer.Down.weights.export_f32(dst + off);
}

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "swiglu.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "swiglu.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

} // namespace welvet::swiglu
