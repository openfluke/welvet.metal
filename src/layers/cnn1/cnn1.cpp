#include "welvet/cnn1.hpp"

#include <string>
#include <vector>

namespace welvet::cnn1 {
namespace {

struct Layout {
  int batch = 0, in_c = 0, seq = 0, filters = 0, out_len = 0, k = 0, stride = 0, padding = 0;
};

} // namespace

int output_length(int seq_len, int kernel, int stride, int padding) {
  if (stride <= 0) stride = 1;
  return (seq_len + 2 * padding - kernel) / stride + 1;
}

int out_len(const Config& cfg) {
  return output_length(cfg.seq_len, cfg.kernel, cfg.stride, cfg.padding);
}

int patch_dim(const Config& cfg) { return cfg.in_channels * cfg.kernel; }

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "cnn1: nil config");
  if (cfg->in_channels <= 0 || cfg->filters <= 0 || cfg->seq_len <= 0 || cfg->kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: need positive dims");
  }
  if (cfg->stride <= 0) cfg->stride = 1;
  if (cfg->padding < 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: Padding < 0");
  }
  if (out_len(*cfg) <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: output length <= 0");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* w_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "cnn1: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(patch_dim(cfg), cfg.filters, cfg.activation, dt, w_init,
                                 &layer.proj);
      !st) {
    return Status::Err(st.code, std::string("cnn1 proj: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "cnn1.forward: Metal kernel not implemented; CPU execution is forbidden");
}

int grad_w_size(const Layer& layer) {
  return layer.proj.weights.rows * layer.proj.weights.cols;
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "cnn1.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "cnn1.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

Status export_weights_f32(const Layer& layer, float* dst) {
  if (!dst) return Status::Err(StatusCode::InvalidArgument, "cnn1: nil dst");
  return layer.proj.weights.export_f32(dst);
}

} // namespace welvet::cnn1
