#include "welvet/convt3.hpp"

#include "welvet/activate.hpp"

#include <string>
#include <vector>

namespace welvet::convt3 {
namespace {

struct Layout {
  int batch = 0, in_c = 0, in_d = 0, in_h = 0, in_w = 0, filters = 0, out_d = 0, out_h = 0,
      out_w = 0, k = 0, stride = 0, padding = 0;
};

} // namespace

int output_spatial(int spatial, int kernel, int stride, int padding, int output_padding) {
  if (stride <= 0) stride = 1;
  return (spatial - 1) * stride - 2 * padding + kernel + output_padding;
}

int out_d(const Config& cfg) {
  return output_spatial(cfg.depth, cfg.kernel, cfg.stride, cfg.padding, cfg.output_padding);
}
int out_h(const Config& cfg) {
  return output_spatial(cfg.height, cfg.kernel, cfg.stride, cfg.padding, cfg.output_padding);
}
int out_w(const Config& cfg) {
  return output_spatial(cfg.width, cfg.kernel, cfg.stride, cfg.padding, cfg.output_padding);
}

int patch_dim(const Config& cfg) {
  return cfg.in_channels * cfg.kernel * cfg.kernel * cfg.kernel;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "convt3: nil config");
  if (cfg->in_channels <= 0 || cfg->filters <= 0 || cfg->depth <= 0 || cfg->height <= 0 ||
      cfg->width <= 0 || cfg->kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: need positive dims");
  }
  if (cfg->stride <= 0) cfg->stride = 1;
  if (cfg->padding < 0 || cfg->output_padding < 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: Padding/OutputPadding < 0");
  }
  if (cfg->output_padding >= cfg->stride) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: OutputPadding must be < Stride");
  }
  if (out_d(*cfg) <= 0 || out_h(*cfg) <= 0 || out_w(*cfg) <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: bad output spatial");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* w_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "convt3: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(patch_dim(cfg), cfg.filters, cfg.activation, dt, w_init,
                                 &layer.proj);
      !st) {
    return Status::Err(st.code, std::string("convt3 proj: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "convt3.forward: Metal kernel not implemented; CPU execution is forbidden");
}


int grad_w_size(const Layer& layer) {
  return layer.proj.weights.rows * layer.proj.weights.cols;
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "convt3.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "convt3.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

Status export_weights_f32(const Layer& layer, float* dst) {
  if (!dst) return Status::Err(StatusCode::InvalidArgument, "convt3: nil dst");
  return layer.proj.weights.export_f32(dst);
}

} // namespace welvet::convt3
