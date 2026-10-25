#include "welvet/cnn1.hpp"
#include "welvet/metal/layer_utils.hpp"

#include <string>
#include <vector>

namespace welvet::cnn1 {
namespace {

struct Layout {
  int batch = 0, in_c = 0, seq = 0, filters = 0, out_len = 0, k = 0, stride = 0, padding = 0;
};

Status parse_layout(const Config &cfg, const TensorF32 &input, Layout *lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: empty input");
  }
  if (input.shape.size() != 3) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: need [batch,inC,seq]");
  }
  lay->batch = static_cast<int>(input.shape[0]);
  lay->in_c = cfg.in_channels;
  lay->seq = cfg.seq_len;
  lay->filters = cfg.filters;
  lay->out_len = out_len(cfg);
  lay->k = cfg.kernel;
  lay->stride = cfg.stride;
  lay->padding = cfg.padding;
  if (static_cast<int>(input.shape[1]) != lay->in_c ||
      static_cast<int>(input.shape[2]) != lay->seq) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: shape mismatch");
  }
  if (lay->batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: bad batch");
  }
  const int want = lay->batch * lay->in_c * lay->seq;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: short data");
  }
  return Status::Ok();
}

} // namespace

int output_length(int seq_len, int kernel, int stride, int padding) {
  if (stride <= 0)
    stride = 1;
  return (seq_len + 2 * padding - kernel) / stride + 1;
}

int out_len(const Config &cfg) {
  return output_length(cfg.seq_len, cfg.kernel, cfg.stride, cfg.padding);
}

int patch_dim(const Config &cfg) { return cfg.in_channels * cfg.kernel; }

Status validate(Config *cfg) {
  if (!cfg)
    return Status::Err(StatusCode::InvalidArgument, "cnn1: nil config");
  if (cfg->in_channels <= 0 || cfg->filters <= 0 || cfg->seq_len <= 0 || cfg->kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: need positive dims");
  }
  if (cfg->stride <= 0)
    cfg->stride = 1;
  if (cfg->padding < 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: Padding < 0");
  }
  if (out_len(*cfg) <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: output length <= 0");
  }
  return Status::Ok();
}

Status new_configured(const Config &cfg_in, DType dt, const float *w_init, Layer *out) {
  if (!out)
    return Status::Err(StatusCode::InvalidArgument, "cnn1: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st)
    return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st =
          dense::new_layer(patch_dim(cfg), cfg.filters, cfg.activation, dt, w_init, &layer.proj);
      !st) {
    return Status::Err(st.code, std::string("cnn1 proj: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer &layer, const TensorF32 &input, TensorF32 *pre, TensorF32 *post) {
  if (!pre || !post)
    return Status::Err(StatusCode::InvalidArgument, "cnn1: nil output");
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st)
    return st;
  std::vector<float> w;
  if (auto st = metal::decode_xy(layer.proj.weights, &w); !st)
    return st;
  const int count = lay.batch * cfg.filters * 1 * 1 * out_len(cfg);
  std::vector<float> bias(layer.proj.weights.bias.size() * 2);
  for (std::size_t i = 0; i < layer.proj.weights.bias.size(); ++i)
    metal::split_double(layer.proj.weights.bias[i], &bias[2 * i], &bias[2 * i + 1]);
  pre->shape = {lay.batch, cfg.filters, out_len(cfg)};
  post->shape = pre->shape;
  pre->data.resize(count);
  post->data.resize(count);
  return metal::run("op_conv_forward",
                    {metal::read(input.data), metal::read(w), metal::write(pre->data),
                     metal::write(post->data), metal::read(bias)},
                    {lay.batch, cfg.in_channels, cfg.filters, 1, 1, cfg.seq_len, 1, 1, out_len(cfg),
                     cfg.kernel, cfg.stride, cfg.padding, 0, int(cfg.activation), 1,
                     int(layer.proj.weights.bias.size())},
                    {}, count);
}

int grad_w_size(const Layer &layer) { return layer.proj.weights.rows * layer.proj.weights.cols; }

Status backward_f32(Layer &layer, const TensorF32 &grad_out, const TensorF32 &input,
                    const TensorF32 &pre, TensorF32 *grad_in, TensorF32 *grad_w) {
  if (!grad_in || !grad_w)
    return Status::Err(StatusCode::InvalidArgument, "cnn1: nil gradients");
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st)
    return st;
  std::vector<float> w;
  if (auto st = metal::decode_xy(layer.proj.weights, &w); !st)
    return st;
  const int count = lay.batch * cfg.filters * 1 * 1 * out_len(cfg);
  if (grad_out.data.size() < count || pre.data.size() < count)
    return Status::Err(StatusCode::InvalidArgument, "cnn1: gradient/tape shape");
  *grad_in = input;
  *grad_w = TensorF32::zeros(layer.proj.weights.rows, layer.proj.weights.cols);
  if (auto st = metal::run(
          "op_conv_backward_x",
          {metal::read(input.data), metal::read(w), metal::read(grad_out.data),
           metal::read(pre.data), metal::write(grad_in->data), metal::write(grad_w->data)},
          {lay.batch, cfg.in_channels, cfg.filters, 1, 1, cfg.seq_len, 1, 1, out_len(cfg),
           cfg.kernel, cfg.stride, cfg.padding, 0, int(cfg.activation), 1},
          {}, grad_in->data.size());
      !st)
    return st;
  return metal::run("op_conv_backward_w",
                    {metal::read(input.data), metal::read(w), metal::read(grad_out.data),
                     metal::read(pre.data), metal::write(grad_in->data),
                     metal::write(grad_w->data)},
                    {lay.batch, cfg.in_channels, cfg.filters, 1, 1, cfg.seq_len, 1, 1, out_len(cfg),
                     cfg.kernel, cfg.stride, cfg.padding, 0, int(cfg.activation), 1},
                    {}, grad_w->data.size());
}

Status apply_grad_sgd(Layer *layer, const TensorF32 &dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: ApplyGradSGD nil");
  }
  return dense::apply_grad_sgd(&layer->proj, dW, lr);
}

Status export_weights_f32(const Layer &layer, float *dst) {
  if (!dst)
    return Status::Err(StatusCode::InvalidArgument, "cnn1: nil dst");
  return layer.proj.weights.export_f32(dst);
}

} // namespace welvet::cnn1
