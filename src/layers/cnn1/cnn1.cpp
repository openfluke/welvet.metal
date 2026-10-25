#include "welvet/cnn1.hpp"

#include <string>
#include <vector>

namespace welvet::cnn1 {
namespace {

struct Layout {
  int batch = 0, in_c = 0, seq = 0, filters = 0, out_len = 0, k = 0, stride = 0, padding = 0;
};

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
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

TensorF32 im2col(const TensorF32& input, const Layout& lay) {
  const int cols = lay.in_c * lay.k;
  const int rows = lay.batch * lay.out_len;
  TensorF32 out;
  out.shape = {rows, cols};
  out.data.assign(static_cast<std::size_t>(rows * cols), 0.0f);
  for (int b = 0; b < lay.batch; ++b) {
    for (int o = 0; o < lay.out_len; ++o) {
      const int base = (b * lay.out_len + o) * cols;
      for (int ic = 0; ic < lay.in_c; ++ic) {
        for (int k = 0; k < lay.k; ++k) {
          const int in_pos = o * lay.stride + k - lay.padding;
          float v = 0.0f;
          if (in_pos >= 0 && in_pos < lay.seq) {
            v = input.data[static_cast<std::size_t>(b * lay.in_c * lay.seq + ic * lay.seq +
                                                    in_pos)];
          }
          out.data[static_cast<std::size_t>(base + ic * lay.k + k)] = v;
        }
      }
    }
  }
  return out;
}

void loom_from_dense(const std::vector<float>& flat, int batch, int filters, int out_len_n,
                     TensorF32* out) {
  out->shape = {batch, filters, out_len_n};
  out->data.resize(static_cast<std::size_t>(batch * filters * out_len_n));
  for (int b = 0; b < batch; ++b) {
    for (int o = 0; o < out_len_n; ++o) {
      for (int f = 0; f < filters; ++f) {
        out->data[static_cast<std::size_t>(b * filters * out_len_n + f * out_len_n + o)] =
            flat[static_cast<std::size_t>((b * out_len_n + o) * filters + f)];
      }
    }
  }
}

TensorF32 dense_from_loom(const std::vector<float>& loom, int batch, int filters, int out_len_n) {
  TensorF32 out;
  out.shape = {batch * out_len_n, filters};
  out.data.resize(static_cast<std::size_t>(batch * out_len_n * filters));
  for (int b = 0; b < batch; ++b) {
    for (int o = 0; o < out_len_n; ++o) {
      for (int f = 0; f < filters; ++f) {
        out.data[static_cast<std::size_t>((b * out_len_n + o) * filters + f)] =
            loom[static_cast<std::size_t>(b * filters * out_len_n + f * out_len_n + o)];
      }
    }
  }
  return out;
}

TensorF32 col2im(const TensorF32& gx_col, const Layout& lay) {
  const int cols = lay.in_c * lay.k;
  TensorF32 out;
  out.shape = {lay.batch, lay.in_c, lay.seq};
  out.data.assign(static_cast<std::size_t>(lay.batch * lay.in_c * lay.seq), 0.0f);
  std::vector<double> acc(static_cast<std::size_t>(lay.batch * lay.in_c * lay.seq), 0.0);
  for (int b = 0; b < lay.batch; ++b) {
    for (int o = 0; o < lay.out_len; ++o) {
      const int base = (b * lay.out_len + o) * cols;
      for (int ic = 0; ic < lay.in_c; ++ic) {
        for (int k = 0; k < lay.k; ++k) {
          const int in_pos = o * lay.stride + k - lay.padding;
          if (in_pos < 0 || in_pos >= lay.seq) continue;
          acc[static_cast<std::size_t>(b * lay.in_c * lay.seq + ic * lay.seq + in_pos)] +=
              static_cast<double>(
                  gx_col.data[static_cast<std::size_t>(base + ic * lay.k + k)]);
        }
      }
    }
  }
  for (std::size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = static_cast<float>(acc[i]);
  }
  return out;
}

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
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  layer.cfg = cfg;

  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st) return st;

  TensorF32 col = im2col(input, lay);
  layer.proj.activation = cfg.activation;

  TensorF32 pre_flat, post_flat;
  if (auto st = dense::forward_f32(layer.proj, col, &pre_flat, &post_flat); !st) {
    return Status::Err(st.code, std::string("cnn1 dense fwd: ") + st.message);
  }
  loom_from_dense(pre_flat.data, lay.batch, lay.filters, lay.out_len, pre);
  loom_from_dense(post_flat.data, lay.batch, lay.filters, lay.out_len, post);
  return Status::Ok();
}

int grad_w_size(const Layer& layer) {
  return layer.proj.weights.rows * layer.proj.weights.cols;
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: nil grad_in/grad_w");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  layer.cfg = cfg;

  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st) return st;

  const int want_out = lay.batch * lay.filters * lay.out_len;
  if (static_cast<int>(grad_out.data.size()) < want_out ||
      static_cast<int>(pre.data.size()) < want_out) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: grad/pre short");
  }

  TensorF32 col = im2col(input, lay);
  TensorF32 gy = dense_from_loom(grad_out.data, lay.batch, lay.filters, lay.out_len);
  TensorF32 pre_flat = dense_from_loom(pre.data, lay.batch, lay.filters, lay.out_len);

  layer.proj.activation = cfg.activation;

  TensorF32 gx_col;
  if (auto st = dense::backward_f32(layer.proj, gy, col, pre_flat, &gx_col, grad_w); !st) {
    return Status::Err(st.code, std::string("cnn1 dense bwd: ") + st.message);
  }
  *grad_in = col2im(gx_col, lay);
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "cnn1: ApplyGradSGD nil");
  }
  return dense::apply_grad_sgd(&layer->proj, dW, lr);
}

Status export_weights_f32(const Layer& layer, float* dst) {
  if (!dst) return Status::Err(StatusCode::InvalidArgument, "cnn1: nil dst");
  return layer.proj.weights.export_f32(dst);
}

} // namespace welvet::cnn1
