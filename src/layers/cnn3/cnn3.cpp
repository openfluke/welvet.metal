#include "welvet/cnn3.hpp"

#include <string>
#include <vector>

namespace welvet::cnn3 {
namespace {

struct Layout {
  int batch = 0, in_c = 0, in_d = 0, in_h = 0, in_w = 0, filters = 0, out_d = 0, out_h = 0,
      out_w = 0, k = 0, stride = 0, padding = 0;
};

int spatial_n(const Layout& lay) { return lay.out_d * lay.out_h * lay.out_w; }

int in_idx(const Layout& lay, int b, int ic, int id, int ih, int iw) {
  return b * lay.in_c * lay.in_d * lay.in_h * lay.in_w + ic * lay.in_d * lay.in_h * lay.in_w +
         id * lay.in_h * lay.in_w + ih * lay.in_w + iw;
}

int patch_off(const Layout& lay, int ic, int kd, int kh, int kw) {
  return ic * lay.k * lay.k * lay.k + kd * lay.k * lay.k + kh * lay.k + kw;
}

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: empty input");
  }
  if (input.shape.size() != 5) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: need [batch,inC,D,H,W]");
  }
  lay->batch = static_cast<int>(input.shape[0]);
  lay->in_c = cfg.in_channels;
  lay->in_d = cfg.depth;
  lay->in_h = cfg.height;
  lay->in_w = cfg.width;
  lay->filters = cfg.filters;
  lay->out_d = out_d(cfg);
  lay->out_h = out_h(cfg);
  lay->out_w = out_w(cfg);
  lay->k = cfg.kernel;
  lay->stride = cfg.stride;
  lay->padding = cfg.padding;
  if (static_cast<int>(input.shape[1]) != lay->in_c ||
      static_cast<int>(input.shape[2]) != lay->in_d ||
      static_cast<int>(input.shape[3]) != lay->in_h ||
      static_cast<int>(input.shape[4]) != lay->in_w) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: shape mismatch");
  }
  if (lay->batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: bad batch");
  }
  const int want = lay->batch * lay->in_c * lay->in_d * lay->in_h * lay->in_w;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: short data");
  }
  return Status::Ok();
}

TensorF32 im2col(const TensorF32& input, const Layout& lay) {
  const int cols = lay.in_c * lay.k * lay.k * lay.k;
  const int rows = lay.batch * spatial_n(lay);
  TensorF32 out;
  out.shape = {rows, cols};
  out.data.assign(static_cast<std::size_t>(rows * cols), 0.0f);
  for (int b = 0; b < lay.batch; ++b) {
    for (int od = 0; od < lay.out_d; ++od) {
      for (int oh = 0; oh < lay.out_h; ++oh) {
        for (int ow = 0; ow < lay.out_w; ++ow) {
          const int base =
              (b * spatial_n(lay) + od * lay.out_h * lay.out_w + oh * lay.out_w + ow) * cols;
          for (int ic = 0; ic < lay.in_c; ++ic) {
            for (int kd = 0; kd < lay.k; ++kd) {
              for (int kh = 0; kh < lay.k; ++kh) {
                for (int kw = 0; kw < lay.k; ++kw) {
                  const int id = od * lay.stride + kd - lay.padding;
                  const int ih = oh * lay.stride + kh - lay.padding;
                  const int iw = ow * lay.stride + kw - lay.padding;
                  float v = 0.0f;
                  if (id >= 0 && id < lay.in_d && ih >= 0 && ih < lay.in_h && iw >= 0 &&
                      iw < lay.in_w) {
                    v = input.data[static_cast<std::size_t>(in_idx(lay, b, ic, id, ih, iw))];
                  }
                  out.data[static_cast<std::size_t>(base + patch_off(lay, ic, kd, kh, kw))] = v;
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

void loom_from_dense(const std::vector<float>& flat, int batch, int filters, int out_d_n,
                     int out_h_n, int out_w_n, TensorF32* out) {
  const int sp = out_d_n * out_h_n * out_w_n;
  out->shape = {batch, filters, out_d_n, out_h_n, out_w_n};
  out->data.resize(static_cast<std::size_t>(batch * filters * sp));
  for (int b = 0; b < batch; ++b) {
    for (int od = 0; od < out_d_n; ++od) {
      for (int oh = 0; oh < out_h_n; ++oh) {
        for (int ow = 0; ow < out_w_n; ++ow) {
          const int si = od * out_h_n * out_w_n + oh * out_w_n + ow;
          for (int f = 0; f < filters; ++f) {
            out->data[static_cast<std::size_t>(b * filters * sp + f * sp + si)] =
                flat[static_cast<std::size_t>((b * sp + si) * filters + f)];
          }
        }
      }
    }
  }
}

TensorF32 dense_from_loom(const std::vector<float>& loom, int batch, int filters, int out_d_n,
                          int out_h_n, int out_w_n) {
  const int sp = out_d_n * out_h_n * out_w_n;
  TensorF32 out;
  out.shape = {batch * sp, filters};
  out.data.resize(static_cast<std::size_t>(batch * sp * filters));
  for (int b = 0; b < batch; ++b) {
    for (int od = 0; od < out_d_n; ++od) {
      for (int oh = 0; oh < out_h_n; ++oh) {
        for (int ow = 0; ow < out_w_n; ++ow) {
          const int si = od * out_h_n * out_w_n + oh * out_w_n + ow;
          for (int f = 0; f < filters; ++f) {
            out.data[static_cast<std::size_t>((b * sp + si) * filters + f)] =
                loom[static_cast<std::size_t>(b * filters * sp + f * sp + si)];
          }
        }
      }
    }
  }
  return out;
}

TensorF32 col2im(const TensorF32& gx_col, const Layout& lay) {
  const int cols = lay.in_c * lay.k * lay.k * lay.k;
  TensorF32 out;
  out.shape = {lay.batch, lay.in_c, lay.in_d, lay.in_h, lay.in_w};
  out.data.assign(
      static_cast<std::size_t>(lay.batch * lay.in_c * lay.in_d * lay.in_h * lay.in_w), 0.0f);
  std::vector<double> acc(out.data.size(), 0.0);
  for (int b = 0; b < lay.batch; ++b) {
    for (int od = 0; od < lay.out_d; ++od) {
      for (int oh = 0; oh < lay.out_h; ++oh) {
        for (int ow = 0; ow < lay.out_w; ++ow) {
          const int base =
              (b * spatial_n(lay) + od * lay.out_h * lay.out_w + oh * lay.out_w + ow) * cols;
          for (int ic = 0; ic < lay.in_c; ++ic) {
            for (int kd = 0; kd < lay.k; ++kd) {
              for (int kh = 0; kh < lay.k; ++kh) {
                for (int kw = 0; kw < lay.k; ++kw) {
                  const int id = od * lay.stride + kd - lay.padding;
                  const int ih = oh * lay.stride + kh - lay.padding;
                  const int iw = ow * lay.stride + kw - lay.padding;
                  if (id < 0 || id >= lay.in_d || ih < 0 || ih >= lay.in_h || iw < 0 ||
                      iw >= lay.in_w) {
                    continue;
                  }
                  acc[static_cast<std::size_t>(in_idx(lay, b, ic, id, ih, iw))] +=
                      static_cast<double>(gx_col.data[static_cast<std::size_t>(
                          base + patch_off(lay, ic, kd, kh, kw))]);
                }
              }
            }
          }
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

int output_spatial(int spatial, int kernel, int stride, int padding) {
  if (stride <= 0) stride = 1;
  return (spatial + 2 * padding - kernel) / stride + 1;
}

int out_d(const Config& cfg) {
  return output_spatial(cfg.depth, cfg.kernel, cfg.stride, cfg.padding);
}
int out_h(const Config& cfg) {
  return output_spatial(cfg.height, cfg.kernel, cfg.stride, cfg.padding);
}
int out_w(const Config& cfg) {
  return output_spatial(cfg.width, cfg.kernel, cfg.stride, cfg.padding);
}

int patch_dim(const Config& cfg) {
  return cfg.in_channels * cfg.kernel * cfg.kernel * cfg.kernel;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "cnn3: nil config");
  if (cfg->in_channels <= 0 || cfg->filters <= 0 || cfg->depth <= 0 || cfg->height <= 0 ||
      cfg->width <= 0 || cfg->kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: need positive dims");
  }
  if (cfg->stride <= 0) cfg->stride = 1;
  if (cfg->padding < 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: Padding < 0");
  }
  if (out_d(*cfg) <= 0 || out_h(*cfg) <= 0 || out_w(*cfg) <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: bad output spatial");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* w_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "cnn3: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(patch_dim(cfg), cfg.filters, cfg.activation, dt, w_init,
                                 &layer.proj);
      !st) {
    return Status::Err(st.code, std::string("cnn3 proj: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: nil pre/post");
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
    return Status::Err(st.code, std::string("cnn3 dense fwd: ") + st.message);
  }
  loom_from_dense(pre_flat.data, lay.batch, lay.filters, lay.out_d, lay.out_h, lay.out_w, pre);
  loom_from_dense(post_flat.data, lay.batch, lay.filters, lay.out_d, lay.out_h, lay.out_w, post);
  return Status::Ok();
}

int grad_w_size(const Layer& layer) {
  return layer.proj.weights.rows * layer.proj.weights.cols;
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: nil grad_in/grad_w");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  layer.cfg = cfg;

  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st) return st;

  const int want_out = lay.batch * lay.filters * lay.out_d * lay.out_h * lay.out_w;
  if (static_cast<int>(grad_out.data.size()) < want_out ||
      static_cast<int>(pre.data.size()) < want_out) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: grad/pre short");
  }

  TensorF32 col = im2col(input, lay);
  TensorF32 gy =
      dense_from_loom(grad_out.data, lay.batch, lay.filters, lay.out_d, lay.out_h, lay.out_w);
  TensorF32 pre_flat =
      dense_from_loom(pre.data, lay.batch, lay.filters, lay.out_d, lay.out_h, lay.out_w);

  layer.proj.activation = cfg.activation;

  TensorF32 gx_col;
  if (auto st = dense::backward_f32(layer.proj, gy, col, pre_flat, &gx_col, grad_w); !st) {
    return Status::Err(st.code, std::string("cnn3 dense bwd: ") + st.message);
  }
  *grad_in = col2im(gx_col, lay);
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "cnn3: ApplyGradSGD nil");
  }
  return dense::apply_grad_sgd(&layer->proj, dW, lr);
}

Status export_weights_f32(const Layer& layer, float* dst) {
  if (!dst) return Status::Err(StatusCode::InvalidArgument, "cnn3: nil dst");
  return layer.proj.weights.export_f32(dst);
}

} // namespace welvet::cnn3
