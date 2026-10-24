#include "welvet/cnn2.hpp"

#include <string>
#include <vector>

namespace welvet::cnn2 {
namespace {

struct Layout {
  int batch = 0, in_c = 0, in_h = 0, in_w = 0, filters = 0, out_h = 0, out_w = 0, k = 0,
      stride = 0, padding = 0;
};

int spatial_n(const Layout& lay) { return lay.out_h * lay.out_w; }

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "cnn2: empty input");
  }
  if (input.shape.size() != 4) {
    return Status::Err(StatusCode::InvalidArgument, "cnn2: need [batch,inC,H,W]");
  }
  lay->batch = static_cast<int>(input.shape[0]);
  lay->in_c = cfg.in_channels;
  lay->in_h = cfg.height;
  lay->in_w = cfg.width;
  lay->filters = cfg.filters;
  lay->out_h = out_h(cfg);
  lay->out_w = out_w(cfg);
  lay->k = cfg.kernel;
  lay->stride = cfg.stride;
  lay->padding = cfg.padding;
  if (static_cast<int>(input.shape[1]) != lay->in_c ||
      static_cast<int>(input.shape[2]) != lay->in_h ||
      static_cast<int>(input.shape[3]) != lay->in_w) {
    return Status::Err(StatusCode::InvalidArgument, "cnn2: shape mismatch");
  }
  if (lay->batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn2: bad batch");
  }
  const int want = lay->batch * lay->in_c * lay->in_h * lay->in_w;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "cnn2: short data");
  }
  return Status::Ok();
}

TensorF32 im2col(const TensorF32& input, const Layout& lay) {
  const int cols = lay.in_c * lay.k * lay.k;
  const int rows = lay.batch * spatial_n(lay);
  TensorF32 out;
  out.shape = {rows, cols};
  out.data.assign(static_cast<std::size_t>(rows * cols), 0.0f);
  for (int b = 0; b < lay.batch; ++b) {
    for (int oh = 0; oh < lay.out_h; ++oh) {
      for (int ow = 0; ow < lay.out_w; ++ow) {
        const int base = (b * spatial_n(lay) + oh * lay.out_w + ow) * cols;
        for (int ic = 0; ic < lay.in_c; ++ic) {
          for (int kh = 0; kh < lay.k; ++kh) {
            for (int kw = 0; kw < lay.k; ++kw) {
              const int ih = oh * lay.stride + kh - lay.padding;
              const int iw = ow * lay.stride + kw - lay.padding;
              float v = 0.0f;
              if (ih >= 0 && ih < lay.in_h && iw >= 0 && iw < lay.in_w) {
                v = input.data[static_cast<std::size_t>(
                    b * lay.in_c * lay.in_h * lay.in_w + ic * lay.in_h * lay.in_w +
                    ih * lay.in_w + iw)];
              }
              out.data[static_cast<std::size_t>(base + ic * lay.k * lay.k + kh * lay.k + kw)] =
                  v;
            }
          }
        }
      }
    }
  }
  return out;
}

void loom_from_dense(const std::vector<float>& flat, int batch, int filters, int out_h_n,
                     int out_w_n, TensorF32* out) {
  const int sp = out_h_n * out_w_n;
  out->shape = {batch, filters, out_h_n, out_w_n};
  out->data.resize(static_cast<std::size_t>(batch * filters * sp));
  for (int b = 0; b < batch; ++b) {
    for (int oh = 0; oh < out_h_n; ++oh) {
      for (int ow = 0; ow < out_w_n; ++ow) {
        for (int f = 0; f < filters; ++f) {
          out->data[static_cast<std::size_t>(b * filters * sp + f * sp + oh * out_w_n + ow)] =
              flat[static_cast<std::size_t>((b * sp + oh * out_w_n + ow) * filters + f)];
        }
      }
    }
  }
}



} // namespace

int output_spatial(int spatial, int kernel, int stride, int padding) {
  if (stride <= 0) stride = 1;
  return (spatial + 2 * padding - kernel) / stride + 1;
}

int out_h(const Config& cfg) {
  return output_spatial(cfg.height, cfg.kernel, cfg.stride, cfg.padding);
}

int out_w(const Config& cfg) {
  return output_spatial(cfg.width, cfg.kernel, cfg.stride, cfg.padding);
}

int patch_dim(const Config& cfg) { return cfg.in_channels * cfg.kernel * cfg.kernel; }

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "cnn2: nil config");
  if (cfg->in_channels <= 0 || cfg->filters <= 0 || cfg->height <= 0 || cfg->width <= 0 ||
      cfg->kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn2: need positive dims");
  }
  if (cfg->stride <= 0) cfg->stride = 1;
  if (cfg->padding < 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn2: Padding < 0");
  }
  if (out_h(*cfg) <= 0 || out_w(*cfg) <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cnn2: bad output spatial");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* w_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "cnn2: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(patch_dim(cfg), cfg.filters, cfg.activation, dt, w_init,
                                 &layer.proj);
      !st) {
    return Status::Err(st.code, std::string("cnn2 proj: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "cnn2: nil pre/post");
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
    return Status::Err(st.code, std::string("cnn2 dense fwd: ") + st.message);
  }
  loom_from_dense(pre_flat.data, lay.batch, lay.filters, lay.out_h, lay.out_w, pre);
  loom_from_dense(post_flat.data, lay.batch, lay.filters, lay.out_h, lay.out_w, post);
  return Status::Ok();
}

} // namespace welvet::cnn2
