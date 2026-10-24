#include "welvet/convt1.hpp"

#include "welvet/activate.hpp"

#include <string>
#include <vector>

namespace welvet::convt1 {
namespace {

Status flatten_w_f32(const WeightStore& ws, std::vector<float>* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "convt1: nil w");
  out->resize(static_cast<std::size_t>(ws.rows * ws.cols));
  std::vector<double> row(static_cast<std::size_t>(ws.cols));
  for (int r = 0; r < ws.rows; ++r) {
    if (auto st = ws.decode_row_f64(r, row.data()); !st) return st;
    for (int c = 0; c < ws.cols; ++c) {
      (*out)[static_cast<std::size_t>(r * ws.cols + c)] =
          static_cast<float>(row[static_cast<std::size_t>(c)]);
    }
  }
  return Status::Ok();
}

struct Layout {
  int batch = 0, in_c = 0, seq = 0, filters = 0, out_len = 0, k = 0, stride = 0, padding = 0;
};

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: empty input");
  }
  if (input.shape.size() != 3) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: need [batch,inC,seq]");
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
    return Status::Err(StatusCode::InvalidArgument, "convt1: shape mismatch");
  }
  if (lay->batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: bad batch");
  }
  const int want = lay->batch * lay->in_c * lay->seq;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: short data");
  }
  return Status::Ok();
}

} // namespace

int output_length(int seq_len, int kernel, int stride, int padding, int output_padding) {
  if (stride <= 0) stride = 1;
  return (seq_len - 1) * stride - 2 * padding + kernel + output_padding;
}

int out_len(const Config& cfg) {
  return output_length(cfg.seq_len, cfg.kernel, cfg.stride, cfg.padding, cfg.output_padding);
}

int patch_dim(const Config& cfg) { return cfg.in_channels * cfg.kernel; }

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "convt1: nil config");
  if (cfg->in_channels <= 0 || cfg->filters <= 0 || cfg->seq_len <= 0 || cfg->kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: need positive dims");
  }
  if (cfg->stride <= 0) cfg->stride = 1;
  if (cfg->padding < 0 || cfg->output_padding < 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: Padding/OutputPadding < 0");
  }
  if (cfg->output_padding >= cfg->stride) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: OutputPadding must be < Stride");
  }
  if (out_len(*cfg) <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: output length <= 0");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* w_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "convt1: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(patch_dim(cfg), cfg.filters, cfg.activation, dt, w_init,
                                 &layer.proj);
      !st) {
    return Status::Err(st.code, std::string("convt1 proj: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  layer.cfg = cfg;

  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st) return st;

  std::vector<float> w;
  if (auto st = flatten_w_f32(layer.proj.weights, &w); !st) {
    return Status::Err(st.code, std::string("convt1 weights: ") + st.message);
  }
  const int want_w = lay.filters * lay.in_c * lay.k;
  if (static_cast<int>(w.size()) < want_w) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: weight short");
  }

  const int n_out = lay.batch * lay.filters * lay.out_len;
  std::vector<double> acc(static_cast<std::size_t>(n_out), 0.0);

  for (int b = 0; b < lay.batch; ++b) {
    for (int ic = 0; ic < lay.in_c; ++ic) {
      for (int iw = 0; iw < lay.seq; ++iw) {
        const double input_val = static_cast<double>(
            input.data[static_cast<std::size_t>(b * lay.in_c * lay.seq + ic * lay.seq + iw)]);
        for (int f = 0; f < lay.filters; ++f) {
          for (int k = 0; k < lay.k; ++k) {
            const int ow = iw * lay.stride - lay.padding + k;
            if (ow < 0 || ow >= lay.out_len) continue;
            const int out_idx = b * lay.filters * lay.out_len + f * lay.out_len + ow;
            const int kw_idx = f * lay.in_c * lay.k + ic * lay.k + k;
            acc[static_cast<std::size_t>(out_idx)] +=
                input_val * static_cast<double>(w[static_cast<std::size_t>(kw_idx)]);
          }
        }
      }
    }
  }

  pre->shape = {lay.batch, lay.filters, lay.out_len};
  post->shape = {lay.batch, lay.filters, lay.out_len};
  pre->data.resize(static_cast<std::size_t>(n_out));
  post->data.resize(static_cast<std::size_t>(n_out));
  for (int i = 0; i < n_out; ++i) {
    pre->data[static_cast<std::size_t>(i)] = static_cast<float>(acc[static_cast<std::size_t>(i)]);
    // Match Go: Activate(pre[i], act) — act on already-cast pre value.
    post->data[static_cast<std::size_t>(i)] = static_cast<float>(
        activate_f64(static_cast<double>(pre->data[static_cast<std::size_t>(i)]), cfg.activation));
  }
  return Status::Ok();
}

} // namespace welvet::convt1
