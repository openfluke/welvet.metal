#include "welvet/rnn.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace welvet::rnn {
namespace {

struct Layout {
  int batch = 0, seq = 0, in = 0, hid = 0;
};

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: empty input");
  }
  if (input.shape.size() != 3) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: need [batch,seq,in]");
  }
  lay->batch = static_cast<int>(input.shape[0]);
  lay->seq = cfg.seq_len;
  lay->in = cfg.input_size;
  lay->hid = cfg.hidden_size;
  if (static_cast<int>(input.shape[1]) != lay->seq ||
      static_cast<int>(input.shape[2]) != lay->in) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: shape mismatch");
  }
  if (lay->batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: bad batch");
  }
  const int want = lay->batch * lay->seq * lay->in;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: short data");
  }
  return Status::Ok();
}

TensorF32 x_at(const TensorF32& input, const Layout& lay, int t) {
  TensorF32 xt;
  xt.shape = {lay.batch, lay.in};
  xt.data.resize(static_cast<std::size_t>(lay.batch * lay.in));
  for (int b = 0; b < lay.batch; ++b) {
    const int src = b * lay.seq * lay.in + t * lay.in;
    for (int i = 0; i < lay.in; ++i) {
      xt.data[static_cast<std::size_t>(b * lay.in + i)] =
          input.data[static_cast<std::size_t>(src + i)];
    }
  }
  return xt;
}

TensorF32 h_tensor(const std::vector<float>& h_prev, int batch, int hid) {
  TensorF32 ht;
  ht.shape = {batch, hid};
  ht.data.assign(h_prev.begin(), h_prev.begin() + static_cast<std::ptrdiff_t>(batch * hid));
  return ht;
}

void sync_projs(Layer& layer) {
  layer.IH.activation = Activation::Linear;
  layer.HH.activation = Activation::Linear;
}

} // namespace

int weight_count(const Config& cfg) {
  const int h = cfg.hidden_size;
  const int in = cfg.input_size;
  return h * in + h * h + h;
}

Status validate(const Config& cfg) {
  if (cfg.input_size <= 0 || cfg.hidden_size <= 0 || cfg.seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: need positive Input/Hidden/SeqLen");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg, DType dt, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "rnn: nil out");
  if (auto st = validate(cfg); !st) return st;

  const int in = cfg.input_size;
  const int h = cfg.hidden_size;
  const int ih_n = h * in;
  const int hh_n = h * h;

  const float* ih_init = nullptr;
  const float* hh_init = nullptr;
  std::vector<double> bias(static_cast<std::size_t>(h), 0.0);
  if (packed_or_null) {
    ih_init = packed_or_null;
    hh_init = packed_or_null + ih_n;
    for (int i = 0; i < h; ++i) {
      bias[static_cast<std::size_t>(i)] =
          static_cast<double>(packed_or_null[ih_n + hh_n + i]);
    }
  }

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(in, h, Activation::Linear, dt, ih_init, &layer.IH); !st) {
    return Status::Err(st.code, std::string("rnn IH: ") + st.message);
  }
  if (auto st = dense::new_layer(h, h, Activation::Linear, dt, hh_init, &layer.HH); !st) {
    return Status::Err(st.code, std::string("rnn HH: ") + st.message);
  }
  layer.IH.weights.bias = std::move(bias);
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: nil pre/post");
  }
  if (auto st = validate(layer.cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(layer.cfg, input, &lay); !st) return st;

  sync_projs(layer);

  pre->shape = {lay.batch, lay.seq, lay.hid};
  post->shape = {lay.batch, lay.seq, lay.hid};
  const int n = lay.batch * lay.seq * lay.hid;
  pre->data.assign(static_cast<std::size_t>(n), 0.0f);
  post->data.assign(static_cast<std::size_t>(n), 0.0f);

  std::vector<float> h_prev(static_cast<std::size_t>(lay.batch * lay.hid), 0.0f);

  for (int t = 0; t < lay.seq; ++t) {
    TensorF32 xt = x_at(input, lay, t);
    TensorF32 ht = h_tensor(h_prev, lay.batch, lay.hid);
    TensorF32 ih_pre, ih_post, hh_pre, hh_post;
    if (auto st = dense::forward_f32(layer.IH, xt, &ih_pre, &ih_post); !st) {
      return Status::Err(st.code, std::string("rnn IH: ") + st.message);
    }
    if (auto st = dense::forward_f32(layer.HH, ht, &hh_pre, &hh_post); !st) {
      return Status::Err(st.code, std::string("rnn HH: ") + st.message);
    }
    for (int b = 0; b < lay.batch; ++b) {
      for (int h = 0; h < lay.hid; ++h) {
        const int i = b * lay.hid + h;
        const double s = static_cast<double>(ih_post.data[static_cast<std::size_t>(i)]) +
                         static_cast<double>(hh_post.data[static_cast<std::size_t>(i)]);
        const int idx = b * lay.seq * lay.hid + t * lay.hid + h;
        pre->data[static_cast<std::size_t>(idx)] = static_cast<float>(s);
        const double th = std::tanh(s);
        post->data[static_cast<std::size_t>(idx)] = static_cast<float>(th);
        h_prev[static_cast<std::size_t>(i)] = static_cast<float>(th);
      }
    }
  }
  return Status::Ok();
}

int grad_w_size(const Layer& layer) { return weight_count(layer.cfg); }

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: nil grad out");
  }
  Layout lay;
  if (auto st = parse_layout(layer.cfg, input, &lay); !st) return st;
  const int want = lay.batch * lay.seq * lay.hid;
  if (static_cast<int>(grad_out.data.size()) < want || static_cast<int>(pre.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: grad/pre short");
  }

  sync_projs(layer);

  const int ih_n = lay.hid * lay.in;
  const int hh_n = lay.hid * lay.hid;
  std::vector<double> d_ih(static_cast<std::size_t>(ih_n), 0.0);
  std::vector<double> d_hh(static_cast<std::size_t>(hh_n), 0.0);
  std::vector<double> d_b(static_cast<std::size_t>(lay.hid), 0.0);
  std::vector<double> gi(static_cast<std::size_t>(lay.batch * lay.seq * lay.in), 0.0);
  std::vector<double> g_h(static_cast<std::size_t>(lay.batch * lay.hid), 0.0);

  std::vector<float> h_seq(static_cast<std::size_t>(want));
  for (int i = 0; i < want; ++i) {
    h_seq[static_cast<std::size_t>(i)] =
        static_cast<float>(std::tanh(static_cast<double>(pre.data[static_cast<std::size_t>(i)])));
  }

  for (int t = lay.seq - 1; t >= 0; --t) {
    TensorF32 g_pre;
    g_pre.shape = {lay.batch, lay.hid};
    g_pre.data.resize(static_cast<std::size_t>(lay.batch * lay.hid));
    for (int b = 0; b < lay.batch; ++b) {
      for (int h = 0; h < lay.hid; ++h) {
        const int idx = b * lay.seq * lay.hid + t * lay.hid + h;
        const double h_val =
            std::tanh(static_cast<double>(pre.data[static_cast<std::size_t>(idx)]));
        const double g =
            (g_h[static_cast<std::size_t>(b * lay.hid + h)] +
             static_cast<double>(grad_out.data[static_cast<std::size_t>(idx)])) *
            (1.0 - h_val * h_val);
        g_pre.data[static_cast<std::size_t>(b * lay.hid + h)] = static_cast<float>(g);
        d_b[static_cast<std::size_t>(h)] += g;
      }
    }

    TensorF32 xt = x_at(input, lay, t);
    TensorF32 h_prev;
    h_prev.shape = {lay.batch, lay.hid};
    h_prev.data.assign(static_cast<std::size_t>(lay.batch * lay.hid), 0.0f);
    if (t > 0) {
      for (int b = 0; b < lay.batch; ++b) {
        const int src = b * lay.seq * lay.hid + (t - 1) * lay.hid;
        for (int h = 0; h < lay.hid; ++h) {
          h_prev.data[static_cast<std::size_t>(b * lay.hid + h)] =
              h_seq[static_cast<std::size_t>(src + h)];
        }
      }
    }

    TensorF32 ih_pre, ih_post, hh_pre, hh_post;
    if (auto st = dense::forward_f32(layer.IH, xt, &ih_pre, &ih_post); !st) {
      return Status::Err(st.code, std::string("rnn IH recompute: ") + st.message);
    }
    if (auto st = dense::forward_f32(layer.HH, h_prev, &hh_pre, &hh_post); !st) {
      return Status::Err(st.code, std::string("rnn HH recompute: ") + st.message);
    }

    TensorF32 gx, d_wih, gh, d_whh;
    if (auto st = dense::backward_f32(layer.IH, g_pre, xt, ih_pre, &gx, &d_wih); !st) {
      return Status::Err(st.code, std::string("rnn IH bwd: ") + st.message);
    }
    if (auto st = dense::backward_f32(layer.HH, g_pre, h_prev, hh_pre, &gh, &d_whh); !st) {
      return Status::Err(st.code, std::string("rnn HH bwd: ") + st.message);
    }

    for (int i = 0; i < ih_n; ++i) {
      d_ih[static_cast<std::size_t>(i)] +=
          static_cast<double>(d_wih.data[static_cast<std::size_t>(i)]);
    }
    for (int i = 0; i < hh_n; ++i) {
      d_hh[static_cast<std::size_t>(i)] +=
          static_cast<double>(d_whh.data[static_cast<std::size_t>(i)]);
    }
    for (int b = 0; b < lay.batch; ++b) {
      for (int i = 0; i < lay.in; ++i) {
        gi[static_cast<std::size_t>(b * lay.seq * lay.in + t * lay.in + i)] +=
            static_cast<double>(gx.data[static_cast<std::size_t>(b * lay.in + i)]);
      }
      for (int h = 0; h < lay.hid; ++h) {
        g_h[static_cast<std::size_t>(b * lay.hid + h)] =
            static_cast<double>(gh.data[static_cast<std::size_t>(b * lay.hid + h)]);
      }
    }
  }

  grad_in->shape = {lay.batch, lay.seq, lay.in};
  grad_in->data.resize(static_cast<std::size_t>(lay.batch * lay.seq * lay.in));
  for (std::size_t i = 0; i < gi.size(); ++i) {
    grad_in->data[i] = static_cast<float>(gi[i]);
  }

  const int nw = weight_count(layer.cfg);
  grad_w->shape = {1, nw};
  grad_w->data.resize(static_cast<std::size_t>(nw));
  int off = 0;
  for (int i = 0; i < ih_n; ++i) {
    grad_w->data[static_cast<std::size_t>(off + i)] =
        static_cast<float>(d_ih[static_cast<std::size_t>(i)]);
  }
  off += ih_n;
  for (int i = 0; i < hh_n; ++i) {
    grad_w->data[static_cast<std::size_t>(off + i)] =
        static_cast<float>(d_hh[static_cast<std::size_t>(i)]);
  }
  off += hh_n;
  for (int i = 0; i < lay.hid; ++i) {
    grad_w->data[static_cast<std::size_t>(off + i)] =
        static_cast<float>(d_b[static_cast<std::size_t>(i)]);
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: ApplyGradSGD nil");
  }
  const int need = weight_count(layer->cfg);
  if (static_cast<int>(dW.data.size()) < need) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: dW short");
  }
  const int ih_n = layer->IH.weights.rows * layer->IH.weights.cols;
  const int hh_n = layer->HH.weights.rows * layer->HH.weights.cols;
  TensorF32 ih_slice;
  ih_slice.shape = {layer->IH.weights.rows, layer->IH.weights.cols};
  ih_slice.data.assign(dW.data.begin(), dW.data.begin() + ih_n);
  if (auto st = dense::apply_grad_sgd(&layer->IH, ih_slice, lr); !st) return st;
  TensorF32 hh_slice;
  hh_slice.shape = {layer->HH.weights.rows, layer->HH.weights.cols};
  hh_slice.data.assign(dW.data.begin() + ih_n, dW.data.begin() + ih_n + hh_n);
  if (auto st = dense::apply_grad_sgd(&layer->HH, hh_slice, lr); !st) return st;
  std::vector<double> d_b(static_cast<std::size_t>(layer->cfg.hidden_size));
  for (int i = 0; i < layer->cfg.hidden_size; ++i) {
    d_b[static_cast<std::size_t>(i)] =
        static_cast<double>(dW.data[static_cast<std::size_t>(ih_n + hh_n + i)]);
  }
  return layer->IH.weights.apply_bias_sgd(d_b.data(), layer->cfg.hidden_size, lr);
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "rnn: nil dst");
  }
  const int h = layer.cfg.hidden_size;
  const int in = layer.cfg.input_size;
  const int ih_n = h * in;
  const int hh_n = h * h;
  if (auto st = layer.IH.weights.export_f32(dst); !st) return st;
  if (auto st = layer.HH.weights.export_f32(dst + ih_n); !st) return st;
  for (int i = 0; i < h; ++i) {
    const double b = i < static_cast<int>(layer.IH.weights.bias.size())
                         ? layer.IH.weights.bias[static_cast<std::size_t>(i)]
                         : 0.0;
    dst[ih_n + hh_n + i] = static_cast<float>(b);
  }
  return Status::Ok();
}

} // namespace welvet::rnn
