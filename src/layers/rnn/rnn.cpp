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

} // namespace welvet::rnn
