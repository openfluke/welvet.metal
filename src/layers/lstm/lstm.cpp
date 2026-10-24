#include "welvet/lstm.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::lstm {
namespace {

struct Layout {
  int batch = 0, seq = 0, in = 0, hid = 0;
};

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: empty input");
  }
  if (input.shape.size() != 3) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: need [batch,seq,in]");
  }
  lay->batch = static_cast<int>(input.shape[0]);
  lay->seq = cfg.seq_len;
  lay->in = cfg.input_size;
  lay->hid = cfg.hidden_size;
  if (static_cast<int>(input.shape[1]) != lay->seq ||
      static_cast<int>(input.shape[2]) != lay->in) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: shape mismatch");
  }
  if (lay->batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: bad batch");
  }
  const int want = lay->batch * lay->seq * lay->in;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: short data");
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

Status new_gate(const Config& cfg, DType dt, const float* packed_or_null, Gate* out) {
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
  if (auto st = dense::new_layer(in, h, Activation::Linear, dt, ih_init, &out->IH); !st) {
    return st;
  }
  if (auto st = dense::new_layer(h, h, Activation::Linear, dt, hh_init, &out->HH); !st) {
    return st;
  }
  out->IH.weights.bias = std::move(bias);
  return Status::Ok();
}

Status gate_linear(Gate& g, const TensorF32& xt, const TensorF32& ht,
                   std::vector<double>* out) {
  g.IH.activation = Activation::Linear;
  g.HH.activation = Activation::Linear;
  TensorF32 ih_pre, ih_post, hh_pre, hh_post;
  if (auto st = dense::forward_f32(g.IH, xt, &ih_pre, &ih_post); !st) return st;
  if (auto st = dense::forward_f32(g.HH, ht, &hh_pre, &hh_post); !st) return st;
  out->resize(ih_post.data.size());
  for (std::size_t i = 0; i < ih_post.data.size(); ++i) {
    (*out)[i] = static_cast<double>(ih_post.data[i]) + static_cast<double>(hh_post.data[i]);
  }
  return Status::Ok();
}

} // namespace

int gate_size(const Config& cfg) {
  const int h = cfg.hidden_size;
  const int in = cfg.input_size;
  return h * in + h * h + h;
}

int weight_count(const Config& cfg) { return 4 * gate_size(cfg); }

Status validate(const Config& cfg) {
  if (cfg.input_size <= 0 || cfg.hidden_size <= 0 || cfg.seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: need positive Input/Hidden/SeqLen");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg, DType dt, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "lstm: nil out");
  if (auto st = validate(cfg); !st) return st;

  const int gn = gate_size(cfg);
  Layer layer;
  layer.cfg = cfg;

  auto mk = [&](Gate* g, int off, const char* name) -> Status {
    const float* slice = packed_or_null ? packed_or_null + off : nullptr;
    if (auto st = new_gate(cfg, dt, slice, g); !st) {
      return Status::Err(st.code, std::string("lstm ") + name + ": " + st.message);
    }
    return Status::Ok();
  };
  if (auto st = mk(&layer.I, 0, "i"); !st) return st;
  if (auto st = mk(&layer.F, gn, "f"); !st) return st;
  if (auto st = mk(&layer.G, 2 * gn, "g"); !st) return st;
  if (auto st = mk(&layer.O, 3 * gn, "o"); !st) return st;

  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: nil pre/post");
  }
  if (auto st = validate(layer.cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(layer.cfg, input, &lay); !st) return st;

  pre->shape = {lay.batch, lay.seq, 5 * lay.hid};
  post->shape = {lay.batch, lay.seq, lay.hid};
  pre->data.assign(static_cast<std::size_t>(lay.batch * lay.seq * 5 * lay.hid), 0.0f);
  post->data.assign(static_cast<std::size_t>(lay.batch * lay.seq * lay.hid), 0.0f);

  std::vector<float> h_prev(static_cast<std::size_t>(lay.batch * lay.hid), 0.0f);
  std::vector<double> c_prev(static_cast<std::size_t>(lay.batch * lay.hid), 0.0);

  for (int t = 0; t < lay.seq; ++t) {
    TensorF32 xt = x_at(input, lay, t);
    TensorF32 ht = h_tensor(h_prev, lay.batch, lay.hid);
    std::vector<double> i_sum, f_sum, g_sum, o_sum;
    if (auto st = gate_linear(layer.I, xt, ht, &i_sum); !st) {
      return Status::Err(st.code, std::string("lstm i: ") + st.message);
    }
    if (auto st = gate_linear(layer.F, xt, ht, &f_sum); !st) {
      return Status::Err(st.code, std::string("lstm f: ") + st.message);
    }
    if (auto st = gate_linear(layer.G, xt, ht, &g_sum); !st) {
      return Status::Err(st.code, std::string("lstm g: ") + st.message);
    }
    if (auto st = gate_linear(layer.O, xt, ht, &o_sum); !st) {
      return Status::Err(st.code, std::string("lstm o: ") + st.message);
    }

    for (int b = 0; b < lay.batch; ++b) {
      const int p_base = b * lay.seq * 5 * lay.hid + t * 5 * lay.hid;
      for (int h = 0; h < lay.hid; ++h) {
        const int i = b * lay.hid + h;
        const double i_s = i_sum[static_cast<std::size_t>(i)];
        const double f_s = f_sum[static_cast<std::size_t>(i)];
        const double g_s = g_sum[static_cast<std::size_t>(i)];
        const double o_s = o_sum[static_cast<std::size_t>(i)];
        pre->data[static_cast<std::size_t>(p_base + h)] = static_cast<float>(i_s);
        pre->data[static_cast<std::size_t>(p_base + lay.hid + h)] = static_cast<float>(f_s);
        pre->data[static_cast<std::size_t>(p_base + 2 * lay.hid + h)] = static_cast<float>(g_s);
        pre->data[static_cast<std::size_t>(p_base + 3 * lay.hid + h)] = static_cast<float>(o_s);

        const double i_g = 1.0 / (1.0 + std::exp(-i_s));
        const double f_g = 1.0 / (1.0 + std::exp(-f_s));
        const double g_g = std::tanh(g_s);
        const double o_g = 1.0 / (1.0 + std::exp(-o_s));
        const double c_c = f_g * c_prev[static_cast<std::size_t>(i)] + i_g * g_g;
        const double h_c = o_g * std::tanh(c_c);

        pre->data[static_cast<std::size_t>(p_base + 4 * lay.hid + h)] = static_cast<float>(c_c);
        post->data[static_cast<std::size_t>(b * lay.seq * lay.hid + t * lay.hid + h)] =
            static_cast<float>(h_c);
        h_prev[static_cast<std::size_t>(i)] = static_cast<float>(h_c);
        c_prev[static_cast<std::size_t>(i)] = c_c;
      }
    }
  }
  return Status::Ok();
}

} // namespace welvet::lstm
