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


int grad_w_size(const Layer& layer) { return weight_count(layer.cfg); }

Status accumulate_gate(Gate& g, const TensorF32& delta, const TensorF32& xt,
                       const TensorF32& h_prev, double* d_ih, double* d_hh, double* d_b, int ih_n,
                       int hh_n, int hid, std::vector<double>* gx, std::vector<double>* gh) {
  g.IH.activation = Activation::Linear;
  g.HH.activation = Activation::Linear;
  TensorF32 ih_pre, ih_post, hh_pre, hh_post;
  if (auto st = dense::forward_f32(g.IH, xt, &ih_pre, &ih_post); !st) return st;
  if (auto st = dense::forward_f32(g.HH, h_prev, &hh_pre, &hh_post); !st) return st;
  TensorF32 gx_t, d_wih, gh_t, d_whh;
  if (auto st = dense::backward_f32(g.IH, delta, xt, ih_pre, &gx_t, &d_wih); !st) return st;
  if (auto st = dense::backward_f32(g.HH, delta, h_prev, hh_pre, &gh_t, &d_whh); !st) return st;
  for (int i = 0; i < ih_n; ++i) {
    d_ih[i] += static_cast<double>(d_wih.data[static_cast<std::size_t>(i)]);
  }
  for (int i = 0; i < hh_n; ++i) {
    d_hh[i] += static_cast<double>(d_whh.data[static_cast<std::size_t>(i)]);
  }
  const int batch = static_cast<int>(delta.shape[0]);
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < hid; ++h) {
      d_b[h] += static_cast<double>(delta.data[static_cast<std::size_t>(b * hid + h)]);
    }
  }
  gx->resize(gx_t.data.size());
  gh->resize(gh_t.data.size());
  for (std::size_t i = 0; i < gx_t.data.size(); ++i) {
    (*gx)[i] = static_cast<double>(gx_t.data[i]);
  }
  for (std::size_t i = 0; i < gh_t.data.size(); ++i) {
    (*gh)[i] = static_cast<double>(gh_t.data[i]);
  }
  return Status::Ok();
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: nil grad out");
  }
  Layout lay;
  if (auto st = parse_layout(layer.cfg, input, &lay); !st) return st;
  const int want_pre = lay.batch * lay.seq * 5 * lay.hid;
  const int want_out = lay.batch * lay.seq * lay.hid;
  if (static_cast<int>(pre.data.size()) < want_pre ||
      static_cast<int>(grad_out.data.size()) < want_out) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: grad/pre short");
  }

  const int ih_n = lay.hid * lay.in;
  const int hh_n = lay.hid * lay.hid;
  const int gate_n = ih_n + hh_n + lay.hid;
  std::vector<double> d_pack(static_cast<std::size_t>(4 * gate_n), 0.0);
  double* d_ih[4];
  double* d_hh[4];
  double* d_b[4];
  for (int g = 0; g < 4; ++g) {
    const int base = g * gate_n;
    d_ih[g] = d_pack.data() + base;
    d_hh[g] = d_pack.data() + base + ih_n;
    d_b[g] = d_pack.data() + base + ih_n + hh_n;
  }
  Gate* gates[4] = {&layer.I, &layer.F, &layer.G, &layer.O};

  std::vector<double> gi(static_cast<std::size_t>(lay.batch * lay.seq * lay.in), 0.0);
  std::vector<double> grad_h(static_cast<std::size_t>(lay.batch * lay.hid), 0.0);
  std::vector<double> grad_c(static_cast<std::size_t>(lay.batch * lay.hid), 0.0);

  for (int t = lay.seq - 1; t >= 0; --t) {
    std::vector<double> next_gh(static_cast<std::size_t>(lay.batch * lay.hid), 0.0);
    std::vector<double> next_gc(static_cast<std::size_t>(lay.batch * lay.hid), 0.0);
    TensorF32 delta_t[4];
    for (int g = 0; g < 4; ++g) {
      delta_t[g].shape = {lay.batch, lay.hid};
      delta_t[g].data.assign(static_cast<std::size_t>(lay.batch * lay.hid), 0.0f);
    }

    for (int b = 0; b < lay.batch; ++b) {
      const int p_base = b * lay.seq * 5 * lay.hid + t * 5 * lay.hid;
      for (int h = 0; h < lay.hid; ++h) {
        const int i = b * lay.hid + h;
        const double dh =
            grad_h[static_cast<std::size_t>(i)] +
            static_cast<double>(
                grad_out.data[static_cast<std::size_t>(b * lay.seq * lay.hid + t * lay.hid + h)]);
        const double i_s = static_cast<double>(pre.data[static_cast<std::size_t>(p_base + h)]);
        const double f_s =
            static_cast<double>(pre.data[static_cast<std::size_t>(p_base + lay.hid + h)]);
        const double g_s =
            static_cast<double>(pre.data[static_cast<std::size_t>(p_base + 2 * lay.hid + h)]);
        const double o_s =
            static_cast<double>(pre.data[static_cast<std::size_t>(p_base + 3 * lay.hid + h)]);
        const double c_c =
            static_cast<double>(pre.data[static_cast<std::size_t>(p_base + 4 * lay.hid + h)]);
        const double i_g = 1.0 / (1.0 + std::exp(-i_s));
        const double f_g = 1.0 / (1.0 + std::exp(-f_s));
        const double o_g = 1.0 / (1.0 + std::exp(-o_s));
        const double g_g = std::tanh(g_s);
        const double c_t = std::tanh(c_c);
        double c_p = 0.0;
        if (t > 0) {
          c_p = static_cast<double>(
              pre.data[static_cast<std::size_t>(p_base - 5 * lay.hid + 4 * lay.hid + h)]);
        }
        const double dc = grad_c[static_cast<std::size_t>(i)] + dh * o_g * (1.0 - c_t * c_t);
        const double di = dc * g_g * i_g * (1.0 - i_g);
        const double df = dc * c_p * f_g * (1.0 - f_g);
        const double dg = dc * i_g * (1.0 - g_g * g_g);
        const double d_o = dh * c_t * o_g * (1.0 - o_g);
        delta_t[0].data[static_cast<std::size_t>(i)] = static_cast<float>(di);
        delta_t[1].data[static_cast<std::size_t>(i)] = static_cast<float>(df);
        delta_t[2].data[static_cast<std::size_t>(i)] = static_cast<float>(dg);
        delta_t[3].data[static_cast<std::size_t>(i)] = static_cast<float>(d_o);
        next_gc[static_cast<std::size_t>(i)] = dc * f_g;
      }
    }

    TensorF32 xt = x_at(input, lay, t);
    TensorF32 h_prev;
    h_prev.shape = {lay.batch, lay.hid};
    h_prev.data.assign(static_cast<std::size_t>(lay.batch * lay.hid), 0.0f);
    if (t > 0) {
      for (int b = 0; b < lay.batch; ++b) {
        const int p_p = b * lay.seq * 5 * lay.hid + (t - 1) * 5 * lay.hid;
        for (int h = 0; h < lay.hid; ++h) {
          const double o_s =
              static_cast<double>(pre.data[static_cast<std::size_t>(p_p + 3 * lay.hid + h)]);
          const double c_c =
              static_cast<double>(pre.data[static_cast<std::size_t>(p_p + 4 * lay.hid + h)]);
          const double o_g = 1.0 / (1.0 + std::exp(-o_s));
          h_prev.data[static_cast<std::size_t>(b * lay.hid + h)] =
              static_cast<float>(o_g * std::tanh(c_c));
        }
      }
    }

    for (int g = 0; g < 4; ++g) {
      std::vector<double> gx, gh;
      if (auto st = accumulate_gate(*gates[g], delta_t[g], xt, h_prev, d_ih[g], d_hh[g], d_b[g],
                                    ih_n, hh_n, lay.hid, &gx, &gh);
          !st) {
        return Status::Err(st.code, std::string("lstm gate bwd: ") + st.message);
      }
      for (int b = 0; b < lay.batch; ++b) {
        for (int i = 0; i < lay.in; ++i) {
          gi[static_cast<std::size_t>(b * lay.seq * lay.in + t * lay.in + i)] +=
              gx[static_cast<std::size_t>(b * lay.in + i)];
        }
        for (int h = 0; h < lay.hid; ++h) {
          next_gh[static_cast<std::size_t>(b * lay.hid + h)] +=
              gh[static_cast<std::size_t>(b * lay.hid + h)];
        }
      }
    }
    grad_h = std::move(next_gh);
    grad_c = std::move(next_gc);
  }

  grad_in->shape = {lay.batch, lay.seq, lay.in};
  grad_in->data.resize(gi.size());
  for (std::size_t i = 0; i < gi.size(); ++i) {
    grad_in->data[i] = static_cast<float>(gi[i]);
  }
  const int nw = weight_count(layer.cfg);
  grad_w->shape = {1, nw};
  grad_w->data.resize(static_cast<std::size_t>(nw));
  for (int i = 0; i < nw; ++i) {
    grad_w->data[static_cast<std::size_t>(i)] = static_cast<float>(d_pack[static_cast<std::size_t>(i)]);
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: ApplyGradSGD nil");
  }
  const int need = weight_count(layer->cfg);
  if (static_cast<int>(dW.data.size()) < need) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: dW short");
  }
  const int gn = gate_size(layer->cfg);
  const int ih_n = layer->I.IH.weights.rows * layer->I.IH.weights.cols;
  const int hh_n = layer->I.HH.weights.rows * layer->I.HH.weights.cols;
  Gate* gates[4] = {&layer->I, &layer->F, &layer->G, &layer->O};
  for (int gi = 0; gi < 4; ++gi) {
    const int off = gi * gn;
    Gate* g = gates[gi];
    TensorF32 ih_slice;
    ih_slice.shape = {g->IH.weights.rows, g->IH.weights.cols};
    ih_slice.data.assign(dW.data.begin() + off, dW.data.begin() + off + ih_n);
    if (auto st = dense::apply_grad_sgd(&g->IH, ih_slice, lr); !st) return st;
    TensorF32 hh_slice;
    hh_slice.shape = {g->HH.weights.rows, g->HH.weights.cols};
    hh_slice.data.assign(dW.data.begin() + off + ih_n, dW.data.begin() + off + ih_n + hh_n);
    if (auto st = dense::apply_grad_sgd(&g->HH, hh_slice, lr); !st) return st;
    std::vector<double> d_b(static_cast<std::size_t>(layer->cfg.hidden_size));
    for (int i = 0; i < layer->cfg.hidden_size; ++i) {
      d_b[static_cast<std::size_t>(i)] =
          static_cast<double>(dW.data[static_cast<std::size_t>(off + ih_n + hh_n + i)]);
    }
    if (auto st = g->IH.weights.apply_bias_sgd(d_b.data(), layer->cfg.hidden_size, lr); !st) {
      return st;
    }
  }
  return Status::Ok();
}

Status export_gate_packed(const Gate& g, int h, int in, float* dst) {
  const int ih_n = h * in;
  const int hh_n = h * h;
  if (auto st = g.IH.weights.export_f32(dst); !st) return st;
  if (auto st = g.HH.weights.export_f32(dst + ih_n); !st) return st;
  for (int i = 0; i < h; ++i) {
    const double b =
        i < static_cast<int>(g.IH.weights.bias.size()) ? g.IH.weights.bias[static_cast<std::size_t>(i)]
                                                       : 0.0;
    dst[ih_n + hh_n + i] = static_cast<float>(b);
  }
  return Status::Ok();
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: nil dst");
  }
  const int gn = gate_size(layer.cfg);
  const int h = layer.cfg.hidden_size;
  const int in = layer.cfg.input_size;
  const Gate* gates[4] = {&layer.I, &layer.F, &layer.G, &layer.O};
  for (int i = 0; i < 4; ++i) {
    if (auto st = export_gate_packed(*gates[i], h, in, dst + i * gn); !st) return st;
  }
  return Status::Ok();
}

} // namespace welvet::lstm
