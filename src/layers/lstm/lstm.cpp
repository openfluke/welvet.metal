#include "welvet/lstm.hpp"
#include "welvet/metal/layer_utils.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::lstm {
namespace {

struct Layout {
  int batch = 0, seq = 0, in = 0, hid = 0;
};

Status parse_layout(const Config &cfg, const TensorF32 &input, Layout *lay) {
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
  if (static_cast<int>(input.shape[1]) != lay->seq || static_cast<int>(input.shape[2]) != lay->in) {
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

Status new_gate(const Config &cfg, DType dt, const float *packed_or_null, Gate *out) {
  const int in = cfg.input_size;
  const int h = cfg.hidden_size;
  const int ih_n = h * in;
  const int hh_n = h * h;
  const float *ih_init = nullptr;
  const float *hh_init = nullptr;
  std::vector<double> bias(static_cast<std::size_t>(h), 0.0);
  if (packed_or_null) {
    ih_init = packed_or_null;
    hh_init = packed_or_null + ih_n;
    for (int i = 0; i < h; ++i) {
      bias[static_cast<std::size_t>(i)] = static_cast<double>(packed_or_null[ih_n + hh_n + i]);
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

} // namespace

int gate_size(const Config &cfg) {
  const int h = cfg.hidden_size;
  const int in = cfg.input_size;
  return h * in + h * h + h;
}

int weight_count(const Config &cfg) { return 4 * gate_size(cfg); }

Status validate(const Config &cfg) {
  if (cfg.input_size <= 0 || cfg.hidden_size <= 0 || cfg.seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: need positive Input/Hidden/SeqLen");
  }
  return Status::Ok();
}

Status new_configured(const Config &cfg, DType dt, const float *packed_or_null, Layer *out) {
  if (!out)
    return Status::Err(StatusCode::InvalidArgument, "lstm: nil out");
  if (auto st = validate(cfg); !st)
    return st;

  const int gn = gate_size(cfg);
  Layer layer;
  layer.cfg = cfg;

  auto mk = [&](Gate *g, int off, const char *name) -> Status {
    const float *slice = packed_or_null ? packed_or_null + off : nullptr;
    if (auto st = new_gate(cfg, dt, slice, g); !st) {
      return Status::Err(st.code, std::string("lstm ") + name + ": " + st.message);
    }
    return Status::Ok();
  };
  if (auto st = mk(&layer.I, 0, "i"); !st)
    return st;
  if (auto st = mk(&layer.F, gn, "f"); !st)
    return st;
  if (auto st = mk(&layer.G, 2 * gn, "g"); !st)
    return st;
  if (auto st = mk(&layer.O, 3 * gn, "o"); !st)
    return st;

  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer &layer, const TensorF32 &input, TensorF32 *pre, TensorF32 *post) {
  if (!pre || !post)
    return Status::Err(StatusCode::InvalidArgument, "lstm: nil outputs");
  Layout lay;
  if (auto st = parse_layout(layer.cfg, input, &lay); !st)
    return st;
  const int B = lay.batch, T = lay.seq, I = lay.in, H = lay.hid;
  std::vector<float> packed;
  dense::Layer *ihs[] = {&layer.I.IH, &layer.F.IH, &layer.G.IH, &layer.O.IH};
  dense::Layer *hhs[] = {&layer.I.HH, &layer.F.HH, &layer.G.HH, &layer.O.HH};
  for (int gate = 0; gate < 4; ++gate) {
    std::vector<float> wi, wh;
    if (auto st = metal::decode_xy(ihs[gate]->weights, &wi); !st)
      return st;
    if (auto st = metal::decode_xy(hhs[gate]->weights, &wh); !st)
      return st;
    packed.insert(packed.end(), wi.begin(), wi.end());
    packed.insert(packed.end(), wh.begin(), wh.end());
    for (int i = 0; i < H; ++i) {
      float hi = 0, lo = 0;
      if (i < int(ihs[gate]->weights.bias.size()))
        metal::split_double(ihs[gate]->weights.bias[i], &hi, &lo);
      packed.push_back(hi);
      packed.push_back(lo);
    }
  }

  pre->shape = {B, T, 5 * H};
  post->shape = {B, T, H};
  pre->data.resize(B * T * 5 * H);
  post->data.resize(B * T * H);
  std::vector<float> hp(B * H, 0), cell(B * H * 2, 0), sums(B * H * 4 * 2, 0);
  return metal::run("op_recurrent_forward",
                    {metal::read(input.data), metal::read(packed), metal::write(pre->data),
                     metal::write(post->data), metal::write(hp), metal::write(cell),
                     metal::write(sums)},
                    {B, T, I, H, 4});
}

int grad_w_size(const Layer &layer) { return weight_count(layer.cfg); }

Status backward_f32(Layer &layer, const TensorF32 &grad_out, const TensorF32 &input,
                    const TensorF32 &pre, TensorF32 *grad_in, TensorF32 *grad_w) {
  if (!grad_in || !grad_w)
    return Status::Err(StatusCode::InvalidArgument, "lstm: nil gradients");
  Layout lay;
  if (auto st = parse_layout(layer.cfg, input, &lay); !st)
    return st;
  const int B = lay.batch, T = lay.seq, I = lay.in, H = lay.hid;
  std::vector<float> packed;
  dense::Layer *ihs[] = {&layer.I.IH, &layer.F.IH, &layer.G.IH, &layer.O.IH};
  dense::Layer *hhs[] = {&layer.I.HH, &layer.F.HH, &layer.G.HH, &layer.O.HH};
  for (int gate = 0; gate < 4; ++gate) {
    std::vector<float> wi, wh;
    if (auto st = metal::decode_xy(ihs[gate]->weights, &wi); !st)
      return st;
    if (auto st = metal::decode_xy(hhs[gate]->weights, &wh); !st)
      return st;
    packed.insert(packed.end(), wi.begin(), wi.end());
    packed.insert(packed.end(), wh.begin(), wh.end());
    for (int i = 0; i < H; ++i) {
      float hi = 0, lo = 0;
      if (i < int(ihs[gate]->weights.bias.size()))
        metal::split_double(ihs[gate]->weights.bias[i], &hi, &lo);
      packed.push_back(hi);
      packed.push_back(lo);
    }
  }

  if (grad_out.data.size() < B * T * H || pre.data.size() < B * T * 5 * H)
    return Status::Err(StatusCode::InvalidArgument, "lstm: short tape/gradient");
  grad_in->shape = {B, T, I};
  grad_in->data.resize(B * T * I);
  *grad_w = TensorF32::zeros(1, weight_count(layer.cfg));
  std::vector<float> scratch(
      2 * (weight_count(layer.cfg) + 3 * B * H + B * T * I) + B * 4 * H + B * H, 0),
      empty;
  return metal::run("op_recurrent_backward",
                    {metal::read(input.data), metal::read(packed), metal::read(grad_out.data),
                     metal::read(pre.data), metal::write(grad_in->data), metal::write(grad_w->data),
                     metal::read(empty), metal::write(scratch)},
                    {B, T, I, H, 4});
}

Status apply_grad_sgd(Layer *layer, const TensorF32 &dW, double lr) {
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
  Gate *gates[4] = {&layer->I, &layer->F, &layer->G, &layer->O};
  for (int gi = 0; gi < 4; ++gi) {
    const int off = gi * gn;
    Gate *g = gates[gi];
    TensorF32 ih_slice;
    ih_slice.shape = {g->IH.weights.rows, g->IH.weights.cols};
    ih_slice.data.assign(dW.data.begin() + off, dW.data.begin() + off + ih_n);
    if (auto st = dense::apply_grad_sgd(&g->IH, ih_slice, lr); !st)
      return st;
    TensorF32 hh_slice;
    hh_slice.shape = {g->HH.weights.rows, g->HH.weights.cols};
    hh_slice.data.assign(dW.data.begin() + off + ih_n, dW.data.begin() + off + ih_n + hh_n);
    if (auto st = dense::apply_grad_sgd(&g->HH, hh_slice, lr); !st)
      return st;
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

Status export_gate_packed(const Gate &g, int h, int in, float *dst) {
  const int ih_n = h * in;
  const int hh_n = h * h;
  if (auto st = g.IH.weights.export_f32(dst); !st)
    return st;
  if (auto st = g.HH.weights.export_f32(dst + ih_n); !st)
    return st;
  for (int i = 0; i < h; ++i) {
    const double b = i < static_cast<int>(g.IH.weights.bias.size())
                         ? g.IH.weights.bias[static_cast<std::size_t>(i)]
                         : 0.0;
    dst[ih_n + hh_n + i] = static_cast<float>(b);
  }
  return Status::Ok();
}

Status export_packed_f32(const Layer &layer, float *dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "lstm: nil dst");
  }
  const int gn = gate_size(layer.cfg);
  const int h = layer.cfg.hidden_size;
  const int in = layer.cfg.input_size;
  const Gate *gates[4] = {&layer.I, &layer.F, &layer.G, &layer.O};
  for (int i = 0; i < 4; ++i) {
    if (auto st = export_gate_packed(*gates[i], h, in, dst + i * gn); !st)
      return st;
  }
  return Status::Ok();
}

} // namespace welvet::lstm
