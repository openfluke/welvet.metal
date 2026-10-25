#include "welvet/lstm.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::lstm {
namespace {

struct Layout {
  int batch = 0, seq = 0, in = 0, hid = 0;
};

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
  return Status::Err(StatusCode::Unsupported, "lstm.forward: Metal kernel not implemented; CPU execution is forbidden");
}


int grad_w_size(const Layer& layer) { return weight_count(layer.cfg); }


Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "lstm.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "lstm.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
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
