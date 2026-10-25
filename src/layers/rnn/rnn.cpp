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
  return Status::Err(StatusCode::Unsupported, "rnn.forward: Metal kernel not implemented; CPU execution is forbidden");
}

int grad_w_size(const Layer& layer) { return weight_count(layer.cfg); }

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "rnn.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "rnn.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
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
