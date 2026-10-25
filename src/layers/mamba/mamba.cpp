#include "welvet/mamba.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::mamba {
namespace {

} // namespace

int inner_dim(const Config& cfg) {
  int e = cfg.expand;
  if (e <= 0) e = 2;
  return e * cfg.d_model;
}

int weight_count(const Config& cfg) {
  Config c = cfg;
  if (c.expand <= 0) c.expand = 2;
  const int inner = c.expand * c.d_model;
  return inner * (3 * c.d_model + 2);
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "mamba: nil config");
  if (cfg->d_model <= 0 || cfg->d_state <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: need positive DModel/DState");
  }
  if (cfg->expand <= 0) cfg->expand = 2;
  if (cfg->seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: need positive SeqLen");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "mamba: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  const int d = cfg.d_model;
  const int inner = inner_dim(cfg);
  const int in_n = 2 * inner * d;
  const int out_n = d * inner;

  Layer layer;
  layer.cfg = cfg;

  const float* in_init = nullptr;
  const float* out_init = nullptr;
  if (packed_or_null) {
    in_init = packed_or_null;
    out_init = packed_or_null + in_n;
  }
  if (auto st = dense::new_layer(d, 2 * inner, Activation::Linear, dt, in_init, &layer.in_proj); !st) {
    return Status::Err(st.code, std::string("mamba in: ") + st.message);
  }
  if (auto st = dense::new_layer(inner, d, Activation::Linear, dt, out_init, &layer.out_proj); !st) {
    return Status::Err(st.code, std::string("mamba out: ") + st.message);
  }

  layer.a_log.resize(static_cast<std::size_t>(inner));
  layer.d_skip.resize(static_cast<std::size_t>(inner));
  if (packed_or_null) {
    const float* a = packed_or_null + in_n + out_n;
    const float* ds = a + inner;
    for (int i = 0; i < inner; ++i) {
      layer.a_log[static_cast<std::size_t>(i)] = a[i];
      layer.d_skip[static_cast<std::size_t>(i)] = ds[i];
    }
  } else {
    for (int i = 0; i < inner; ++i) {
      layer.a_log[static_cast<std::size_t>(i)] = -static_cast<float>(i + 1) * 0.1f;
      layer.d_skip[static_cast<std::size_t>(i)] = 1.0f;
    }
  }

  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "mamba.forward: Metal kernel not implemented; CPU execution is forbidden");
}

int grad_w_size(const Layer& layer) {
  const int in_n = layer.in_proj.weights.rows * layer.in_proj.weights.cols;
  const int out_n = layer.out_proj.weights.rows * layer.out_proj.weights.cols;
  return in_n + out_n + static_cast<int>(layer.a_log.size()) +
         static_cast<int>(layer.d_skip.size());
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& /*pre*/, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "mamba.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "mamba.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: nil dst");
  }
  int off = 0;
  const int n_in = layer.in_proj.weights.rows * layer.in_proj.weights.cols;
  if (auto st = layer.in_proj.weights.export_f32(dst + off); !st) return st;
  off += n_in;
  const int n_out = layer.out_proj.weights.rows * layer.out_proj.weights.cols;
  if (auto st = layer.out_proj.weights.export_f32(dst + off); !st) return st;
  off += n_out;
  for (float v : layer.a_log) {
    dst[off++] = v;
  }
  for (float v : layer.d_skip) {
    dst[off++] = v;
  }
  return Status::Ok();
}

} // namespace welvet::mamba
