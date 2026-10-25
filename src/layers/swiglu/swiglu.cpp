#include "welvet/swiglu.hpp"

#include "welvet/activate.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace welvet::swiglu {
namespace {

struct Layout {
  int batch = 1;
  int seq = 1;
  int in = 0;
};

Status parse_layout(int in_dim, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: empty input");
  }
  lay->in = in_dim;
  if (input.shape.size() == 2) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = 1;
    if (static_cast<int>(input.shape[1]) != in_dim) {
      return Status::Err(StatusCode::InvalidArgument, "swiglu: width != InputDim");
    }
  } else if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != in_dim) {
      return Status::Err(StatusCode::InvalidArgument, "swiglu: last dim != InputDim");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: need [batch,in] or [batch,seq,in]");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: bad batch/seq");
  }
  const int want = lay->batch * lay->seq * lay->in;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: short data");
  }
  return Status::Ok();
}

} // namespace

Status validate(const Config& cfg) {
  if (cfg.input_dim <= 0 || cfg.intermediate_dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: need InputDim>0 and IntermediateDim>0");
  }
  return Status::Ok();
}

Config default_ffn(int d_model) {
  int inter = (d_model * 8) / 3;
  inter = (inter + 7) / 8 * 8;
  if (inter < d_model) inter = d_model * 2;
  return Config{d_model, inter};
}

Status new_configured(const Config& cfg, DType dt, const float* gate_init, const float* up_init,
                      const float* down_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "swiglu: nil out");
  if (auto st = validate(cfg); !st) return st;

  const int in = cfg.input_dim;
  const int inter = cfg.intermediate_dim;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(in, inter, Activation::Linear, dt, gate_init, &layer.Gate); !st) {
    return Status::Err(st.code, std::string("swiglu Gate: ") + st.message);
  }
  if (auto st = dense::new_layer(in, inter, Activation::Linear, dt, up_init, &layer.Up); !st) {
    return Status::Err(st.code, std::string("swiglu Up: ") + st.message);
  }
  if (auto st = dense::new_layer(inter, in, Activation::Linear, dt, down_init, &layer.Down); !st) {
    return Status::Err(st.code, std::string("swiglu Down: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: nil pre/post");
  }
  if (auto st = validate(layer.cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(layer.cfg.input_dim, input, &lay); !st) return st;

  const int in = layer.cfg.input_dim;
  const int inter = layer.cfg.intermediate_dim;
  const int bs = lay.batch * lay.seq;

  TensorF32 flat;
  flat.shape = {bs, in};
  flat.data.assign(input.data.begin(), input.data.begin() + static_cast<std::size_t>(bs * in));

  TensorF32 gate_pre, gate_post, up_pre, up_post;
  if (auto st = dense::forward_f32(layer.Gate, flat, &gate_pre, &gate_post); !st) {
    return Status::Err(st.code, std::string("swiglu Gate: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.Up, flat, &up_pre, &up_post); !st) {
    return Status::Err(st.code, std::string("swiglu Up: ") + st.message);
  }

  // Match Go: SiLU(gatePre) → f32, then f64(g)*f64(upPost) → f32.
  TensorF32 h;
  h.shape = {bs, inter};
  h.data.resize(static_cast<std::size_t>(bs * inter));
  const int n = bs * inter;
  for (int i = 0; i < n; ++i) {
    const double silu = activate_f64(static_cast<double>(gate_pre.data[static_cast<std::size_t>(i)]),
                                     Activation::Silu);
    const float g_f = static_cast<float>(silu);
    h.data[static_cast<std::size_t>(i)] = static_cast<float>(
        static_cast<double>(g_f) * static_cast<double>(up_post.data[static_cast<std::size_t>(i)]));
  }

  TensorF32 down_pre, down_post;
  if (auto st = dense::forward_f32(layer.Down, h, &down_pre, &down_post); !st) {
    return Status::Err(st.code, std::string("swiglu Down: ") + st.message);
  }

  if (lay.seq == 1) {
    pre->shape = {lay.batch, inter};
    post->shape = {lay.batch, in};
  } else {
    pre->shape = {lay.batch, lay.seq, inter};
    post->shape = {lay.batch, lay.seq, in};
  }
  pre->data = std::move(h.data);
  post->data = std::move(down_post.data);
  return Status::Ok();
}

int grad_w_size(const Layer& layer) {
  return layer.Gate.weights.rows * layer.Gate.weights.cols +
         layer.Up.weights.rows * layer.Up.weights.cols +
         layer.Down.weights.rows * layer.Down.weights.cols;
}

Status export_weights_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: nil export");
  }
  int off = 0;
  if (auto st = layer.Gate.weights.export_f32(dst + off); !st) return st;
  off += layer.Gate.weights.rows * layer.Gate.weights.cols;
  if (auto st = layer.Up.weights.export_f32(dst + off); !st) return st;
  off += layer.Up.weights.rows * layer.Up.weights.cols;
  return layer.Down.weights.export_f32(dst + off);
}

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: nil grad out");
  }
  if (auto st = validate(layer.cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(layer.cfg.input_dim, input, &lay); !st) return st;

  const int in = layer.cfg.input_dim;
  const int inter = layer.cfg.intermediate_dim;
  const int bs = lay.batch * lay.seq;
  if (static_cast<int>(pre.data.size()) < bs * inter) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: pre (h) missing or wrong size");
  }
  if (static_cast<int>(grad_out.data.size()) < bs * in) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: gradOut short");
  }

  TensorF32 flat;
  flat.shape = {bs, in};
  flat.data.assign(input.data.begin(), input.data.begin() + static_cast<std::size_t>(bs * in));

  TensorF32 gy;
  gy.shape = {bs, in};
  gy.data.assign(grad_out.data.begin(), grad_out.data.begin() + static_cast<std::size_t>(bs * in));

  TensorF32 h;
  h.shape = {bs, inter};
  h.data.assign(pre.data.begin(), pre.data.begin() + static_cast<std::size_t>(bs * inter));

  TensorF32 down_pre, down_post;
  if (auto st = dense::forward_f32(layer.Down, h, &down_pre, &down_post); !st) {
    return Status::Err(st.code, std::string("swiglu Down recompute: ") + st.message);
  }
  TensorF32 grad_h, grad_wd;
  if (auto st = dense::backward_f32(layer.Down, gy, h, down_pre, &grad_h, &grad_wd); !st) {
    return Status::Err(st.code, std::string("swiglu Down bwd: ") + st.message);
  }

  TensorF32 gate_pre, gate_post, up_pre, up_post;
  if (auto st = dense::forward_f32(layer.Gate, flat, &gate_pre, &gate_post); !st) {
    return Status::Err(st.code, std::string("swiglu Gate recompute: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.Up, flat, &up_pre, &up_post); !st) {
    return Status::Err(st.code, std::string("swiglu Up recompute: ") + st.message);
  }

  TensorF32 d_gate, d_up;
  d_gate.shape = {bs, inter};
  d_up.shape = {bs, inter};
  d_gate.data.resize(static_cast<std::size_t>(bs * inter));
  d_up.data.resize(static_cast<std::size_t>(bs * inter));
  const int n = bs * inter;
  for (int i = 0; i < n; ++i) {
    const double gp = static_cast<double>(gate_pre.data[static_cast<std::size_t>(i)]);
    // Match Go Activate/ActivateDeriv → T (f32) then AsFloat64 in the products.
    const float silu_f = static_cast<float>(activate_f64(gp, Activation::Silu));
    const float dsilu_f = static_cast<float>(activate_deriv_f64(gp, Activation::Silu));
    const double dh = static_cast<double>(grad_h.data[static_cast<std::size_t>(i)]);
    const double up = static_cast<double>(up_post.data[static_cast<std::size_t>(i)]);
    d_up.data[static_cast<std::size_t>(i)] =
        static_cast<float>(dh * static_cast<double>(silu_f));
    d_gate.data[static_cast<std::size_t>(i)] =
        static_cast<float>(dh * up * static_cast<double>(dsilu_f));
  }

  TensorF32 grad_in_g, grad_wg, grad_in_u, grad_wu;
  if (auto st = dense::backward_f32(layer.Gate, d_gate, flat, gate_pre, &grad_in_g, &grad_wg); !st) {
    return Status::Err(st.code, std::string("swiglu Gate bwd: ") + st.message);
  }
  if (auto st = dense::backward_f32(layer.Up, d_up, flat, up_pre, &grad_in_u, &grad_wu); !st) {
    return Status::Err(st.code, std::string("swiglu Up bwd: ") + st.message);
  }

  if (lay.seq == 1) {
    grad_in->shape = {lay.batch, in};
  } else {
    grad_in->shape = {lay.batch, lay.seq, in};
  }
  grad_in->data.resize(static_cast<std::size_t>(bs * in));
  for (int i = 0; i < bs * in; ++i) {
    grad_in->data[static_cast<std::size_t>(i)] = static_cast<float>(
        static_cast<double>(grad_in_g.data[static_cast<std::size_t>(i)]) +
        static_cast<double>(grad_in_u.data[static_cast<std::size_t>(i)]));
  }

  const int nw = grad_w_size(layer);
  *grad_w = TensorF32::zeros(1, nw);
  int off = 0;
  std::memcpy(grad_w->data.data() + off, grad_wg.data.data(),
              grad_wg.data.size() * sizeof(float));
  off += static_cast<int>(grad_wg.data.size());
  std::memcpy(grad_w->data.data() + off, grad_wu.data.data(),
              grad_wu.data.size() * sizeof(float));
  off += static_cast<int>(grad_wu.data.size());
  std::memcpy(grad_w->data.data() + off, grad_wd.data.data(),
              grad_wd.data.size() * sizeof(float));
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: ApplyGradSGD nil");
  }
  const int need = grad_w_size(*layer);
  if (static_cast<int>(dW.data.size()) < need) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: dW short");
  }
  int off = 0;
  dense::Layer* parts[3] = {&layer->Gate, &layer->Up, &layer->Down};
  for (dense::Layer* p : parts) {
    const int n = p->weights.rows * p->weights.cols;
    TensorF32 slice;
    slice.shape = {p->weights.rows, p->weights.cols};
    slice.data.assign(dW.data.begin() + off, dW.data.begin() + off + n);
    if (auto st = dense::apply_grad_sgd(p, slice, lr); !st) return st;
    off += n;
  }
  return Status::Ok();
}

} // namespace welvet::swiglu
