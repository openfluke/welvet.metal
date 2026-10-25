#include "welvet/mha.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::mha {
namespace {

struct Layout {
  int batch = 1;
  int seq_len = 0;
  int d_model = 0;
  int elem_stride = 0;
};

// Accumulates into gQ [seq*qDim], gK/gV [msl*kvDim]. No dropout path (baseline).

} // namespace

double Config::scale() const {
  if (scale_override > 0) return scale_override;
  return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "mha: nil config");
  if (cfg->d_model <= 0 || cfg->num_heads <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: need DModel>0 and NumHeads>0");
  }
  if (cfg->head_dim <= 0) {
    if (cfg->d_model % cfg->num_heads != 0) {
      return Status::Err(StatusCode::InvalidArgument, "mha: HeadDim unset and DModel%NumHeads!=0");
    }
    cfg->head_dim = cfg->d_model / cfg->num_heads;
  }
  if (cfg->num_kv_heads <= 0) cfg->num_kv_heads = cfg->num_heads;
  if (cfg->num_heads % cfg->num_kv_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: NumHeads not divisible by NumKVHeads");
  }
  if (cfg->max_seq_len <= 0) cfg->max_seq_len = 512;
  if (cfg->mask == MaskKind::Unspecified) {
    cfg->mask = MaskKind::Causal;
    cfg->causal = true;
  }
  if (cfg->mask == MaskKind::Causal) cfg->causal = true;
  if (cfg->rope_theta <= 0) cfg->rope_theta = 10000;
  if (cfg->softmax != SoftmaxKind::Standard && cfg->softmax != SoftmaxKind::Sigmoid) {
    return Status::Err(StatusCode::InvalidArgument, "mha: unknown Softmax");
  }
  return Status::Ok();
}

Config decoder_causal(int d_model, int num_heads, int num_kv_heads) {
  Config c;
  c.d_model = d_model;
  c.num_heads = num_heads;
  c.num_kv_heads = num_kv_heads;
  c.mask = MaskKind::Causal;
  c.pos = PosKind::RoPE;
  c.mode = AttnMode::Self;
  c.causal = true;
  return c;
}

Status new_configured(const Config& cfg_in, DType dt, const float* q_init, const float* k_init,
                      const float* v_init, const float* o_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "mha: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  const int d = cfg.d_model;
  const int q_dim = cfg.q_dim();
  const int kv_dim = cfg.kv_dim();

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(d, q_dim, Activation::Linear, dt, q_init, &layer.Q); !st) {
    return Status::Err(st.code, std::string("mha Q: ") + st.message);
  }
  if (auto st = dense::new_layer(d, kv_dim, Activation::Linear, dt, k_init, &layer.K); !st) {
    return Status::Err(st.code, std::string("mha K: ") + st.message);
  }
  if (auto st = dense::new_layer(d, kv_dim, Activation::Linear, dt, v_init, &layer.V); !st) {
    return Status::Err(st.code, std::string("mha V: ") + st.message);
  }
  if (auto st = dense::new_layer(q_dim, d, Activation::Linear, dt, o_init, &layer.O); !st) {
    return Status::Err(st.code, std::string("mha O: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "mha.forward: Metal kernel not implemented; CPU execution is forbidden");
}

int grad_w_size(const Layer& layer) {
  auto n = [](const dense::Layer& p) { return p.weights.rows * p.weights.cols; };
  return n(layer.Q) + n(layer.K) + n(layer.V) + n(layer.O);
}

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "mha.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "mha.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

Status export_weights_f32(const Layer& layer, float* dst) {
  if (!dst) return Status::Err(StatusCode::InvalidArgument, "mha: nil dst");
  int off = 0;
  const dense::Layer* projs[] = {&layer.Q, &layer.K, &layer.V, &layer.O};
  for (const dense::Layer* p : projs) {
    const int n = p->weights.rows * p->weights.cols;
    if (auto st = p->weights.export_f32(dst + off); !st) return st;
    off += n;
  }
  return Status::Ok();
}

} // namespace welvet::mha
