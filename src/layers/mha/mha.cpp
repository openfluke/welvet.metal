#include "welvet/metal/layer_utils.hpp"
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

Status parse_layout(int d_model, const TensorF32 &input, Layout *lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "mha: empty input");
  }
  lay->d_model = d_model;
  if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq_len = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != d_model) {
      return Status::Err(StatusCode::InvalidArgument, "mha: last dim != d_model");
    }
  } else if (input.shape.size() == 2) {
    lay->batch = 1;
    if (static_cast<int>(input.shape[1]) != d_model) {
      return Status::Err(StatusCode::InvalidArgument, "mha: width != d_model");
    }
    lay->seq_len = static_cast<int>(input.shape[0]);
  } else {
    return Status::Err(StatusCode::InvalidArgument, "mha: need [batch,seq,d] or [seq,d]");
  }
  if (lay->batch <= 0 || lay->seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: bad batch/seq");
  }
  lay->elem_stride = lay->seq_len * lay->d_model;
  const int want = lay->batch * lay->elem_stride;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "mha: short data");
  }
  return Status::Ok();
}

void copy_grad(std::vector<float> &dst, int *off, const TensorF32 &src) {
  for (float v : src.data) {
    dst[static_cast<std::size_t>((*off)++)] = v;
  }
}

} // namespace

double Config::scale() const {
  if (scale_override > 0)
    return scale_override;
  return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

Status validate(Config *cfg) {
  if (!cfg)
    return Status::Err(StatusCode::InvalidArgument, "mha: nil config");
  if (cfg->d_model <= 0 || cfg->num_heads <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: need DModel>0 and NumHeads>0");
  }
  if (cfg->head_dim <= 0) {
    if (cfg->d_model % cfg->num_heads != 0) {
      return Status::Err(StatusCode::InvalidArgument, "mha: HeadDim unset and DModel%NumHeads!=0");
    }
    cfg->head_dim = cfg->d_model / cfg->num_heads;
  }
  if (cfg->num_kv_heads <= 0)
    cfg->num_kv_heads = cfg->num_heads;
  if (cfg->num_heads % cfg->num_kv_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: NumHeads not divisible by NumKVHeads");
  }
  if (cfg->max_seq_len <= 0)
    cfg->max_seq_len = 512;
  if (cfg->mask == MaskKind::Unspecified) {
    cfg->mask = MaskKind::Causal;
    cfg->causal = true;
  }
  if (cfg->mask == MaskKind::Causal)
    cfg->causal = true;
  if (cfg->rope_theta <= 0)
    cfg->rope_theta = 10000;
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

Status new_configured(const Config &cfg_in, DType dt, const float *q_init, const float *k_init,
                      const float *v_init, const float *o_init, Layer *out) {
  if (!out)
    return Status::Err(StatusCode::InvalidArgument, "mha: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st)
    return st;

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

Status forward(Layer &layer, const TensorF32 &input, TensorF32 *pre, TensorF32 *post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "mha: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  if (cfg.mode != AttnMode::Self) {
    return Status::Gap("mha forward: ModeCross not in baseline");
  }

  Layout lay;
  if (auto st = parse_layout(cfg.d_model, input, &lay); !st)
    return st;

  const int q_dim = cfg.q_dim();
  const int kv_dim = cfg.kv_dim();
  const int msl = cfg.max_seq_len;
  const int d = cfg.d_model;
  const int bs = lay.batch * lay.seq_len;

  if (lay.seq_len > msl) {
    return Status::Err(StatusCode::InvalidArgument, "mha: seq > MaxSeqLen");
  }

  TensorF32 flat_in;
  flat_in.shape = {bs, d};
  flat_in.data.assign(input.data.begin(), input.data.begin() + static_cast<std::size_t>(bs * d));

  TensorF32 q_pre, q_post, k_pre, k_post, v_pre, v_post;
  if (auto st = dense::forward_f32(layer.Q, flat_in, &q_pre, &q_post); !st) {
    return Status::Err(st.code, std::string("mha Q: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.K, flat_in, &k_pre, &k_post); !st) {
    return Status::Err(st.code, std::string("mha K: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.V, flat_in, &v_pre, &v_post); !st) {
    return Status::Err(st.code, std::string("mha V: ") + st.message);
  }

  TensorF32 attn_flat = TensorF32::zeros(bs, q_dim);
  std::vector<float> scratch(2 * (4 * msl * kv_dim + 2 * lay.seq_len * q_dim + 3 * bs), 0), empty;
  float th, tl, sh, sl;
  metal::split_double(cfg.rope_theta, &th, &tl);
  metal::split_double(cfg.scale_override, &sh, &sl);
  if (auto st =
          metal::run("op_attention",
                     {metal::read(q_post.data), metal::read(k_post.data), metal::read(v_post.data),
                      metal::read(empty), metal::write(attn_flat.data), metal::read(empty),
                      metal::write(scratch)},
                     {lay.batch, lay.seq_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, msl,
                      int(cfg.mask), int(cfg.pos), int(cfg.softmax), cfg.window, cfg.prefix_len,
                      int(cfg.window_causal || cfg.causal), 0},
                     {th, tl, sh, sl});
      !st)
    return st;

  TensorF32 o_pre, o_post;
  if (auto st = dense::forward_f32(layer.O, attn_flat, &o_pre, &o_post); !st) {
    return Status::Err(st.code, std::string("mha O: ") + st.message);
  }

  pre->shape = {lay.batch, lay.seq_len, q_dim};
  pre->data = attn_flat.data;
  post->shape = {lay.batch, lay.seq_len, d};
  post->data = o_post.data;
  return Status::Ok();
}

int grad_w_size(const Layer &layer) {
  auto n = [](const dense::Layer &p) { return p.weights.rows * p.weights.cols; };
  return n(layer.Q) + n(layer.K) + n(layer.V) + n(layer.O);
}

Status backward_f32(const Layer &layer, const TensorF32 &grad_out, const TensorF32 &input,
                    const TensorF32 &pre, TensorF32 *grad_in, TensorF32 *grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "mha: nil grad_in/grad_w");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  if (cfg.mode != AttnMode::Self) {
    return Status::Gap("mha backward: ModeCross not in baseline");
  }

  Layout lay;
  if (auto st = parse_layout(cfg.d_model, input, &lay); !st)
    return st;

  const int q_dim = cfg.q_dim();
  const int kv_dim = cfg.kv_dim();
  const int msl = cfg.max_seq_len;
  const int d = cfg.d_model;
  const int bs = lay.batch * lay.seq_len;

  if (static_cast<int>(pre.data.size()) < bs * q_dim) {
    return Status::Err(StatusCode::InvalidArgument, "mha: pre (attn) missing or wrong size");
  }
  if (static_cast<int>(grad_out.data.size()) < bs * d) {
    return Status::Err(StatusCode::InvalidArgument, "mha: gradOut short");
  }

  TensorF32 flat_in;
  flat_in.shape = {bs, d};
  flat_in.data.assign(input.data.begin(), input.data.begin() + static_cast<std::size_t>(bs * d));

  TensorF32 grad_out_flat;
  grad_out_flat.shape = {bs, d};
  grad_out_flat.data.assign(grad_out.data.begin(),
                            grad_out.data.begin() + static_cast<std::size_t>(bs * d));

  TensorF32 attn_flat;
  attn_flat.shape = {bs, q_dim};
  attn_flat.data.assign(pre.data.begin(), pre.data.begin() + static_cast<std::size_t>(bs * q_dim));

  TensorF32 o_pre, o_post;
  if (auto st = dense::forward_f32(layer.O, attn_flat, &o_pre, &o_post); !st) {
    return Status::Err(st.code, std::string("mha O recompute: ") + st.message);
  }
  TensorF32 grad_attn, grad_wo;
  if (auto st = dense::backward_f32(layer.O, grad_out_flat, attn_flat, o_pre, &grad_attn, &grad_wo);
      !st) {
    return Status::Err(st.code, std::string("mha O bwd: ") + st.message);
  }

  TensorF32 q_pre, q_post, k_pre, k_post, v_pre, v_post;
  if (auto st = dense::forward_f32(layer.Q, flat_in, &q_pre, &q_post); !st) {
    return Status::Err(st.code, std::string("mha Q recompute: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.K, flat_in, &k_pre, &k_post); !st) {
    return Status::Err(st.code, std::string("mha K recompute: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.V, flat_in, &v_pre, &v_post); !st) {
    return Status::Err(st.code, std::string("mha V recompute: ") + st.message);
  }

  std::vector<float> grads(bs * (q_dim + 2 * kv_dim), 0), empty,
      scratch(2 * (4 * msl * kv_dim + 2 * lay.seq_len * q_dim + 3 * bs), 0);
  float th, tl, sh, sl;
  metal::split_double(cfg.rope_theta, &th, &tl);
  metal::split_double(cfg.scale_override, &sh, &sl);
  if (auto st = metal::run("op_attention",
                           {metal::read(q_post.data), metal::read(k_post.data),
                            metal::read(v_post.data), metal::read(grad_attn.data),
                            metal::read(empty), metal::write(grads), metal::write(scratch)},
                           {lay.batch, lay.seq_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                            msl, int(cfg.mask), int(cfg.pos), int(cfg.softmax), cfg.window,
                            cfg.prefix_len, int(cfg.window_causal || cfg.causal), 1},
                           {th, tl, sh, sl});
      !st)
    return st;
  TensorF32 gQT;
  gQT.shape = {bs, q_dim};
  gQT.data.assign(grads.begin(), grads.begin() + bs * q_dim);

  TensorF32 grad_in_q, grad_wq;
  if (auto st = dense::backward_f32(layer.Q, gQT, flat_in, q_pre, &grad_in_q, &grad_wq); !st) {
    return Status::Err(st.code, std::string("mha Q bwd: ") + st.message);
  }

  TensorF32 gKT, gVT;
  gKT.shape = {bs, kv_dim};
  gVT.shape = {bs, kv_dim};
  gKT.data.resize(static_cast<std::size_t>(bs * kv_dim));
  gVT.data.resize(static_cast<std::size_t>(bs * kv_dim));
  std::copy(grads.begin() + bs * q_dim, grads.begin() + bs * (q_dim + kv_dim), gKT.data.begin());
  std::copy(grads.begin() + bs * (q_dim + kv_dim), grads.end(), gVT.data.begin());

  TensorF32 grad_in_k, grad_wk, grad_in_v, grad_wv;
  if (auto st = dense::backward_f32(layer.K, gKT, flat_in, k_pre, &grad_in_k, &grad_wk); !st) {
    return Status::Err(st.code, std::string("mha K bwd: ") + st.message);
  }
  if (auto st = dense::backward_f32(layer.V, gVT, flat_in, v_pre, &grad_in_v, &grad_wv); !st) {
    return Status::Err(st.code, std::string("mha V bwd: ") + st.message);
  }

  grad_in->shape = {lay.batch, lay.seq_len, d};
  grad_in->data.resize(static_cast<std::size_t>(bs * d));
  if (auto st = metal::run("op_add3",
                           {metal::read(grad_in_q.data), metal::read(grad_in_k.data),
                            metal::read(grad_in_v.data), metal::write(grad_in->data)},
                           {bs * d}, {}, bs * d);
      !st)
    return st;

  const int nw = grad_w_size(layer);
  grad_w->shape = {nw};
  grad_w->data.assign(static_cast<std::size_t>(nw), 0.0f);
  int off = 0;
  copy_grad(grad_w->data, &off, grad_wq);
  copy_grad(grad_w->data, &off, grad_wk);
  copy_grad(grad_w->data, &off, grad_wv);
  copy_grad(grad_w->data, &off, grad_wo);
  return Status::Ok();
}

Status apply_grad_sgd(Layer *layer, const TensorF32 &dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "mha: ApplyGradSGD nil");
  }
  const int need = grad_w_size(*layer);
  if (static_cast<int>(dW.data.size()) < need) {
    return Status::Err(StatusCode::InvalidArgument, "mha: dW short");
  }
  int off = 0;
  dense::Layer *projs[] = {&layer->Q, &layer->K, &layer->V, &layer->O};
  for (dense::Layer *p : projs) {
    const int n = p->weights.rows * p->weights.cols;
    TensorF32 slice;
    slice.shape = {p->weights.rows, p->weights.cols};
    slice.data.assign(dW.data.begin() + off, dW.data.begin() + off + n);
    if (auto st = dense::apply_grad_sgd(p, slice, lr); !st)
      return st;
    off += n;
  }
  return Status::Ok();
}

Status export_weights_f32(const Layer &layer, float *dst) {
  if (!dst)
    return Status::Err(StatusCode::InvalidArgument, "mha: nil dst");
  int off = 0;
  const dense::Layer *projs[] = {&layer.Q, &layer.K, &layer.V, &layer.O};
  for (const dense::Layer *p : projs) {
    const int n = p->weights.rows * p->weights.cols;
    if (auto st = p->weights.export_f32(dst + off); !st)
      return st;
    off += n;
  }
  return Status::Ok();
}

} // namespace welvet::mha
