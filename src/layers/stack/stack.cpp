#include "welvet/stack.hpp"
#include "welvet/parallel.hpp"

#include <string>
#include <utility>

namespace welvet::stack {
namespace {

template<class F> Status with_parallel(Child& child, F fn) {
  parallel::Layer layer;
  const auto& cfg = child.parallel_nest;
  layer.cfg = {cfg.dim, cfg.out_feat, int(child.nested.size()), parallel::Combine(cfg.combine),
               cfg.seq_len, cfg.sparse_k, cfg.disagree_beta};
  layer.controls = std::move(child.controls);
  layer.accel = std::move(child.accel);
  layer.branches = std::move(child.nested);
  if (cfg.has_gate) layer.gate = std::move(child.dense_op);
  const auto st = fn(layer);
  child.controls = std::move(layer.controls);
  child.accel = std::move(layer.accel);
  child.nested = std::move(layer.branches);
  if (cfg.has_gate) child.dense_op = std::move(*layer.gate);
  return st;
}

} // namespace

void flatten_ops(Child& c, std::vector<Child*>* out) {
  if (!out) return;
  if (c.kind == Kind::NestedStack) {
    for (auto& ch : c.nested) {
      flatten_ops(ch, out);
    }
    return;
  }
  if (c.kind != Kind::None) {
    out->push_back(&c);
  }
}

void flatten_ops(Layer& root, std::vector<Child*>* out) {
  if (!out) return;
  for (auto& ch : root.children) {
    flatten_ops(ch, out);
  }
}

Status child_forward(Child& c, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "stack: nil pre/post");
  }
  switch (c.kind) {
  case Kind::Dense:
    return dense::forward_f32(c.dense_op, input, pre, post);
  case Kind::RMSNorm:
    return rmsnorm::forward(c.rmsnorm_op, input, pre, post);
  case Kind::LayerNorm:
    return layernorm::forward(c.layernorm_op, input, pre, post);
  case Kind::SwiGLU:
    return swiglu::forward(c.swiglu_op, input, pre, post);
  case Kind::Softmax:
    return softmax::forward(c.softmax_op, input, pre, post);
  case Kind::Embedding:
    return embedding::forward(c.embedding_op, input, pre, post);
  case Kind::MHA:
    return mha::forward(c.mha_op, input, pre, post);
  case Kind::CNN1:
    return cnn1::forward(c.cnn1_op, input, pre, post);
  case Kind::CNN2:
    return cnn2::forward(c.cnn2_op, input, pre, post);
  case Kind::CNN3:
    return cnn3::forward(c.cnn3_op, input, pre, post);
  case Kind::ConvT1:
    return convt1::forward(c.convt1_op, input, pre, post);
  case Kind::ConvT2:
    return convt2::forward(c.convt2_op, input, pre, post);
  case Kind::ConvT3:
    return convt3::forward(c.convt3_op, input, pre, post);
  case Kind::RNN:
    return rnn::forward(c.rnn_op, input, pre, post);
  case Kind::LSTM:
    return lstm::forward(c.lstm_op, input, pre, post);
  case Kind::Sequential:
    return sequential::forward(c.sequential_op, input, pre, post);
  case Kind::Residual:
    return residual::forward(c.residual_op, input, pre, post);
  case Kind::Mamba:
    return mamba::forward(c.mamba_op, input, pre, post);
  case Kind::Metacognition:
    return metacognition::forward(c.metacognition_op, input, pre, post);
  case Kind::GDN:
    return gdn::forward(c.gdn_op, input, pre, post);
  case Kind::KMeans:
    return kmeans::forward(c.kmeans_op, input, pre, post);
  case Kind::NestedStack: {
    Layer tmp;
    tmp.children = std::move(c.nested);
    auto st = forward(tmp, input, pre, post);
    c.nested = std::move(tmp.children);
    return st;
  }
  case Kind::Parallel:
    return with_parallel(c,[&](parallel::Layer& l){return parallel::forward(l,input,pre,post);});
  default:
    return Status::Err(StatusCode::InvalidArgument, "stack: unsupported child kind");
  }
}

Status child_backward(Child& c, const TensorF32& grad_out, const TensorF32& input,
                      const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "stack: nil grad out");
  }
  switch (c.kind) {
  case Kind::Dense:
    return dense::backward_f32(c.dense_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::RMSNorm:
    return rmsnorm::backward_f32(c.rmsnorm_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::LayerNorm:
    return layernorm::backward_f32(c.layernorm_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::SwiGLU:
    return swiglu::backward_f32(c.swiglu_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::Softmax:
    return softmax::backward_f32(c.softmax_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::Embedding:
    return embedding::backward_f32(c.embedding_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::MHA:
    return mha::backward_f32(c.mha_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::CNN1:
    return cnn1::backward_f32(c.cnn1_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::CNN2:
    return cnn2::backward_f32(c.cnn2_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::CNN3:
    return cnn3::backward_f32(c.cnn3_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::ConvT1:
    return convt1::backward_f32(c.convt1_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::ConvT2:
    return convt2::backward_f32(c.convt2_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::ConvT3:
    return convt3::backward_f32(c.convt3_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::RNN:
    return rnn::backward_f32(c.rnn_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::LSTM:
    return lstm::backward_f32(c.lstm_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::Sequential:
    return sequential::backward_f32(c.sequential_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::Residual:
    return residual::backward_f32(c.residual_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::Mamba:
    return mamba::backward_f32(c.mamba_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::Metacognition:
    return metacognition::backward_f32(c.metacognition_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::GDN:
    return gdn::backward_f32(c.gdn_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::KMeans:
    return kmeans::backward_f32(c.kmeans_op, grad_out, input, pre, grad_in, grad_w);
  case Kind::NestedStack: {
    Layer tmp;
    tmp.children = std::move(c.nested);
    auto st = backward_f32(tmp, grad_out, input, pre, grad_in, grad_w);
    c.nested = std::move(tmp.children);
    return st;
  }
  case Kind::Parallel:
    return with_parallel(c,[&](parallel::Layer& l){return parallel::backward_f32(l,grad_out,input,pre,grad_in,grad_w);});
  default:
    return Status::Err(StatusCode::InvalidArgument, "stack: unsupported child bwd");
  }
}

Status child_apply_sgd(Child* c, const TensorF32& dW, double lr) {
  if (!c) {
    return Status::Err(StatusCode::InvalidArgument, "stack: ApplyGradSGD nil");
  }
  if (lr == 0.0 || dW.data.empty()) {
    return Status::Ok();
  }
  switch (c->kind) {
  case Kind::Dense:
    return dense::apply_grad_sgd(&c->dense_op, dW, lr);
  case Kind::RMSNorm:
    return rmsnorm::apply_grad_sgd(&c->rmsnorm_op, dW, lr);
  case Kind::LayerNorm:
    return layernorm::apply_grad_sgd(&c->layernorm_op, dW, lr);
  case Kind::SwiGLU:
    return swiglu::apply_grad_sgd(&c->swiglu_op, dW, lr);
  case Kind::Softmax:
    return softmax::apply_grad_sgd(&c->softmax_op, dW, lr);
  case Kind::Embedding:
    return embedding::apply_grad_sgd(&c->embedding_op, dW, lr);
  case Kind::MHA:
    return mha::apply_grad_sgd(&c->mha_op, dW, lr);
  case Kind::CNN1:
    return cnn1::apply_grad_sgd(&c->cnn1_op, dW, lr);
  case Kind::CNN2:
    return cnn2::apply_grad_sgd(&c->cnn2_op, dW, lr);
  case Kind::CNN3:
    return cnn3::apply_grad_sgd(&c->cnn3_op, dW, lr);
  case Kind::ConvT1:
    return convt1::apply_grad_sgd(&c->convt1_op, dW, lr);
  case Kind::ConvT2:
    return convt2::apply_grad_sgd(&c->convt2_op, dW, lr);
  case Kind::ConvT3:
    return convt3::apply_grad_sgd(&c->convt3_op, dW, lr);
  case Kind::RNN:
    return rnn::apply_grad_sgd(&c->rnn_op, dW, lr);
  case Kind::LSTM:
    return lstm::apply_grad_sgd(&c->lstm_op, dW, lr);
  case Kind::Sequential:
    return sequential::apply_grad_sgd(&c->sequential_op, dW, lr);
  case Kind::Residual:
    return residual::apply_grad_sgd(&c->residual_op, dW, lr);
  case Kind::Mamba:
    return mamba::apply_grad_sgd(&c->mamba_op, dW, lr);
  case Kind::Metacognition:
    return metacognition::apply_grad_sgd(&c->metacognition_op, dW, lr);
  case Kind::GDN:
    return gdn::apply_grad_sgd(&c->gdn_op, dW, lr);
  case Kind::KMeans:
    return kmeans::apply_grad_sgd(&c->kmeans_op, dW, lr);
  case Kind::NestedStack: {
    Layer tmp;
    tmp.children = std::move(c->nested);
    auto st = apply_grad_sgd(&tmp, dW, lr);
    c->nested = std::move(tmp.children);
    return st;
  }
  case Kind::Parallel:
    return with_parallel(*c,[&](parallel::Layer& l){return parallel::apply_grad_sgd(&l,dW,lr);});
  default:
    return Status::Err(StatusCode::InvalidArgument, "stack: unsupported child sgd");
  }
}

int child_grad_w_size(const Child& c) {
  switch (c.kind) {
  case Kind::Dense:
    return c.dense_op.weights.rows * c.dense_op.weights.cols; // (out, in)
  case Kind::RMSNorm:
    return c.rmsnorm_op.cfg.dim;
  case Kind::LayerNorm:
    return 2 * c.layernorm_op.cfg.dim;
  case Kind::SwiGLU:
    return swiglu::grad_w_size(c.swiglu_op);
  case Kind::Softmax:
    return 0;
  case Kind::Embedding:
    return c.embedding_op.weights.rows * c.embedding_op.weights.cols;
  case Kind::MHA:
    return mha::grad_w_size(c.mha_op);
  case Kind::CNN1:
    return cnn1::grad_w_size(c.cnn1_op);
  case Kind::CNN2:
    return cnn2::grad_w_size(c.cnn2_op);
  case Kind::CNN3:
    return cnn3::grad_w_size(c.cnn3_op);
  case Kind::ConvT1:
    return convt1::grad_w_size(c.convt1_op);
  case Kind::ConvT2:
    return convt2::grad_w_size(c.convt2_op);
  case Kind::ConvT3:
    return convt3::grad_w_size(c.convt3_op);
  case Kind::RNN:
    return rnn::grad_w_size(c.rnn_op);
  case Kind::LSTM:
    return lstm::grad_w_size(c.lstm_op);
  case Kind::Sequential:
    return sequential::grad_w_size(c.sequential_op);
  case Kind::Residual:
    return residual::grad_w_size(c.residual_op);
  case Kind::Mamba:
    return mamba::grad_w_size(c.mamba_op);
  case Kind::Metacognition:
    return metacognition::grad_w_size(c.metacognition_op);
  case Kind::GDN:
    return gdn::grad_w_size(c.gdn_op);
  case Kind::KMeans:
    return kmeans::grad_w_size(c.kmeans_op);
  case Kind::NestedStack: {
    int n = 0;
    for (const auto& ch : c.nested) {
      n += child_grad_w_size(ch);
    }
    return n;
  }
  case Kind::Parallel: {
    int n = 0;
    for (const auto& ch : c.nested) {
      n += child_grad_w_size(ch);
    }
    if (c.parallel_nest.has_gate) {
      n += c.dense_op.weights.rows * c.dense_op.weights.cols;
    }
    return n;
  }
  default:
    return 0;
  }
}

Status child_export_f32(const Child& c, float* dst) {
  if (!dst && child_grad_w_size(c) > 0) {
    return Status::Err(StatusCode::InvalidArgument, "stack: nil export dst");
  }
  switch (c.kind) {
  case Kind::Dense:
    return c.dense_op.weights.export_f32(dst);
  case Kind::RMSNorm:
    return c.rmsnorm_op.gamma.export_f32(dst);
  case Kind::LayerNorm: {
    const int d = c.layernorm_op.cfg.dim;
    if (auto st = c.layernorm_op.gamma.export_f32(dst); !st) return st;
    return c.layernorm_op.beta.export_f32(dst + d);
  }
  case Kind::SwiGLU:
    return swiglu::export_weights_f32(c.swiglu_op, dst);
  case Kind::Softmax:
    return Status::Ok();
  case Kind::Embedding:
    return c.embedding_op.weights.export_f32(dst);
  case Kind::MHA:
    return mha::export_weights_f32(c.mha_op, dst);
  case Kind::CNN1:
    return cnn1::export_weights_f32(c.cnn1_op, dst);
  case Kind::CNN2:
    return cnn2::export_weights_f32(c.cnn2_op, dst);
  case Kind::CNN3:
    return cnn3::export_weights_f32(c.cnn3_op, dst);
  case Kind::ConvT1:
    return convt1::export_weights_f32(c.convt1_op, dst);
  case Kind::ConvT2:
    return convt2::export_weights_f32(c.convt2_op, dst);
  case Kind::ConvT3:
    return convt3::export_weights_f32(c.convt3_op, dst);
  case Kind::RNN:
    return rnn::export_packed_f32(c.rnn_op, dst);
  case Kind::LSTM:
    return lstm::export_packed_f32(c.lstm_op, dst);
  case Kind::Sequential:
    return sequential::export_packed_f32(c.sequential_op, dst);
  case Kind::Residual:
    return residual::export_packed_f32(c.residual_op, dst);
  case Kind::Mamba:
    return mamba::export_packed_f32(c.mamba_op, dst);
  case Kind::Metacognition:
    return metacognition::export_packed_f32(c.metacognition_op, dst);
  case Kind::GDN:
    return gdn::export_packed_f32(c.gdn_op, dst);
  case Kind::KMeans:
    return kmeans::export_packed_f32(c.kmeans_op, dst);
  case Kind::NestedStack: {
    int off = 0;
    for (const auto& ch : c.nested) {
      const int n = child_grad_w_size(ch);
      if (n > 0) {
        if (auto st = child_export_f32(ch, dst + off); !st) return st;
      }
      off += n;
    }
    return Status::Ok();
  }
  case Kind::Parallel: {
    int off = 0;
    for (const auto& ch : c.nested) {
      const int n = child_grad_w_size(ch);
      if (n > 0) {
        if (auto st = child_export_f32(ch, dst + off); !st) return st;
      }
      off += n;
    }
    if (c.parallel_nest.has_gate) {
      if (auto st = c.dense_op.weights.export_f32(dst + off); !st) return st;
    }
    return Status::Ok();
  }
  default:
    return Status::Err(StatusCode::InvalidArgument, "stack: unsupported child export");
  }
}

Status child_family_bp(Child& c, const TensorF32& grad_out, const TensorF32& input,
                       const TensorF32& pre, double lr, TensorF32* grad_in) {
  if (!grad_in) {
    return Status::Err(StatusCode::InvalidArgument, "stack: nil grad_in");
  }
  TensorF32 dW;
  if (auto st = child_backward(c, grad_out, input, pre, grad_in, &dW); !st) return st;
  if (lr != 0.0 && !dW.data.empty()) {
    if (auto st = child_apply_sgd(&c, dW, lr); !st) return st;
  }
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "stack: nil pre/post");
  }
  std::vector<Child*> ops;
  flatten_ops(layer, &ops);
  if (ops.empty()) {
    *pre = input;
    *post = input;
    return Status::Ok();
  }
  TensorF32 current = input;
  TensorF32 last_pre;
  for (Child* ch : ops) {
    TensorF32 p, o;
    if (auto st = child_forward(*ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("stack fwd: ") + st.message);
    }
    last_pre = std::move(p);
    current = std::move(o);
  }
  *pre = std::move(last_pre);
  *post = std::move(current);
  return Status::Ok();
}

int grad_w_size(const Layer& layer) {
  int n = 0;
  for (const auto& ch : layer.children) {
    n += child_grad_w_size(ch);
  }
  return n;
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& /*pre*/, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "stack: nil grad out");
  }
  std::vector<Child*> ops;
  flatten_ops(layer, &ops);
  const int n = static_cast<int>(ops.size());
  if (n == 0) {
    *grad_in = grad_out;
    grad_w->shape = {1, 0};
    grad_w->data.clear();
    return Status::Ok();
  }

  std::vector<TensorF32> ins(static_cast<std::size_t>(n));
  std::vector<TensorF32> pres(static_cast<std::size_t>(n));
  TensorF32 current = input;
  for (int i = 0; i < n; ++i) {
    ins[static_cast<std::size_t>(i)] = current;
    TensorF32 p, o;
    if (auto st = child_forward(*ops[static_cast<std::size_t>(i)], current, &p, &o); !st) {
      return Status::Err(st.code, std::string("stack recompute: ") + st.message);
    }
    pres[static_cast<std::size_t>(i)] = std::move(p);
    current = std::move(o);
  }

  TensorF32 gy = grad_out;
  std::vector<TensorF32> d_ws(static_cast<std::size_t>(n));
  for (int i = n - 1; i >= 0; --i) {
    TensorF32 gx, dw;
    if (auto st = child_backward(*ops[static_cast<std::size_t>(i)], gy,
                                 ins[static_cast<std::size_t>(i)],
                                 pres[static_cast<std::size_t>(i)], &gx, &dw);
        !st) {
      return Status::Err(st.code, std::string("stack bwd: ") + st.message);
    }
    d_ws[static_cast<std::size_t>(i)] = std::move(dw);
    gy = std::move(gx);
  }
  *grad_in = std::move(gy);

  const int need = grad_w_size(layer);
  grad_w->shape = {1, need};
  grad_w->data.resize(static_cast<std::size_t>(need));
  int off = 0;
  for (int i = 0; i < n; ++i) {
    const auto& dw = d_ws[static_cast<std::size_t>(i)];
    for (float v : dw.data) {
      if (off >= need) break;
      grad_w->data[static_cast<std::size_t>(off++)] = v;
    }
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "stack: ApplyGradSGD nil");
  }
  std::vector<Child*> ops;
  flatten_ops(*layer, &ops);
  int off = 0;
  for (Child* ch : ops) {
    const int cn = child_grad_w_size(*ch);
    if (cn <= 0) continue;
    if (off + cn > static_cast<int>(dW.data.size())) {
      return Status::Err(StatusCode::InvalidArgument, "stack: dW short");
    }
    TensorF32 slice;
    slice.shape = {1, cn};
    slice.data.assign(dW.data.begin() + off, dW.data.begin() + off + cn);
    if (auto st = child_apply_sgd(ch, slice, lr); !st) return st;
    off += cn;
  }
  return Status::Ok();
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst && grad_w_size(layer) > 0) {
    return Status::Err(StatusCode::InvalidArgument, "stack: nil dst");
  }
  int off = 0;
  for (const auto& ch : layer.children) {
    const int n = child_grad_w_size(ch);
    if (n > 0) {
      if (auto st = child_export_f32(ch, dst + off); !st) return st;
    }
    off += n;
  }
  return Status::Ok();
}

} // namespace welvet::stack
