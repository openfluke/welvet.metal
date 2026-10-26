#include "welvet/systems/tween_split.hpp"

#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/training.hpp"
#include "welvet/systems/tween.hpp"

#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/dense.hpp"
#include "welvet/kmeans.hpp"
#include "welvet/parallel.hpp"
#include "welvet/embedding.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mamba.hpp"
#include "welvet/mha.hpp"
#include "welvet/residual.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/stack.hpp"
#include "welvet/swiglu.hpp"

#include "welvet/metal/ops.hpp"
#include <string>
#include <utility>
#include <vector>

namespace welvet::systems::tween_split {
namespace {

TensorF32 clone_t(const TensorF32& t) {
  TensorF32 o;
  o.shape = t.shape;
  o.data = t.data;
  return o;
}

TensorF32 flatten_seq_like(const TensorF32& t) {
  if (t.shape.size() == 3) {
    TensorF32 out;
    const int b = static_cast<int>(t.shape[0]);
    const int s = static_cast<int>(t.shape[1]);
    const int d = static_cast<int>(t.shape[2]);
    out.shape = {static_cast<int64_t>(b) * s, d};
    out.data = t.data;
    return out;
  }
  return t;
}

Status scale_gap(TensorF32* g, int count) {
  if (!g || count <= 0) return Status::Err(StatusCode::InvalidArgument,"split: invalid leaf count");
  return metal::run("op_split_scale", {metal::write(g->data)}, {int32_t(g->data.size()), count}, {}, g->data.size());
}

struct Leaf {
  stack::Child* child = nullptr;
  dense::Layer* dense = nullptr;
  architecture::Cell* cell = nullptr;
  TensorF32 input;
  TensorF32 pre;
  TensorF32 post;
};

Status apply_dense_leaf(dense::Layer& d, const TensorF32& gy, const TensorF32& input,
                        const TensorF32& pre, double lr) {
  if (gy.data.empty() || input.data.empty()) {
    return Status::Ok();
  }
  TensorF32 g_in, dW;
  if (auto st = dense::backward_f32(d, gy, input, pre, &g_in, &dW); !st) {
    return st;
  }
  if (lr != 0.0 && !dW.data.empty()) {
    return dense::apply_grad_sgd(&d, dW, lr);
  }
  return Status::Ok();
}

Status apply_child_leaf(stack::Child& c, const TensorF32& gy, const TensorF32& input,
                        const TensorF32& pre, double lr) {
  TensorF32 g_in;
  return stack::child_family_bp(c, gy, input, pre, lr, &g_in);
}

Status apply_cell_dw(architecture::Cell& cell, const TensorF32& dW, double lr) {
  if (dW.data.empty() || lr == 0.0) {
    return Status::Ok();
  }
  switch (cell.op_kind) {
  case architecture::OpKind::Dense:
    return dense::apply_grad_sgd(&cell.dense_op, dW, lr);
  case architecture::OpKind::RMSNorm:
    return rmsnorm::apply_grad_sgd(&cell.rmsnorm_op, dW, lr);
  case architecture::OpKind::LayerNorm:
    return layernorm::apply_grad_sgd(&cell.layernorm_op, dW, lr);
  case architecture::OpKind::SwiGLU:
    return swiglu::apply_grad_sgd(&cell.swiglu_op, dW, lr);
  case architecture::OpKind::Softmax:
    return Status::Ok();
  case architecture::OpKind::Embedding:
    return embedding::apply_grad_sgd(&cell.embedding_op, dW, lr);
  case architecture::OpKind::MHA:
    return mha::apply_grad_sgd(&cell.mha_op, dW, lr);
  case architecture::OpKind::CNN1:
    return cnn1::apply_grad_sgd(&cell.cnn1_op, dW, lr);
  case architecture::OpKind::CNN2:
    return cnn2::apply_grad_sgd(&cell.cnn2_op, dW, lr);
  case architecture::OpKind::CNN3:
    return cnn3::apply_grad_sgd(&cell.cnn3_op, dW, lr);
  case architecture::OpKind::ConvT1:
    return convt1::apply_grad_sgd(&cell.convt1_op, dW, lr);
  case architecture::OpKind::ConvT2:
    return convt2::apply_grad_sgd(&cell.convt2_op, dW, lr);
  case architecture::OpKind::ConvT3:
    return convt3::apply_grad_sgd(&cell.convt3_op, dW, lr);
  case architecture::OpKind::RNN:
    return rnn::apply_grad_sgd(&cell.rnn_op, dW, lr);
  case architecture::OpKind::LSTM:
    return lstm::apply_grad_sgd(&cell.lstm_op, dW, lr);
  case architecture::OpKind::Sequential:
    return sequential::apply_grad_sgd(&cell.sequential_op, dW, lr);
  case architecture::OpKind::Residual:
    return residual::apply_grad_sgd(&cell.residual_op, dW, lr);
  case architecture::OpKind::Mamba:
    return mamba::apply_grad_sgd(&cell.mamba_op, dW, lr);
  case architecture::OpKind::Stack:
    return stack::apply_grad_sgd(&cell.stack_op, dW, lr);
  case architecture::OpKind::Metacognition:
    return metacognition::apply_grad_sgd(&cell.metacognition_op, dW, lr);
  case architecture::OpKind::GDN:
    return gdn::apply_grad_sgd(&cell.gdn_op, dW, lr);
  case architecture::OpKind::KMeans:
    return kmeans::apply_grad_sgd(&cell.kmeans_op, dW, lr);
  case architecture::OpKind::Parallel:
    return parallel::apply_grad_sgd(&cell.parallel_op, dW, lr);
  default:
    return Status::Ok();
  }
}

Status apply_cell_leaf(architecture::Cell& cell, const TensorF32& gy, const TensorF32& input,
                       const TensorF32& pre, double lr) {
  TensorF32 g_in, dW;
  if (auto st = runtime::backward::cell(cell, gy, input, pre, &g_in, &dW); !st) {
    return st;
  }
  return apply_cell_dw(cell, dW, lr);
}

Status collect_seq_dense(sequential::Layer& layer, const TensorF32& input, std::vector<Leaf>* out) {
  TensorF32 flat_in = flatten_seq_like(input);
  TensorF32 current = flat_in;
  for (auto& ch : layer.children) {
    Leaf leaf;
    leaf.dense = &ch;
    leaf.input = current;
    TensorF32 p, o;
    if (auto st = dense::forward_f32(ch, current, &p, &o); !st) {
      return st;
    }
    leaf.pre = std::move(p);
    leaf.post = o;
    out->push_back(std::move(leaf));
    current = std::move(o);
  }
  return Status::Ok();
}

Status collect_res_dense(residual::Layer& layer, const TensorF32& input, std::vector<Leaf>* out) {
  TensorF32 flat_in = flatten_seq_like(input);
  TensorF32 current = flat_in;
  for (auto& ch : layer.children) {
    Leaf leaf;
    leaf.dense = &ch;
    leaf.input = current;
    TensorF32 p, o;
    if (auto st = dense::forward_f32(ch, current, &p, &o); !st) {
      return st;
    }
    leaf.pre = std::move(p);
    leaf.post = o;
    out->push_back(std::move(leaf));
    current = std::move(o);
  }
  return Status::Ok();
}

Status collect_child(stack::Child& c, const TensorF32& input, std::vector<Leaf>* out,
                     TensorF32* output = nullptr) {
  if (c.kind == stack::Kind::Parallel) {
    std::vector<systems::tween::ProjectedLeaf> captured;
    TensorF32 post;
    if (auto st = systems::tween::collect_leaves(c, input, &captured, &post); !st) return st;
    for (auto& source : captured) {
      Leaf leaf;
      leaf.child = source.child;
      leaf.dense = source.dense;
      leaf.input = std::move(source.input);
      leaf.pre = std::move(source.pre);
      leaf.post = std::move(source.post);
      out->push_back(std::move(leaf));
    }
    if (output) *output = std::move(post);
    return Status::Ok();
  }
  if (c.kind == stack::Kind::Sequential || c.kind == stack::Kind::Residual) {
    const TensorF32 original = input;
    const size_t start = out->size();
    Status st = c.kind == stack::Kind::Sequential ? collect_seq_dense(c.sequential_op,original,out)
                                                 : collect_res_dense(c.residual_op,original,out);
    if (!st) return st;
    if (output) {
      *output = out->size() > start ? out->back().post : original;
      if (c.kind == stack::Kind::Residual && output->data.size() == original.data.size()) {
        output->shape = original.shape;
        if (auto add = metal::run("op_add", {metal::read(original.data),metal::read(output->data),
                                  metal::write(output->data)}, {int32_t(original.data.size())}, {},
                                  original.data.size()); !add) return add;
      }
    }
    return Status::Ok();
  }
  if (c.kind == stack::Kind::NestedStack) {
    TensorF32 current = input;
    for (auto& ch : c.nested) {
      if (auto st = collect_child(ch,current,out,&current); !st) return st;
    }
    if (output) *output = std::move(current);
    return Status::Ok();
  }
  Leaf leaf;
  leaf.child = &c;
  leaf.input = input;
  if (auto st = stack::child_forward(c,input,&leaf.pre,&leaf.post); !st) return st;
  if (output) *output = leaf.post;
  if (stack::child_grad_w_size(c) > 0) out->push_back(std::move(leaf));
  return Status::Ok();
}

Status collect_cell(architecture::Cell& cell, const TensorF32& input, const TensorF32& pre,
                    const TensorF32& post, std::vector<Leaf>* out) {
  if (cell.meta.is_disabled) {
    return Status::Ok();
  }
  if (cell.op_kind == architecture::OpKind::Sequential) {
    return collect_seq_dense(cell.sequential_op, input, out);
  }
  if (cell.op_kind == architecture::OpKind::Residual) {
    return collect_res_dense(cell.residual_op, input, out);
  }
  if (cell.op_kind == architecture::OpKind::Softmax || cell.op_kind == architecture::OpKind::None) {
    return Status::Ok();
  }
  Leaf leaf;
  leaf.cell = &cell;
  leaf.input = input;
  leaf.pre = pre;
  leaf.post = post;
  out->push_back(std::move(leaf));
  return Status::Ok();
}

Status apply_one_leaf(Leaf& leaf, const TensorF32& gy, double lr) {
  if (leaf.dense) {
    return apply_dense_leaf(*leaf.dense, gy, leaf.input, leaf.pre, lr);
  }
  if (leaf.child) {
    return apply_child_leaf(*leaf.child, gy, leaf.input, leaf.pre, lr);
  }
  if (leaf.cell) {
    return apply_cell_leaf(*leaf.cell, gy, leaf.input, leaf.pre, lr);
  }
  return Status::Ok();
}

Status apply_leaves(std::vector<Leaf>& leaves, const TensorF32& grad_out, double lr) {
  const int n = static_cast<int>(leaves.size());
  if (n == 0) {
    return Status::Ok();
  }
  for (int i = 0; i < n; ++i) {
    Leaf& leaf = leaves[static_cast<std::size_t>(i)];
    if (leaf.post.data.empty()) {
      continue;
    }
    TensorF32 gy;
    if (auto st = systems::tween::project_gap(grad_out, leaf.post.shape, &gy); !st) return st;
    if (auto st = scale_gap(&gy, n); !st) return st;
    if (auto st = apply_one_leaf(leaf, gy, lr); !st) {
      return Status::Err(st.code, std::string("tween_split leaf: ") + st.message);
    }
  }
  return Status::Ok();
}


Status collect_stack_leaves(stack::Layer& ops, const TensorF32& input, std::vector<Leaf>* leaves,
                            TensorF32* output = nullptr) {
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops,&flat);
  if (flat.empty()) return Status::Err(StatusCode::InvalidArgument,"tween_split: empty stack");
  TensorF32 current = input;
  for (auto* child : flat) {
    if (auto st = collect_child(*child,current,leaves,&current); !st) return st;
  }
  if (output) *output = std::move(current);
  return Status::Ok();
}

Status collect_volume_leaves(architecture::Grid& g, const runtime::forward::Result& fwd,
                             std::vector<Leaf>* leaves) {
  for (const auto& step : fwd.steps) {
    architecture::Cell* cell =
        architecture::at(g, step.coord.z, step.coord.y, step.coord.x, step.coord.l);
    if (!cell) continue;
    if (auto st = collect_cell(*cell, step.input, step.pre, step.post, leaves); !st) {
      return st;
    }
  }
  return Status::Ok();
}

Status collect_cell_leaves(architecture::Cell& cell, const TensorF32& input, const TensorF32& pre,
                           const TensorF32& post, std::vector<Leaf>* leaves) {
  if (!leaves) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split: nil leaves");
  }
  leaves->clear();
  auto expand_dense_chain = [&](std::vector<dense::Layer>& children) -> Status {
    TensorF32 cur = flatten_seq_like(input);
    for (auto& ch : children) {
      TensorF32 p, o;
      if (auto st = dense::forward_f32(ch, cur, &p, &o); !st) {
        return st;
      }
      Leaf lf;
      lf.dense = &ch;
      lf.input = cur;
      lf.pre = std::move(p);
      lf.post = o;
      leaves->push_back(lf);
      cur = std::move(o);
    }
    return Status::Ok();
  };
  if (cell.op_kind == architecture::OpKind::Sequential) {
    return expand_dense_chain(cell.sequential_op.children);
  }
  if (cell.op_kind == architecture::OpKind::Residual) {
    return expand_dense_chain(cell.residual_op.children);
  }
  Leaf lf;
  lf.cell = &cell;
  lf.input = input;
  lf.pre = pre;
  lf.post = post;
  leaves->push_back(lf);
  return Status::Ok();
}

} // namespace

Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target,
                       double lr, double* loss) {
  if (!loss) return Status::Err(StatusCode::InvalidArgument,"tween_split: nil loss");
  std::vector<Leaf> leaves;
  TensorF32 output, gy;
  if (auto st = collect_stack_leaves(ops,input,&leaves,&output); !st) return st;
  if (auto st = runtime::training::mse(output,target,loss); !st) return st;
  if (auto st = runtime::training::mse_grad(output,target,&gy); !st) return st;
  return apply_leaves(leaves,gy,lr);
}

Status apply_split(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input, double lr) {
  std::vector<Leaf> leaves;
  if (auto st = collect_stack_leaves(ops, input, &leaves); !st) {
    return st;
  }
  return apply_leaves(leaves, grad_out, lr);
}

Status apply_split_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                         double lr) {
  std::vector<Leaf> leaves;
  if (auto st = collect_child(c, input, &leaves); !st) {
    return st;
  }
  return apply_leaves(leaves, grad_out, lr);
}

Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split: nil loss");
  }
  if (fwd.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split: empty forward tape");
  }
  if (auto st = runtime::training::mse(fwd.output, target, loss); !st) {
    return st;
  }
  TensorF32 gy;
  if (auto st = runtime::training::mse_grad(fwd.output, target, &gy); !st) {
    return st;
  }
  std::vector<Leaf> leaves;
  if (auto st = collect_volume_leaves(g, fwd, &leaves); !st) {
    return st;
  }
  return apply_leaves(leaves, gy, lr);
}

Status apply_cell_split(architecture::Cell& cell, const TensorF32& grad_out, const TensorF32& input,
                        const TensorF32& pre, const TensorF32& post, double lr) {
  if (cell.meta.is_disabled || grad_out.data.empty() || input.data.empty()) {
    return Status::Ok();
  }
  std::vector<Leaf> leaves;
  if (auto st = collect_cell_leaves(cell, input, pre, post, &leaves); !st) {
    return st;
  }
  return apply_leaves(leaves, grad_out, lr);
}

} // namespace welvet::systems::tween_split
