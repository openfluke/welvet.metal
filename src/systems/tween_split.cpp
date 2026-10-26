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
#include <cstring>
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
  bool dw_only = false;
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
                        const TensorF32& pre, double lr, bool dw_only) {
  TensorF32 g_in;
  if(dw_only && c.kind==stack::Kind::Dense){TensorF32 dw;if(auto st=dense::grad_w_only(c.dense_op,gy,input,pre,&dw);!st)return st;
    return lr==0?Status::Ok():dense::apply_grad_sgd(&c.dense_op,dw,lr);}
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
      leaf.dw_only = true;
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
    for(size_t i=start;i<out->size();++i)(*out)[i].dw_only=true;
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
  leaf.dw_only = true;
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
    if(leaf.dw_only){TensorF32 dw;if(auto st=dense::grad_w_only(*leaf.dense,gy,leaf.input,leaf.pre,&dw);!st)return st;
      return lr==0?Status::Ok():dense::apply_grad_sgd(leaf.dense,dw,lr);}
    return apply_dense_leaf(*leaf.dense, gy, leaf.input, leaf.pre, lr);
  }
  if (leaf.child) {
    return apply_child_leaf(*leaf.child, gy, leaf.input, leaf.pre, lr,leaf.dw_only);
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

// Proxy families share the exact collect tape and native GPU update boundaries.
dense::Layer* dense_of(Leaf& l) {
  if(l.dense)return l.dense;
  if(l.child && l.child->kind==stack::Kind::Dense)return &l.child->dense_op;
  if(l.cell && l.cell->op_kind==architecture::OpKind::Dense)return &l.cell->dense_op;
  return nullptr;
}
Status leaf_backward(Leaf& l,const TensorF32& gy,TensorF32* gx,TensorF32* dw,bool refwd=false) {
  TensorF32 pre=l.pre,post;
  if(auto* d=dense_of(l))return dense::backward_f32(*d,gy,l.input,pre,gx,dw);
  if(refwd) pre={}; // Go linearGradInAny passes nil pre directly.
  if(l.child)return stack::child_backward(*l.child,gy,l.input,pre,gx,dw);
  if(l.cell)return runtime::backward::cell(*l.cell,gy,l.input,pre,gx,dw);
  return Status::Err(StatusCode::Internal,"proxy: missing leaf");
}
Status leaf_sgd(Leaf& l,const TensorF32& dw,double lr) {
  if(lr==0 || dw.data.empty())return Status::Ok();
  if(auto* d=dense_of(l))return dense::apply_grad_sgd(d,dw,lr);
  if(l.child)return stack::child_apply_sgd(l.child,dw,lr);
  return apply_cell_dw(*l.cell,dw,lr);
}
Status leaf_dw(Leaf& l,const TensorF32& gy,double lr) {
  TensorF32 dw,gx;
  if(auto* d=dense_of(l)){if(auto st=dense::grad_w_only(*d,gy,l.input,l.pre,&dw);!st)return st;}
  else if(auto st=leaf_backward(l,gy,&gx,&dw);!st)return st;
  return leaf_sgd(l,dw,lr);
}
Status leaf_linear(Leaf& l,const TensorF32& gy,TensorF32* gx,bool nil_pre=false) {
  if(auto* d=dense_of(l))return dense::linear_grad_in(*d,gy,l.input,gx);
  if(nil_pre){
    const int kind=l.child?int(l.child->kind):int(l.cell->op_kind);
    // These Go backward paths ignore pre; other Ops reject nil pre and use P(gy).
    if(kind!=int(stack::Kind::Embedding) && kind!=int(stack::Kind::Mamba) &&
       kind!=int(stack::Kind::GDN) && kind!=int(stack::Kind::Metacognition) && kind!=int(stack::Kind::NestedStack) && kind!=int(stack::Kind::Parallel))
      return tween::project_gap(gy,l.input.shape,gx);
    nil_pre=kind==int(stack::Kind::Metacognition);
  }
  TensorF32 dw;
  auto st=leaf_backward(l,gy,gx,&dw,nil_pre);
  if(!st || gx->data.empty())return tween::project_gap(gy,l.input.shape,gx);
  return Status::Ok();
}
Status add_gx(TensorF32* a,const TensorF32& b) {
  if(b.data.empty())return Status::Ok();
  if(a->data.empty()){*a=b;return Status::Ok();}
  if(a->data.size()!=b.data.size())return Status::Ok();
  return metal::run("op_add",{metal::read(a->data),metal::read(b.data),metal::write(a->data)},
                     {int(a->data.size())},{},a->data.size());
}
Status hidden_dw(std::vector<Leaf>& leaves,const TensorF32& src,double lr,int count,int selected=-1) {
  for(int i=0;i<int(leaves.size())-1;++i){
    if(selected>=0 && i!=selected)continue;
    auto& l=leaves[i];if(l.post.data.empty())continue;
    TensorF32 gi;
    if(auto st=tween::project_gap(src,l.post.shape,&gi);!st)return st;
    if(count>1)if(auto st=scale_gap(&gi,count);!st)return st;
    if(auto st=leaf_dw(l,gi,lr);!st)return st;
  }
  return Status::Ok();
}
Leaf* find_leaf(std::vector<Leaf>& leaves,stack::Child* ch,dense::Layer* d=nullptr) {
  for(auto& l:leaves)if((ch && l.child==ch)||(d && dense_of(l)==d))return &l;
  return nullptr;
}
Status linear_leaf(Leaf& l,const TensorF32& gy,double lr,int n,std::vector<Leaf>& leaves,
                    std::vector<std::vector<float>>* fill,TensorF32* gx) {
  TensorF32 gi;
  if(auto st=tween::project_gap(gy,l.post.shape,&gi);!st)return st;
  if(fill)(*fill)[size_t(&l-leaves.data())]=gi.data;
  if(auto st=scale_gap(&gi,n);!st)return st;
  if(auto st=leaf_dw(l,gi,lr);!st)return st;
  // Go walks through the UPDATED weights and skips Dense activation derivatives.
  return leaf_linear(l,gy,gx);
}
Status split_linear_parallel(stack::Child& ch,const TensorF32& gy,std::vector<Leaf>& leaves,
                               std::vector<TensorF32>* branches) {
  const auto& cfg=ch.parallel_nest;const int nb=int(ch.nested.size());
  std::vector<TensorF32> posts;std::vector<int32_t> widths;std::vector<float> packed;
  const int mode=cfg.combine==3?1:cfg.combine;
  const int rows=mode==0?int(gy.shape[0]):1;
  int out=mode==0?int(gy.shape[1]):int(gy.data.size());
  for(auto& b:ch.nested){auto* l=find_leaf(leaves,&b);posts.push_back(l?l->post:gy);
    int width=mode==0?int(posts.back().shape.back()):int(gy.data.size());widths.push_back(width);
    if(posts.back().data.size()<size_t(rows*width))return Status::Err(StatusCode::InvalidArgument,"linear Parallel branch shape");
    packed.insert(packed.end(),posts.back().data.begin(),posts.back().data.begin()+rows*width);
  }
  std::vector<float> forward(rows*out),grads(packed.size()),scratch(4*nb),unused;
  uint64_t bits;double beta=cfg.disagree_beta==0?1:cfg.disagree_beta;std::memcpy(&bits,&beta,sizeof(bits));
  parallel::Config pc;pc.branches=nb;pc.sparse_k=cfg.sparse_k;
  if(auto st=metal::run("op_parallel",{metal::read(packed),metal::read(widths),metal::read(unused),metal::read(gy.data),
    metal::write(forward),metal::write(grads),metal::write(unused),metal::write(scratch)},
    {rows,nb,out,mode,parallel::sparse_k_eff(pc),1,int(uint32_t(bits)),int(uint32_t(bits>>32))});!st)return st;
  size_t off=0;branches->clear();for(int i=0;i<nb;++i){TensorF32 g;g.shape=mode==0?std::vector<int64_t>{rows,widths[i]}:gy.shape;
    g.data.assign(grads.begin()+off,grads.begin()+off+rows*widths[i]);off+=rows*widths[i];branches->push_back(std::move(g));}
  return Status::Ok();
}
Status walk_child(stack::Child& ch,const TensorF32& down,std::vector<Leaf>& leaves,double lr,
                    std::vector<std::vector<float>>* fill,TensorF32* gx) {
  const int n=int(leaves.size());
  if(ch.kind==stack::Kind::NestedStack){TensorF32 gy=down;
    for(int i=int(ch.nested.size())-1;i>=0;--i){TensorF32 next;if(auto st=walk_child(ch.nested[i],gy,leaves,lr,fill,&next);!st)return st;if(!next.data.empty())gy=std::move(next);}*gx=std::move(gy);return Status::Ok();}
  if(ch.kind==stack::Kind::Sequential || ch.kind==stack::Kind::Residual){
    auto& children=ch.kind==stack::Kind::Sequential?ch.sequential_op.children:ch.residual_op.children;TensorF32 gy=down;
    for(int i=int(children.size())-1;i>=0;--i)if(auto* l=find_leaf(leaves,nullptr,&children[i])){
      TensorF32 next;if(auto st=linear_leaf(*l,gy,lr,n,leaves,fill,&next);!st)return st;if(!next.data.empty())gy=std::move(next);}
    if(ch.kind==stack::Kind::Residual)if(auto st=add_gx(&gy,down);!st)return st;
    *gx=std::move(gy);return Status::Ok();}
  if(ch.kind==stack::Kind::Parallel){std::vector<TensorF32> bgy;
    if(auto st=split_linear_parallel(ch,down,leaves,&bgy);!st)return st;
    *gx={};for(size_t i=0;i<ch.nested.size();++i){TensorF32 next;if(auto st=walk_child(ch.nested[i],bgy[i],leaves,lr,fill,&next);!st)return st;if(auto st=add_gx(gx,next);!st)return st;}return Status::Ok();}
  if(auto* l=find_leaf(leaves,&ch))return linear_leaf(*l,down,lr,n,leaves,fill,gx);
  *gx={};return Status::Ok(); // weightless nodes have no proxy tape entry in Go.
}
Status apply_variant_leaves(std::vector<Leaf>& leaves,const TensorF32& gy,double lr,Variant variant,
                              stack::SplitAccel* accel,stack::Layer* root=nullptr,stack::Child* child=nullptr) {
  if(int(variant)<0 || int(variant)>6)return Status::Gap("split: unsupported variant");
  int n=int(leaves.size());if(n==0)return Status::Ok();
  if(variant==Variant::Split)return apply_leaves(leaves,gy,lr);
  if(variant==Variant::Linear || variant==Variant::LinearCache){
    std::vector<float> norm(3);
    if(variant==Variant::LinearCache && accel){
      if(auto st=metal::run("op_proxy_norm",{metal::read(gy.data),metal::write(norm)},{int(gy.data.size())});!st)return st;
      const int period=accel->cache_every>0?accel->cache_every:20;
      if(accel->cache.size()==size_t(n) && accel->norm_xy.size()==3 && accel->norm_xy[2]!=0 && accel->tick%period!=0){
        ++accel->tick;
        for(int i=0;i<n;++i){auto& l=leaves[i];if(accel->cache[i].size()!=l.post.data.size() || l.post.data.empty())continue;
          TensorF32 gi;gi.shape=l.post.shape;gi.data.resize(l.post.data.size());
          if(auto st=metal::run("op_proxy_rescale",{metal::read(accel->cache[i]),metal::read(norm),metal::read(accel->norm_xy),metal::write(gi.data)},
                {int(gi.data.size())},{},gi.data.size());!st)return st;
          if(auto st=scale_gap(&gi,n);!st)return st;if(auto st=leaf_dw(l,gi,lr);!st)return st;}
        return Status::Ok();
      }
    }
    std::vector<std::vector<float>> fill;auto* filling=variant==Variant::LinearCache && accel?&fill:nullptr;if(filling)fill.resize(n);
    TensorF32 gx;
    if(root){TensorF32 current=gy;for(int i=int(root->children.size())-1;i>=0;--i){TensorF32 next;if(auto st=walk_child(root->children[i],current,leaves,lr,filling,&next);!st)return st;if(!next.data.empty())current=std::move(next);}}
    else if(child){if(auto st=walk_child(*child,gy,leaves,lr,filling,&gx);!st)return st;}
    else {TensorF32 current=gy;for(int i=n-1;i>=0;--i){TensorF32 next;if(auto st=linear_leaf(leaves[i],current,lr,n,leaves,filling,&next);!st)return st;if(!next.data.empty())current=std::move(next);}}
    if(filling){accel->cache=std::move(fill);accel->norm_xy=std::move(norm);++accel->tick;}
    return Status::Ok();
  }
  Leaf& head=leaves.back();TensorF32 ghead,proxy;
  if(auto st=tween::project_gap(gy,head.post.shape,&ghead);!st)return st;
  if(variant==Variant::HeadProxy){TensorF32 dw;if(auto st=leaf_backward(head,ghead,&proxy,&dw);!st)return st;if(auto st=leaf_sgd(head,dw,lr);!st)return st;
    if(n>1)return hidden_dw(leaves,proxy.data.empty()?gy:proxy,lr,n-1);return Status::Ok();}
  bool stale=variant==Variant::HeadProxyAsync && accel && !accel->proxy.empty() && n>1;
  if(stale){TensorF32 src;src.shape={int64_t(accel->proxy.size())};src.data=accel->proxy;if(auto st=hidden_dw(leaves,src,lr,n-1);!st)return st;}
  if(variant!=Variant::HeadProxyAsync)if(auto st=leaf_linear(head,gy,&proxy,true);!st)return st;
  if(auto st=leaf_dw(head,ghead,lr);!st)return st;
  if(variant==Variant::HeadProxyAsync){if(auto st=leaf_linear(head,gy,&proxy,true);!st)return st;if(accel)accel->proxy=proxy.data;}
  if(n<=1 || stale)return Status::Ok();
  if(variant==Variant::Sparse){int selected=0;if(accel)selected=accel->sparse++%(n-1);return hidden_dw(leaves,proxy.data.empty()?gy:proxy,lr,1,selected);}
  return hidden_dw(leaves,proxy.data.empty()?gy:proxy,lr,n-1);
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

Status apply_split(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input, double lr,bool ordinary_backward) {
  std::vector<Leaf> leaves;
  if (auto st = collect_stack_leaves(ops, input, &leaves); !st) {
    return st;
  }
  if(ordinary_backward)for(auto& leaf:leaves)leaf.dw_only=false;
  return apply_leaves(leaves, grad_out, lr);
}

Status apply_split_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                         double lr,bool ordinary_backward) {
  std::vector<Leaf> leaves;
  if (auto st = collect_child(c, input, &leaves); !st) {
    return st;
  }
  if(ordinary_backward)for(auto& leaf:leaves)leaf.dw_only=false;
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

Status train_stack_variant(stack::Layer& ops,const TensorF32& input,const TensorF32& target,double lr,double* loss,Variant variant){
  if(!loss)return Status::Err(StatusCode::InvalidArgument,"proxy: nil loss");
  std::vector<Leaf> leaves;TensorF32 post,gy;
  if(auto st=collect_stack_leaves(ops,input,&leaves,&post);!st)return st;
  if(auto st=runtime::training::mse(post,target,loss);!st)return st;
  if(auto st=runtime::training::mse_grad(post,target,&gy);!st)return st;
  return apply_variant_leaves(leaves,gy,lr,variant,&ops.accel,&ops);
}
Status apply_variant_child(stack::Child& child,const TensorF32& gy,const TensorF32& input,double lr,Variant variant){
  std::vector<Leaf> leaves;if(auto st=collect_child(child,input,&leaves);!st)return st;
  auto* accel=child.kind==stack::Kind::NestedStack || child.kind==stack::Kind::Parallel?&child.accel:nullptr;
  return apply_variant_leaves(leaves,gy,lr,variant,accel,nullptr,&child);
}
Status train_volume_variant(architecture::Grid& g,const runtime::forward::Result& fwd,const TensorF32& target,double lr,double* loss,Variant variant){
  if(!loss)return Status::Err(StatusCode::InvalidArgument,"proxy: nil loss");
  TensorF32 gy;std::vector<Leaf> leaves;
  if(auto st=runtime::training::mse(fwd.output,target,loss);!st)return st;
  if(auto st=runtime::training::mse_grad(fwd.output,target,&gy);!st)return st;
  if(auto st=collect_volume_leaves(g,fwd,&leaves);!st)return st;
  return apply_variant_leaves(leaves,gy,lr,variant,&g.accel);
}
Status apply_cell_variant(architecture::Cell& cell,const TensorF32& gy,const TensorF32& input,const TensorF32& pre,const TensorF32& post,double lr,Variant variant){
  if(cell.meta.is_disabled || input.data.empty() || gy.data.empty())return Status::Ok();
  std::vector<Leaf> leaves;if(auto st=collect_cell_leaves(cell,input,pre,post,&leaves);!st)return st;
  return apply_variant_leaves(leaves,gy,lr,variant,cell.op_kind==architecture::OpKind::Stack || cell.op_kind==architecture::OpKind::Parallel?&cell.accel:nullptr);
}

} // namespace welvet::systems::tween_split
