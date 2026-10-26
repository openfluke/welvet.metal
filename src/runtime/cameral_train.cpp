#include "welvet/runtime/cameral_train.hpp"
#include "welvet/cam_sync.hpp"
#include "welvet/metal/ops.hpp"
#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/line.hpp"
#include "welvet/runtime/training.hpp"
#include "welvet/runtime/tween_alt_train.hpp"
#include "welvet/runtime/tween_split_train.hpp"
#include "welvet/systems/tween.hpp"
#include "welvet/systems/tween_alt.hpp"
#include "welvet/systems/tween_split.hpp"
#include <cmath>
#include <cstring>

namespace welvet::runtime::cameral_train {
namespace {
Status invalid(const char *msg) {
  return Status::Err(StatusCode::InvalidArgument, msg);
}
bool mixed(const cameral::Controls &c) {
  for (auto m : c.modes)
    if (m != TrainMode::Inherit)
      return true;
  return false;
}
bool nested_mixed(const std::vector<stack::Child> &children) {
  for (auto &c : children) {
    if (mixed(c.controls))
      return true;
    if (c.kind == stack::Kind::NestedStack && nested_mixed(c.nested))
      return true;
  }
  return false;
}
TrainFamily family(TrainMode m) {
  switch (m) {
  case TrainMode::Tween:
  case TrainMode::StepTween:
  case TrainMode::MeshTween:
    return TrainFamily::Tween;
  case TrainMode::TweenChain:
  case TrainMode::StepTweenChain:
  case TrainMode::MeshTweenChain:
    return TrainFamily::TweenChain;
  case TrainMode::TweenAlt:
  case TrainMode::StepTweenAlt:
  case TrainMode::MeshTweenAlt:
    return TrainFamily::TweenAlt;
  default:
    if ((int(m) >= 10 && int(m) <= 11) || (int(m) >= 14 && int(m) <= 29))
      return TrainFamily::TweenSplit;
    return TrainFamily::BP;
  }
}
systems::tween_split::Variant variant(TrainMode m) {
  int v = 0;
  if (int(m) >= 14 && int(m) <= 19)
    v = int(m) - 13;
  if (int(m) >= 24 && int(m) <= 29)
    v = int(m) - 23;
  if (m == TrainMode::MeshTweenSplitFastProxy)
    v = 3;
  if (m == TrainMode::MeshTweenSplitSparse)
    v = 6;
  return systems::tween_split::Variant(v);
}
TensorF32 pass_gap(const TensorF32 &gy, const TensorF32 &input) {
  TensorF32 gx;
  gx.shape = input.shape;
  gx.data.assign(input.data.size(), 0);
  if (gx.data.size() == gy.data.size())
    gx.data = gy.data;
  return gx;
}
TensorF32 branch_input(const stack::Child &child, const TensorF32 &x) {
  TensorF32 out = x;
  if (child.kind == stack::Kind::Dense && x.shape.size() == 3)
    out.shape = {x.shape[0] * x.shape[1], x.shape[2]};
  return out;
}
template <class F> Status with_parallel(stack::Child &c, F fn) {
  parallel::Layer l;
  auto cfg = c.parallel_nest;
  l.cfg = {cfg.dim,
           cfg.out_feat,
           int(c.nested.size()),
           parallel::Combine(cfg.combine),
           cfg.seq_len,
           cfg.sparse_k,
           cfg.disagree_beta};
  l.branches = std::move(c.nested);
  l.controls = std::move(c.controls);
  l.accel = std::move(c.accel);
  if (cfg.has_gate)
    l.gate = std::move(c.dense_op);
  auto st = fn(l);
  c.nested = std::move(l.branches);
  c.controls = std::move(l.controls);
  c.accel = std::move(l.accel);
  if (cfg.has_gate)
    c.dense_op = std::move(*l.gate);
  return st;
}
template <class F> Status as_child(parallel::Layer &l, F fn) {
  stack::Child c;
  c.kind = stack::Kind::Parallel;
  c.parallel_nest = {int(l.cfg.combine), l.cfg.dim,      l.cfg.out_feat,
                     l.cfg.seq_len,      l.cfg.sparse_k, l.cfg.disagree_beta,
                     l.gate ? 1 : 0};
  c.nested = std::move(l.branches);
  c.controls = std::move(l.controls);
  c.accel = std::move(l.accel);
  if (l.gate)
    c.dense_op = std::move(*l.gate);
  auto st = fn(c);
  l.branches = std::move(c.nested);
  l.controls = std::move(c.controls);
  l.accel = std::move(c.accel);
  if (l.gate)
    *l.gate = std::move(c.dense_op);
  return st;
}
Status train_children(std::vector<stack::Child> &children,
                      const cameral::Controls &ctl, const TensorF32 &gy,
                      const TensorF32 &input, TrainMode parent, double lr) {
  std::vector<TensorF32> inputs(children.size()), pres(children.size()),
      posts(children.size());
  TensorF32 current = input;
  for (size_t i = 0; i < children.size(); ++i) {
    inputs[i] = current;
    if (auto st =
            stack::child_forward(children[i], current, &pres[i], &posts[i]);
        !st)
      return st;
    current = posts[i];
  }
  current = gy;
  for (int i = int(children.size()) - 1; i >= 0; --i) {
    TrainMode mode = resolve_train_mode(
        i < int(ctl.modes.size()) ? ctl.modes[i] : TrainMode::Inherit, parent);
    TensorF32 next;
    if (auto st = train_child(children[i], current, inputs[i], pres[i],
                              posts[i], mode, lr, &next);
        !st)
      return st;
    current = std::move(next);
  }
  return Status::Ok();
}
Status train_mixed(parallel::Layer &l, const TensorF32 &gy,
                   const TensorF32 &input, TrainMode parent, double lr) {
  const int nb = int(l.branches.size());
  if (nb == 0)
    return Status::Ok();
  std::vector<TensorF32> inputs(nb), pres(nb), posts(nb), grads;
  for (int i = 0; i < nb; ++i) {
    inputs[i] = branch_input(l.branches[i], input);
    if (auto st =
            stack::child_forward(l.branches[i], inputs[i], &pres[i], &posts[i]);
        !st)
      return st;
  }
  if (auto st = parallel::split_combine_grad(l, gy, posts, &grads); !st)
    return st;
  int teacher = -1;
  for (int i = 0; i < nb; ++i)
    if (train_mode_is_shadow(parallel::effective_branch_mode(l, i, parent))) {
      teacher = i;
      break;
    }
  if (teacher >= 0) {
    double coef = l.controls.shadow_coef == 0 ? 1 : l.controls.shadow_coef;
    uint64_t bits;
    std::memcpy(&bits, &coef, 8);
    for (int i = 0; i < nb; ++i) {
      auto mode = parallel::effective_branch_mode(l, i, parent);
      if (train_mode_is_frozen(mode) ||
          posts[i].data.size() != posts[teacher].data.size() ||
          grads[i].data.size() != posts[i].data.size())
        continue;
      if (auto st = metal::run(
              "op_cam_shadow",
              {metal::read(grads[i].data), metal::read(posts[i].data),
               metal::read(posts[teacher].data), metal::write(grads[i].data)},
              {int(grads[i].data.size()), int(uint32_t(bits)),
               int(uint32_t(bits >> 32))},
              {}, grads[i].data.size());
          !st)
        return st;
    }
  }
  const bool mem_ok = l.controls.surprise_thresh <= 0 ||
                      l.controls.last_loss >= l.controls.surprise_thresh;
  for (int i = 0; i < nb; ++i) {
    auto mode = parallel::effective_branch_mode(l, i, parent);
    if (train_mode_is_frozen(mode) || (mode == TrainMode::Memory && !mem_ok))
      continue;
    double blr =
        lr *
        (i < int(l.controls.branch_lrs.size()) ? l.controls.branch_lrs[i] : 1);
    if (blr == 0)
      continue;
    if (mode == TrainMode::Adversarial)
      blr = -blr;
    if (mode == TrainMode::Adversarial || mode == TrainMode::Memory)
      mode = resolve_train_mode(parent, TrainMode::NormalBP);
    TensorF32 gx;
    if (auto st = train_child(l.branches[i], grads[i], inputs[i], pres[i],
                              posts[i], mode, blr, &gx);
        !st)
      return st;
  }
  if (l.gate && family(parent) != TrainFamily::Tween) {
    TensorF32 x = input;
    if (x.shape.size() == 3)
      x.shape = {x.shape[0] * x.shape[1], x.shape[2]};
    TensorF32 p, o, gx, dw, gate_gy;
    if (auto st = dense::forward_f32(*l.gate, x, &p, &o); st) {
      if (auto st2 = parallel::approx_gate_grad(l, gy, &gate_gy); !st2)
        return st2;
      if (auto st2 = dense::backward_f32(*l.gate, gate_gy, x, p, &gx, &dw);
          st2 && !dw.data.empty() && lr != 0) {
        if (auto st3 = dense::apply_grad_sgd(&*l.gate, dw, lr); !st3)
          return st3;
      }
    }
  }
  return Status::Ok();
}
Status sync_sample(parallel::Layer &l) {
  if (auto st = cameral::maybe_sync(l, cameral::SyncWhen::AfterStep); !st)
    return st;
  return cameral::maybe_sync(l, cameral::SyncWhen::AfterSample);
}
Status sync_sample(stack::Layer &l) {
  if (auto st = cameral::maybe_sync(l, cameral::SyncWhen::AfterStep); !st)
    return st;
  return cameral::maybe_sync(l, cameral::SyncWhen::AfterSample);
}
} // namespace
Status train_parallel(parallel::Layer &l, const TensorF32 &gy,
                      const TensorF32 &input, const TensorF32 &pre,
                      TrainMode parent, double lr) {
  parent = resolve_train_mode(parent, TrainMode::NormalBP);
  if (train_mode_is_frozen(parent))
    return Status::Ok();
  auto fam = family(parent);
  if (fam == TrainFamily::TweenSplit)
    return as_child(l, [&](stack::Child &c) {
      return systems::tween_split::apply_variant_child(c, gy, input, lr,
                                                       variant(parent));
    });
  if (fam == TrainFamily::TweenAlt)
    return as_child(l, [&](stack::Child &c) {
      return systems::tween_alt::apply_alt_child(c, gy, input, lr);
    });
  if (!parallel::has_mixed_branch_modes(l) &&
      (fam == TrainFamily::BP || fam == TrainFamily::TweenChain)) {
    TensorF32 gx, dw;
    if (auto st = parallel::backward_f32(l, gy, input, pre, &gx, &dw); !st)
      return st;
    return parallel::apply_grad_sgd(&l, dw, lr);
  }
  return train_mixed(l, gy, input, parent, lr);
}
Status train_child(stack::Child &c, const TensorF32 &gy, const TensorF32 &input,
                   const TensorF32 &pre, const TensorF32 &post, TrainMode mode,
                   double lr, TensorF32 *gx) {
  mode = resolve_train_mode(mode, TrainMode::NormalBP);
  if (!gx || int(mode) > 33 || !std::isfinite(lr))
    return invalid("cameral invalid mode/rate/output");
  if (train_mode_is_frozen(mode)) {
    TensorF32 dw;
    return stack::child_backward(c, gy, input, pre, gx, &dw);
  }
  auto fam = family(mode);
  if (fam == TrainFamily::Tween || fam == TrainFamily::TweenSplit ||
      fam == TrainFamily::TweenAlt) {
    Status st;
    if (fam == TrainFamily::Tween)
      st = systems::tween::apply_projected(c, gy, input, lr * 0.5);
    else if (fam == TrainFamily::TweenSplit)
      st = systems::tween_split::apply_variant_child(c, gy, input, lr,
                                                     variant(mode));
    else
      st = systems::tween_alt::apply_alt_child(c, gy, input, lr);
    *gx = pass_gap(gy, input);
    return st;
  }
  if (c.kind == stack::Kind::Parallel)
    return with_parallel(c, [&](parallel::Layer &l) {
      if (parallel::has_mixed_branch_modes(l)) {
        if (auto st = train_mixed(l, gy, input, mode, lr); !st)
          return st;
        TensorF32 dw;
        return parallel::backward_f32(l, gy, input, pre, gx, &dw);
      }
      TensorF32 dw;
      if (auto st = parallel::backward_f32(l, gy, input, pre, gx, &dw); !st)
        return st;
      return parallel::apply_grad_sgd(&l, dw, lr);
    });
  if (c.kind == stack::Kind::NestedStack &&
      (mixed(c.controls) || nested_mixed(c.nested))) {
    if (auto st = train_children(c.nested, c.controls, gy, input, mode, lr);
        !st)
      return st;
    TensorF32 dw;
    return stack::child_backward(c, gy, input, pre, gx, &dw);
  }
  return stack::child_family_bp(c, gy, input, pre, lr, gx);
}
Status train_mse(parallel::Layer &l, const TensorF32 &input,
                 const TensorF32 &target, TrainMode mode, double lr,
                 double *loss) {
  if (!loss || int(mode) > 33 || !std::isfinite(lr))
    return invalid("cameral TrainMSE arguments");
  mode = resolve_train_mode(mode, TrainMode::NormalBP);
  auto fam = family(mode);
  if (fam == TrainFamily::TweenSplit || fam == TrainFamily::TweenAlt)
    return as_child(l, [&](stack::Child &c) {
      stack::Layer root;
      root.accel = std::move(c.accel);
      root.children.push_back(std::move(c));
      Status st;
      if (fam == TrainFamily::TweenSplit)
        st = systems::tween_split::train_stack_variant(root, input, target, lr,
                                                       loss, variant(mode));
      else
        st = runtime::tween_alt_train::train_stack_mse(root, input, target, lr,
                                                       loss);
      c = std::move(root.children[0]);
      c.accel = std::move(root.accel);
      return st;
    });
  TensorF32 pre, post, gy;
  if (auto st = parallel::forward(l, input, &pre, &post); !st)
    return st;
  if (auto st = training::mse(post, target, loss); !st)
    return st;
  if (auto st = training::mse_grad(post, target, &gy); !st)
    return st;
  l.controls.last_loss = *loss;
  if (auto st = train_parallel(l, gy, input, pre, mode, lr); !st)
    return st;
  return sync_sample(l);
}
Status train_stack_mse(stack::Layer &l, const TensorF32 &input,
                       const TensorF32 &target, TrainMode mode, double lr,
                       double *loss) {
  if (!loss || l.children.empty() || int(mode) > 33 || !std::isfinite(lr))
    return invalid("cameral TrainStackMSE arguments");
  mode = resolve_train_mode(mode, TrainMode::NormalBP);
  const bool line_mode =
      mode == TrainMode::StepBP || mode == TrainMode::StepTween ||
      mode == TrainMode::StepTweenChain || mode == TrainMode::StepTweenSplit ||
      mode == TrainMode::StepTweenAlt || (int(mode) >= 24 && int(mode) <= 29);
  if (line_mode) {
    int credit = mode == TrainMode::StepTween        ? 1
                 : mode == TrainMode::StepTweenSplit ? 2
                 : mode == TrainMode::StepTweenAlt   ? 3
                 : int(mode) >= 24                   ? int(mode) - 20
                                                     : 0;
    if (!l.line_state)
      l.line_state = std::make_shared<line::State>();
    if (auto st = line::train_mse(l, l.line_state.get(), input, target, lr,
                                  loss, line::Credit(credit));
        !st)
      return st;
    return sync_sample(l);
  }
  Status status;
  auto fam = family(mode);
  if (!train_mode_is_frozen(mode) && fam == TrainFamily::TweenSplit)
    status = systems::tween_split::train_stack_variant(l, input, target, lr,
                                                       loss, variant(mode));
  else if (!train_mode_is_frozen(mode) && fam == TrainFamily::TweenAlt)
    status =
        runtime::tween_alt_train::train_stack_mse(l, input, target, lr, loss);
  else {
    TensorF32 pre, post, gy;
    status = stack::forward(l, input, &pre, &post);
    if (!status)
      return status;
    if (auto st = training::mse(post, target, loss); !st)
      return st;
    if (!train_mode_is_frozen(mode)) {
      if (auto st = training::mse_grad(post, target, &gy); !st)
        return st;
      if (mixed(l.controls) || nested_mixed(l.children) ||
          fam == TrainFamily::Tween)
        status = train_children(l.children, l.controls, gy, input, mode, lr);
      else {
        TensorF32 gx, dw;
        status = stack::backward_f32(l, gy, input, pre, &gx, &dw);
        if (status && !dw.data.empty())
          status = stack::apply_grad_sgd(&l, dw, lr);
      }
    }
  }
  if (!status)
    return status;
  return sync_sample(l);
}
Status train_volume_mse(architecture::Grid &g, const TensorF32 &input,
                        const TensorF32 &target, TrainMode mode, double lr,
                        double *loss) {
  if (!loss || int(mode) > 33 || !std::isfinite(lr))
    return invalid("cameral volume arguments");
  mode = resolve_train_mode(mode, TrainMode::NormalBP);
  if (!train_mode_is_frozen(mode) && family(mode) != TrainFamily::BP)
    return Status::Gap("cameral volume: BP-gated modifiers only");
  forward::Result fwd;
  if (auto st = forward::volume_dense(g, input, &fwd); !st)
    return st;
  if (auto st = training::mse(fwd.output, target, loss); !st)
    return st;
  if (train_mode_is_frozen(mode) || lr == 0)
    return Status::Ok();
  TensorF32 gy;
  if (auto st = training::mse_grad(fwd.output, target, &gy); !st)
    return st;
  backward::Result bwd;
  std::vector<TensorF32> gaps(fwd.steps.size());
  for (int i = int(fwd.steps.size()) - 1; i >= 0; --i) {
    auto &step = fwd.steps[i];
    auto *c = architecture::at(g, step.coord.z, step.coord.y, step.coord.x,
                               step.coord.l);
    TensorF32 gx, dw;
    gaps[i] = gy;
    if (auto st = backward::cell(*c, gy, step.input, step.pre, &gx, &dw); !st)
      return st;
    gy = std::move(gx);
    if (c->op_kind != architecture::OpKind::Parallel &&
        c->op_kind != architecture::OpKind::Stack && !dw.data.empty())
      bwd.grad_ws.push_back({step.coord, std::move(dw)});
  }
  if (auto st = training::sgd_dense(g, fwd, bwd, lr); !st)
    return st;
  for (size_t i = 0; i < fwd.steps.size(); ++i) {
    auto &step = fwd.steps[i];
    auto *c = architecture::at(g, step.coord.z, step.coord.y, step.coord.x,
                               step.coord.l);
    if (c->op_kind == architecture::OpKind::Parallel) {
      c->parallel_op.controls.last_loss = *loss;
      if (auto st = train_parallel(c->parallel_op, gaps[i], step.input,
                                   step.pre, mode, lr);
          !st)
        return st;
      if (auto st = sync_sample(c->parallel_op); !st)
        return st;
    }
    if (c->op_kind == architecture::OpKind::Stack) {
      auto &l = c->stack_op;
      if (mixed(l.controls) || nested_mixed(l.children)) {
        if (auto st = train_children(l.children, l.controls, gaps[i],
                                     step.input, mode, lr);
            !st)
          return st;
      } else {
        TensorF32 gx, dw;
        if (auto st =
                stack::backward_f32(l, gaps[i], step.input, step.pre, &gx, &dw);
            !st)
          return st;
        if (auto st = stack::apply_grad_sgd(&l, dw, lr); !st)
          return st;
      }
      if (auto st = sync_sample(l); !st)
        return st;
    }
  }
  return Status::Ok();
}
} // namespace welvet::runtime::cameral_train
