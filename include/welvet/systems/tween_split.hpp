#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::systems::tween_split {

enum class Variant : int { Split=0, HeadProxy=1, Linear=2, FastProxy=3, LinearCache=4, HeadProxyAsync=5, Sparse=6 };
Status train_stack_variant(stack::Layer& ops,const TensorF32& input,const TensorF32& target,
                            double lr,double* loss,Variant variant);
Status apply_variant_child(stack::Child& child,const TensorF32& gy,const TensorF32& input,
                            double lr,Variant variant);
Status train_volume_variant(architecture::Grid& grid,const runtime::forward::Result& fwd,
                             const TensorF32& target,double lr,double* loss,Variant variant);
Status apply_cell_variant(architecture::Cell& cell,const TensorF32& gy,const TensorF32& input,
                           const TensorF32& pre,const TensorF32& post,double lr,Variant variant);

// One collect tape supplies both the loss and the per-leaf update (no re-forward).
Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target,
                       double lr, double* loss);

// ModeTweenSplit: one global gap, split 1/N across every trainable leaf
// (Go trainTweenSplitFamily / OpenSplitTape). Full lr; no chain rule.
// Alt internally uses ordinary backward (Go trainTweenSplitOp), not GradWOnly.
Status apply_split(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                   double lr, bool ordinary_backward = false);

// One Op: collect leaves under this child, then 1/N (Go trainTweenSplitFamily).
// Used by Line StepTweenSplit (per-hop on pop).
Status apply_split_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                         double lr, bool ordinary_backward = false);

// Volume: collect leaves from every enabled cell on the forward tape, then 1/N.
Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss);

// Last-tick per-cell credit for mesh scheduling.
Status apply_cell_split(architecture::Cell& cell, const TensorF32& grad_out, const TensorF32& input,
                        const TensorF32& pre, const TensorF32& post, double lr);
} // namespace welvet::systems::tween_split
