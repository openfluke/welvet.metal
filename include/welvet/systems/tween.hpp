#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>
#include <vector>

namespace welvet::systems::tween {

// Resize global gap onto dst shape (Go parallel.projectGap — block average).
Status project_gap(const TensorF32& src, const std::vector<int64_t>& dst_shape, TensorF32* out);

// Go applyProjectedTween on one stack child: same global gap on every trainable
// leaf (Sequential/Residual Dense children included) — no chain rule.
Status apply_projected(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                       double soft_lr);

// Whole Stack: same global gap on every flattened child (Go applyProjectedTween(Stack)).
Status apply_projected_ops(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                           double soft_lr);

// One cell: projected gap + local Backward+SGD (FamilyTween / trainTweenLocal).
// Seq/Res use leaf-projected apply (no chain). Other kinds project onto post.
Status apply_cell_projected(architecture::Cell& cell, const TensorF32& grad_out,
                            const TensorF32& input, const TensorF32& pre, const TensorF32& post,
                            double soft_lr);

// ModeTween on a volume tape: reverse HopOrder, each cell gets projected gap +
// local Backward+SGD at soft_lr = lr * 0.5 (Go trainStackMixed + trainTweenLocal).
Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss);

} // namespace welvet::systems::tween
