#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::systems::tween_alt {

// Go trainTweenAltOp on one child: AltTimes × (Split @ lr → Tween @ lr×0.5) with frozen gap.
// Used by Line StepTweenAlt (per-hop on pop). alt_times < 1 ⇒ 1.
Status apply_alt_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                       double lr, int alt_times = 1);

// ModeTweenAlt volume: AltTimes × (Forward → Split @ full lr → Forward → Tween @ lr×0.5).
// Uses volume_dense for both forwards (same as TweenSplit/ModeTween volume wires).
Status train_volume_mse(architecture::Grid& g, const TensorF32& input, const TensorF32& target,
                        double lr, double* loss, int alt_times = 1, runtime::forward::Result* last_fwd = nullptr);

// Mesh-clock per-cell: AltTimes × (Split @ lr → Tween @ lr×0.5) with frozen gap.
Status apply_cell_alt(architecture::Cell& cell, const TensorF32& grad_out, const TensorF32& input,
                      const TensorF32& pre, const TensorF32& post, double lr, int alt_times = 1);

} // namespace welvet::systems::tween_alt
