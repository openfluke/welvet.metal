#pragma once

#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::runtime::tween_alt_train {

// Go TrainStackMSE(..., ModeTweenAlt): AltTimes × (fwd→Split @ lr → fwd→Tween @ lr×0.5).
// alt_times < 1 ⇒ 1 (Go altTimesOf).
Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss, int alt_times = 1);

} // namespace welvet::runtime::tween_alt_train
