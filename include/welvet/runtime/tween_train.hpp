#pragma once

#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::runtime::tween_train {

// Go TrainStackMSE(..., ModeTween): ForwardStack → MSE → reverse trainTweenLocal
// (applyProjectedTween at soft_lr = lr * 0.5 — leaf walk, no chain rule).
Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss);

} // namespace welvet::runtime::tween_train
