#pragma once

#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::runtime::tween_split_train {

// Go TrainStackMSE(..., ModeTweenSplit): ForwardStack → MSE → trainTweenSplitOp
// (1/N projected gap per leaf @ full lr).
Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss);

} // namespace welvet::runtime::tween_split_train
