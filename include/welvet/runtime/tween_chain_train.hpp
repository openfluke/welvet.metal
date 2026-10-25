#pragma once

#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::runtime::tween_chain_train {

// Go TrainStackMSE(..., ModeTweenChain): ForwardStack → MSE → TrainStack
// FamilyTweenChain (= FamilyBP): reverse child_family_bp at full lr with true gin.
Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss);

} // namespace welvet::runtime::tween_chain_train
