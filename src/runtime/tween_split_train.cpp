#include "welvet/runtime/tween_split_train.hpp"
#include "welvet/systems/tween_split.hpp"
namespace welvet::runtime::tween_split_train {
Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target,
                       double lr, double* loss) {
  return systems::tween_split::train_stack_mse(ops,input,target,lr,loss);
}
} // namespace welvet::runtime::tween_split_train
