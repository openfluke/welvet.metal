#include "welvet/runtime/tween_alt_train.hpp"
#include "welvet/runtime/tween_split_train.hpp"

#include "welvet/runtime/training.hpp"
#include "welvet/systems/tween.hpp"
#include "welvet/systems/tween_split.hpp"

#include <string>
#include <utility>
#include <vector>

namespace welvet::runtime::tween_alt_train {
namespace {

Status forward_stack(stack::Layer& ops, const TensorF32& input, TensorF32* post) {
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  if (flat.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "tween_alt_train: empty stack");
  }
  TensorF32 current = input;
  for (stack::Child* ch : flat) {
    TensorF32 p, o;
    if (auto st = stack::child_forward(*ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("tween_alt_train fwd: ") + st.message);
    }
    current = std::move(o);
  }
  *post = std::move(current);
  return Status::Ok();
}

} // namespace

Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss, int alt_times) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween_alt_train: nil loss");
  }
  const int times = alt_times < 1 ? 1 : alt_times;
  double last = 0.0;
  for (int i = 0; i < times; ++i) {
    if (auto st = tween_split_train::train_stack_mse(ops,input,target,lr,&last); !st) return st;
    TensorF32 post, gy;
    if (auto st = forward_stack(ops, input, &post); !st) {
      return st;
    }
    if (auto st = training::mse(post, target, &last); !st) {
      return st;
    }
    if (auto st = training::mse_grad(post, target, &gy); !st) {
      return st;
    }
    const double soft_lr = lr * 0.5;
    if (auto st = systems::tween::apply_projected_ops(ops, gy, input, soft_lr); !st) {
      return Status::Err(st.code, std::string("tween_alt_train tween: ") + st.message);
    }
  }
  *loss = last;
  return Status::Ok();
}

} // namespace welvet::runtime::tween_alt_train
