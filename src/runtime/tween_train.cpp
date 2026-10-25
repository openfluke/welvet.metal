#include "welvet/runtime/tween_train.hpp"

#include "welvet/runtime/training.hpp"
#include "welvet/systems/tween.hpp"

#include <string>
#include <utility>

namespace welvet::runtime::tween_train {

Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween_train: nil loss");
  }
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  const int n = static_cast<int>(flat.size());
  if (n == 0) {
    return Status::Err(StatusCode::InvalidArgument, "tween_train: empty stack");
  }

  std::vector<TensorF32> ins(static_cast<std::size_t>(n));
  std::vector<TensorF32> pres(static_cast<std::size_t>(n));
  std::vector<TensorF32> posts(static_cast<std::size_t>(n));
  TensorF32 current = input;
  for (int i = 0; i < n; ++i) {
    ins[static_cast<std::size_t>(i)] = current;
    TensorF32 p, o;
    if (auto st = stack::child_forward(*flat[static_cast<std::size_t>(i)], current, &p, &o); !st) {
      return Status::Err(st.code, std::string("tween_train fwd: ") + st.message);
    }
    pres[static_cast<std::size_t>(i)] = std::move(p);
    posts[static_cast<std::size_t>(i)] = o;
    current = std::move(o);
  }

  if (auto st = training::mse(current, target, loss); !st) {
    return st;
  }
  TensorF32 gy;
  if (auto st = training::mse_grad(current, target, &gy); !st) {
    return st;
  }

  const double soft_lr = lr * 0.5;
  for (int i = n - 1; i >= 0; --i) {
    (void)posts[static_cast<std::size_t>(i)];
    (void)pres[static_cast<std::size_t>(i)];
    // Go trainTweenLocal → applyProjectedTween (leaf walk; Seq/Res not FamilyBP).
    if (auto st = systems::tween::apply_projected(*flat[static_cast<std::size_t>(i)], gy,
                                                   ins[static_cast<std::size_t>(i)], soft_lr);
        !st) {
      return Status::Err(st.code, std::string("tween_train leaf: ") + st.message);
    }
    // Go trainTweenLocal grad_in: copy gap when lengths match input, else zeros.
    TensorF32 next;
    next.shape = ins[static_cast<std::size_t>(i)].shape;
    next.data.assign(ins[static_cast<std::size_t>(i)].data.size(), 0.0f);
    if (gy.data.size() == next.data.size()) {
      next.data = gy.data;
    }
    gy = std::move(next);
  }
  return Status::Ok();
}

} // namespace welvet::runtime::tween_train
