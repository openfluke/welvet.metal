#include "welvet/runtime/tween_chain_train.hpp"

#include "welvet/runtime/training.hpp"
#include "welvet/stack.hpp"

#include <string>
#include <utility>

namespace welvet::runtime::tween_chain_train {

Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target, double lr,
                       double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween_chain_train: nil loss");
  }
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  const int n = static_cast<int>(flat.size());
  if (n == 0) {
    return Status::Err(StatusCode::InvalidArgument, "tween_chain_train: empty stack");
  }

  std::vector<TensorF32> ins(static_cast<std::size_t>(n));
  std::vector<TensorF32> pres(static_cast<std::size_t>(n));
  TensorF32 current = input;
  for (int i = 0; i < n; ++i) {
    ins[static_cast<std::size_t>(i)] = current;
    TensorF32 p, o;
    if (auto st = stack::child_forward(*flat[static_cast<std::size_t>(i)], current, &p, &o); !st) {
      return Status::Err(st.code, std::string("tween_chain_train fwd: ") + st.message);
    }
    pres[static_cast<std::size_t>(i)] = std::move(p);
    current = std::move(o);
  }

  if (auto st = training::mse(current, target, loss); !st) {
    return st;
  }
  TensorF32 gy;
  if (auto st = training::mse_grad(current, target, &gy); !st) {
    return st;
  }

  // FamilyTweenChain = FamilyBP @ full lr; true chain-rule gin.
  for (int i = n - 1; i >= 0; --i) {
    TensorF32 g_in;
    if (auto st = stack::child_family_bp(*flat[static_cast<std::size_t>(i)], gy,
                                          ins[static_cast<std::size_t>(i)],
                                          pres[static_cast<std::size_t>(i)], lr, &g_in);
        !st) {
      return Status::Err(st.code, std::string("tween_chain_train bp: ") + st.message);
    }
    gy = std::move(g_in);
  }
  return Status::Ok();
}

} // namespace welvet::runtime::tween_chain_train
