#include "welvet/systems/tween_chain.hpp"

#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/training.hpp"

namespace welvet::systems::tween_chain {

Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss,
                        runtime::backward::Result* bwd_out) {
  // Named TweenChain, credit = BP — same as training::step_dense.
  return runtime::training::step_dense(g, fwd, target, lr, loss, bwd_out);
}

} // namespace welvet::systems::tween_chain
