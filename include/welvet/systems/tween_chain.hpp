#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::systems::tween_chain {

// ModeTweenChain on a volume tape: credit = NormalBP (volume Backward + SGD @ full lr).
// Go familyTweenChain ≡ familyBP; cameral volume Step.
Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss,
                        runtime::backward::Result* bwd_out = nullptr);

} // namespace welvet::systems::tween_chain
