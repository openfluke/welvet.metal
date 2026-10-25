#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/train_mode.hpp"

namespace welvet::runtime::training {

// MSE / ∂MSE/∂pred = 2/n (pred − target) — Go training parity.
Status mse(const TensorF32& pred, const TensorF32& target, double* loss);
Status mse_grad(const TensorF32& pred, const TensorF32& target, TensorF32* grad);

// Apply GradWs to Dense cells on the grid (NormalBP).
Status sgd_dense(architecture::Grid& g, const forward::Result& fwd, const backward::Result& bwd,
                 double lr);

// One NormalBP iteration on a Dense volume: uses existing forward tape + target.
Status step_dense(architecture::Grid& g, const forward::Result& fwd, const TensorF32& target,
                  double lr, double* loss, backward::Result* bwd_out);

// Placeholder for remaining TrainModes / cameral (not Dense NormalBP).
} // namespace welvet::runtime::training
