#pragma once
#include "welvet/architecture.hpp"
#include "welvet/parallel.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/train_mode.hpp"
namespace welvet::runtime::cameral_train {
// Go TrainMSE / TrainStackMSE, including modes, overrides and sync boundaries.
Status train_mse(parallel::Layer &, const TensorF32 &, const TensorF32 &,
                 TrainMode, double, double *);
Status train_stack_mse(stack::Layer &, const TensorF32 &, const TensorF32 &,
                       TrainMode, double, double *);
Status train_volume_mse(architecture::Grid &, const TensorF32 &,
                        const TensorF32 &, TrainMode, double, double *);
Status train_child(stack::Child &, const TensorF32 &gy, const TensorF32 &input,
                   const TensorF32 &pre, const TensorF32 &post, TrainMode,
                   double, TensorF32 *gx);
Status train_parallel(parallel::Layer &, const TensorF32 &, const TensorF32 &,
                      const TensorF32 &, TrainMode, double);
} // namespace welvet::runtime::cameral_train
