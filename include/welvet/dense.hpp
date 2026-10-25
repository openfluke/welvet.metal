#pragma once

#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "welvet/metal/matvec.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::dense {

// Metal-only Dense. Forward uses resident soft-double W (mount once, reload after SGD).
// Backward/SGD stay on host FormatNone for Go exact_bit; GPU mirror refreshed on next forward.
struct Layer {
  int input_height = 0;
  int output_height = 0;
  Activation activation = Activation::Linear;
  WeightStore weights;

  // Resident Metal soft-double mirror of weights (not copied).
  mutable metal::ResidentDense gpu{};
  mutable bool gpu_valid = false;
  mutable int gpu_batch = -1;

  Layer() = default;
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;
  Layer(Layer&& o) noexcept
      : input_height(o.input_height), output_height(o.output_height), activation(o.activation),
        weights(std::move(o.weights)), gpu(o.gpu), gpu_valid(o.gpu_valid), gpu_batch(o.gpu_batch) {
    o.gpu = {};
    o.gpu_valid = false;
    o.gpu_batch = -1;
  }
  Layer& operator=(Layer&& o) noexcept {
    if (this == &o) return *this;
    metal::resident_release(&gpu);
    input_height = o.input_height;
    output_height = o.output_height;
    activation = o.activation;
    weights = std::move(o.weights);
    gpu = o.gpu;
    gpu_valid = o.gpu_valid;
    gpu_batch = o.gpu_batch;
    o.gpu = {};
    o.gpu_valid = false;
    o.gpu_batch = -1;
    return *this;
  }
  ~Layer() { metal::resident_release(&gpu); }

  void invalidate_gpu() const {
    gpu_valid = false;
  }
};

Status new_layer(int in_h, int out_h, Activation act, DType dt, const float* init_or_null,
                 Layer* layer);

// True when FormatNone Dense forward on Metal can match Go's f64 MatVec contract
// for this dtype. Unsupported dtypes hard-error — never a different path.
bool dtype_supported_metal_forward(DType dt);

// FormatNone Dense forward on Metal with resident W (decode+upload only when invalidated).
Status forward_f32(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// NormalBP leaf: dPre = gradOut ⊙ act'(pre); dX = Wᵀ·dPre; dW = outer(dPre, x).
// Host f64 MatVecT + outer (Go BackwardCPUTiled / welvet.cpp parity). pre = post-bias.
Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

// Apply dW via weights.ApplySGD (FormatNone native); invalidates resident GPU W.
Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

// Force-decode FormatNone → soft-double into resident buffers (or mount). Used by bench reset.
Status sync_gpu_from_weights(const Layer& layer, int batch);

} // namespace welvet::dense
