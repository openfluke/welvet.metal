#pragma once

#include <cstdint>

#include "welvet/activate.hpp"
#include "welvet/status.hpp"

namespace welvet::metal {

Status require_dense_compute();

// One-shot paths (alloc buffers each call) — used by non-resident forward.
Status dense_matvec_softdouble_batch(const float* W_softdouble_xy, const float* x, float* y,
                                     int32_t batch, int32_t rows, int32_t cols);

Status dense_bias_act(float* pre, float* post, const float* bias_softdouble_xy, int32_t bias_len,
                      int32_t batch, int32_t rows, Activation act);

inline void split_double(double d, float* hi, float* lo) {
  *hi = static_cast<float>(d);
  *lo = static_cast<float>(d - static_cast<double>(*hi));
}

// ---- Resident GPU session (mount vs execute) ----

struct ResidentDense {
  void* buf_w = nullptr;    // id<MTLBuffer> float2 W[rows*cols]
  void* buf_bias = nullptr; // id<MTLBuffer> float2 bias[bias_len] (or dummy)
  void* buf_x = nullptr;
  void* buf_pre = nullptr;
  void* buf_post = nullptr;
  int32_t batch = 0;
  int32_t rows = 0;
  int32_t cols = 0;
  int32_t bias_len = 0;
  Activation act = Activation::Linear;
};

Status resident_mount(ResidentDense* r, const float* W_xy, const float* bias_xy, int32_t bias_len,
                      int32_t batch, int32_t rows, int32_t cols, Activation act);

// Upload x, run matvec+bias_act in one command buffer, download pre/post.
Status resident_execute(ResidentDense* r, const float* x, float* pre, float* post);

void resident_release(ResidentDense* r);

} // namespace welvet::metal
