#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "welvet/metal/library.hpp"
#include "welvet/metal/matvec.hpp"
#include "welvet/metal/pipeline.hpp"
#include "metal/internal.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

namespace welvet::metal {
namespace {

std::mutex g_mu;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_pso_matvec = nil;
id<MTLComputePipelineState> g_pso_bias_act = nil;
id<MTLComputePipelineState> g_pso_act_deriv = nil;
id<MTLComputePipelineState> g_pso_matvect = nil;
id<MTLComputePipelineState> g_pso_outer = nil;
bool g_ready = false;
bool g_probed = false;
std::string g_err;

Status ensure_runtime() {
  auto st = library_ensure_loaded();
  if (!st) {
    g_probed = true;
    g_ready = false;
    g_err = st.message;
    return st;
  }
  id<MTLDevice> device = welvet_metal_device_handle();
  if (!device) {
    return Status::Err(StatusCode::Unsupported, "Metal device unavailable");
  }
  if (!g_queue) {
    g_queue = [device newCommandQueue];
    if (!g_queue) return Status::Err(StatusCode::Internal, "MTLCommandQueue create failed");
  }
  if (!g_pso_matvec) {
    Pipeline pipe;
    st = pipeline_get("welvet_dense_matvec_softdouble_batch", &pipe);
    if (!st) {
      g_probed = true;
      g_ready = false;
      g_err = "Metal Dense soft-double MatVec kernel missing: " + st.message;
      return Status::Err(StatusCode::Unsupported, g_err);
    }
    g_pso_matvec = (__bridge_transfer id<MTLComputePipelineState>)pipe.mtl;
    pipe.mtl = nullptr;
  }
  if (!g_pso_bias_act) {
    Pipeline pipe;
    st = pipeline_get("welvet_dense_bias_act", &pipe);
    if (!st) {
      g_probed = true;
      g_ready = false;
      g_err = "Metal Dense bias_act kernel missing: " + st.message;
      return Status::Err(StatusCode::Unsupported, g_err);
    }
    g_pso_bias_act = (__bridge_transfer id<MTLComputePipelineState>)pipe.mtl;
    pipe.mtl = nullptr;
  }
  if (!g_pso_act_deriv) {
    Pipeline pipe;
    st = pipeline_get("welvet_dense_act_deriv", &pipe);
    if (!st) {
      g_probed = true;
      g_ready = false;
      g_err = "Metal Dense act_deriv kernel missing: " + st.message;
      return Status::Err(StatusCode::Unsupported, g_err);
    }
    g_pso_act_deriv = (__bridge_transfer id<MTLComputePipelineState>)pipe.mtl;
    pipe.mtl = nullptr;
  }
  if (!g_pso_matvect) {
    Pipeline pipe;
    st = pipeline_get("welvet_dense_matvect_softdouble_batch", &pipe);
    if (!st) {
      g_probed = true;
      g_ready = false;
      g_err = "Metal Dense MatVecT kernel missing: " + st.message;
      return Status::Err(StatusCode::Unsupported, g_err);
    }
    g_pso_matvect = (__bridge_transfer id<MTLComputePipelineState>)pipe.mtl;
    pipe.mtl = nullptr;
  }
  if (!g_pso_outer) {
    Pipeline pipe;
    st = pipeline_get("welvet_dense_outer_batch", &pipe);
    if (!st) {
      g_probed = true;
      g_ready = false;
      g_err = "Metal Dense outer kernel missing: " + st.message;
      return Status::Err(StatusCode::Unsupported, g_err);
    }
    g_pso_outer = (__bridge_transfer id<MTLComputePipelineState>)pipe.mtl;
    pipe.mtl = nullptr;
  }
  g_probed = true;
  g_ready = true;
  g_err.clear();
  return Status::Ok();
}

} // namespace

Status require_dense_compute() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_probed && !g_ready) {
    return Status::Err(StatusCode::Unsupported,
                       g_err.empty() ? "Metal Dense compute unavailable" : g_err);
  }
  return ensure_runtime();
}

Status dense_matvec_softdouble_batch(const float* W_xy, const float* x, float* y, int32_t batch,
                                     int32_t rows, int32_t cols) {
  if (!W_xy || !x || !y) return Status::Err(StatusCode::InvalidArgument, "null matvec ptr");
  if (batch <= 0 || rows <= 0 || cols <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "bad matvec dims");
  }

  std::lock_guard<std::mutex> lock(g_mu);
  auto st = ensure_runtime();
  if (!st) return st;

  id<MTLDevice> device = welvet_metal_device_handle();
  const NSUInteger w_bytes =
      static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(cols) * 2 * sizeof(float);
  const NSUInteger x_bytes =
      static_cast<NSUInteger>(batch) * static_cast<NSUInteger>(cols) * sizeof(float);
  const NSUInteger y_bytes =
      static_cast<NSUInteger>(batch) * static_cast<NSUInteger>(rows) * sizeof(float);

  id<MTLBuffer> bufW =
      [device newBufferWithBytes:W_xy length:w_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufX =
      [device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufY = [device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
  if (!bufW || !bufX || !bufY) {
    return Status::Err(StatusCode::Internal, "MTLBuffer alloc failed for matvec");
  }

  uint32_t in_dim = static_cast<uint32_t>(cols);
  uint32_t out_dim = static_cast<uint32_t>(rows);
  uint32_t batch_u = static_cast<uint32_t>(batch);

  id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:g_pso_matvec];
  [enc setBuffer:bufW offset:0 atIndex:0];
  [enc setBuffer:bufX offset:0 atIndex:1];
  [enc setBuffer:bufY offset:0 atIndex:2];
  [enc setBytes:&in_dim length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&out_dim length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&batch_u length:sizeof(uint32_t) atIndex:5];

  const NSUInteger total = static_cast<NSUInteger>(batch) * static_cast<NSUInteger>(rows);
  const NSUInteger tg = std::min<NSUInteger>(g_pso_matvec.maxTotalThreadsPerThreadgroup, 256);
  [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.error) {
    NSString* desc = cmd.error.localizedDescription;
    return Status::Err(StatusCode::Internal,
                       std::string("Metal matvec failed: ") + (desc ? [desc UTF8String] : "?"));
  }
  std::memcpy(y, [bufY contents], y_bytes);
  return Status::Ok();
}

Status dense_bias_act(float* pre, float* post, const float* bias_xy, int32_t bias_len,
                      int32_t batch, int32_t rows, Activation act) {
  if (!pre || !post) return Status::Err(StatusCode::InvalidArgument, "null bias_act ptr");
  if (batch <= 0 || rows <= 0) return Status::Err(StatusCode::InvalidArgument, "bad bias_act dims");
  if (bias_len < 0 || (bias_len > 0 && !bias_xy)) {
    return Status::Err(StatusCode::InvalidArgument, "bad bias");
  }

  std::lock_guard<std::mutex> lock(g_mu);
  auto st = ensure_runtime();
  if (!st) return st;

  id<MTLDevice> device = welvet_metal_device_handle();
  const NSUInteger n = static_cast<NSUInteger>(batch) * static_cast<NSUInteger>(rows);
  const NSUInteger bytes = n * sizeof(float);

  id<MTLBuffer> bufPre =
      [device newBufferWithBytes:pre length:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufPost = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufBias = nil;
  if (bias_len > 0) {
    bufBias = [device newBufferWithBytes:bias_xy
                                  length:static_cast<NSUInteger>(bias_len) * 2 * sizeof(float)
                                 options:MTLResourceStorageModeShared];
  } else {
    float z[2] = {0, 0};
    bufBias = [device newBufferWithBytes:z length:sizeof(z) options:MTLResourceStorageModeShared];
  }
  if (!bufPre || !bufPost || !bufBias) {
    return Status::Err(StatusCode::Internal, "MTLBuffer alloc failed for bias_act");
  }

  uint32_t out_dim = static_cast<uint32_t>(rows);
  uint32_t batch_u = static_cast<uint32_t>(batch);
  int32_t act_i = static_cast<int32_t>(act);
  uint32_t bias_u = static_cast<uint32_t>(bias_len);

  id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:g_pso_bias_act];
  [enc setBuffer:bufPre offset:0 atIndex:0];
  [enc setBuffer:bufPost offset:0 atIndex:1];
  [enc setBuffer:bufBias offset:0 atIndex:2];
  [enc setBytes:&out_dim length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&batch_u length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&act_i length:sizeof(int32_t) atIndex:5];
  [enc setBytes:&bias_u length:sizeof(uint32_t) atIndex:6];

  const NSUInteger tg = std::min<NSUInteger>(g_pso_bias_act.maxTotalThreadsPerThreadgroup, 256);
  [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.error) {
    NSString* desc = cmd.error.localizedDescription;
    return Status::Err(StatusCode::Internal,
                       std::string("Metal bias_act failed: ") + (desc ? [desc UTF8String] : "?"));
  }
  std::memcpy(pre, [bufPre contents], bytes);
  std::memcpy(post, [bufPost contents], bytes);
  return Status::Ok();
}

Status resident_mount(ResidentDense* r, const float* W_xy, const float* bias_xy, int32_t bias_len,
                      int32_t batch, int32_t rows, int32_t cols, Activation act) {
  if (!r || !W_xy) return Status::Err(StatusCode::InvalidArgument, "null resident mount");
  if (batch <= 0 || rows <= 0 || cols <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "bad resident dims");
  }
  if (bias_len < 0 || (bias_len > 0 && !bias_xy)) {
    return Status::Err(StatusCode::InvalidArgument, "bad bias");
  }

  std::lock_guard<std::mutex> lock(g_mu);
  auto st = ensure_runtime();
  if (!st) return st;

  auto clear_buf = [](void** p) {
    if (*p) {
      (void)(__bridge_transfer id<MTLBuffer>)*p;
      *p = nullptr;
    }
  };
  clear_buf(&r->buf_w);
  clear_buf(&r->buf_bias);
  clear_buf(&r->buf_x);
  clear_buf(&r->buf_pre);
  clear_buf(&r->buf_post);

  id<MTLDevice> device = welvet_metal_device_handle();
  const NSUInteger w_bytes =
      static_cast<NSUInteger>(rows) * static_cast<NSUInteger>(cols) * 2 * sizeof(float);
  const NSUInteger x_bytes =
      static_cast<NSUInteger>(batch) * static_cast<NSUInteger>(cols) * sizeof(float);
  const NSUInteger y_bytes =
      static_cast<NSUInteger>(batch) * static_cast<NSUInteger>(rows) * sizeof(float);

  id<MTLBuffer> bufW =
      [device newBufferWithBytes:W_xy length:w_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufX = [device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufPre = [device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufPost = [device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufBias = nil;
  if (bias_len > 0) {
    bufBias = [device newBufferWithBytes:bias_xy
                                  length:static_cast<NSUInteger>(bias_len) * 2 * sizeof(float)
                                 options:MTLResourceStorageModeShared];
  } else {
    float z[2] = {0, 0};
    bufBias = [device newBufferWithBytes:z length:sizeof(z) options:MTLResourceStorageModeShared];
  }
  if (!bufW || !bufX || !bufPre || !bufPost || !bufBias) {
    return Status::Err(StatusCode::Internal, "resident MTLBuffer alloc failed");
  }

  r->buf_w = (__bridge_retained void*)bufW;
  r->buf_bias = (__bridge_retained void*)bufBias;
  r->buf_x = (__bridge_retained void*)bufX;
  r->buf_pre = (__bridge_retained void*)bufPre;
  r->buf_post = (__bridge_retained void*)bufPost;
  r->batch = batch;
  r->rows = rows;
  r->cols = cols;
  r->bias_len = bias_len;
  r->act = act;
  return Status::Ok();
}

Status resident_reload_w(ResidentDense* r, const float* W_xy, const float* bias_xy, int32_t bias_len) {
  if (!r || !r->buf_w || !W_xy) {
    return Status::Err(StatusCode::InvalidArgument, "null resident reload");
  }
  if (bias_len < 0 || (bias_len > 0 && !bias_xy)) {
    return Status::Err(StatusCode::InvalidArgument, "bad bias reload");
  }
  if (bias_len != r->bias_len) {
    return Status::Err(StatusCode::InvalidArgument, "bias_len changed; remount required");
  }
  id<MTLBuffer> bufW = (__bridge id<MTLBuffer>)r->buf_w;
  const NSUInteger w_bytes =
      static_cast<NSUInteger>(r->rows) * static_cast<NSUInteger>(r->cols) * 2 * sizeof(float);
  std::memcpy([bufW contents], W_xy, w_bytes);
  if (bias_len > 0 && r->buf_bias) {
    id<MTLBuffer> bufBias = (__bridge id<MTLBuffer>)r->buf_bias;
    std::memcpy([bufBias contents], bias_xy,
                static_cast<NSUInteger>(bias_len) * 2 * sizeof(float));
  }
  return Status::Ok();
}

Status resident_execute(ResidentDense* r, const float* x, float* pre, float* post) {
  if (!r || !r->buf_w || !x || !pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "null resident execute");
  }

  std::lock_guard<std::mutex> lock(g_mu);
  auto st = ensure_runtime();
  if (!st) return st;

  id<MTLBuffer> bufW = (__bridge id<MTLBuffer>)r->buf_w;
  id<MTLBuffer> bufBias = (__bridge id<MTLBuffer>)r->buf_bias;
  id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)r->buf_x;
  id<MTLBuffer> bufPre = (__bridge id<MTLBuffer>)r->buf_pre;
  id<MTLBuffer> bufPost = (__bridge id<MTLBuffer>)r->buf_post;

  const NSUInteger x_bytes =
      static_cast<NSUInteger>(r->batch) * static_cast<NSUInteger>(r->cols) * sizeof(float);
  const NSUInteger y_bytes =
      static_cast<NSUInteger>(r->batch) * static_cast<NSUInteger>(r->rows) * sizeof(float);
  std::memcpy([bufX contents], x, x_bytes);

  uint32_t in_dim = static_cast<uint32_t>(r->cols);
  uint32_t out_dim = static_cast<uint32_t>(r->rows);
  uint32_t batch_u = static_cast<uint32_t>(r->batch);
  int32_t act_i = static_cast<int32_t>(r->act);
  uint32_t bias_u = static_cast<uint32_t>(r->bias_len);

  id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

  // MatVec
  [enc setComputePipelineState:g_pso_matvec];
  [enc setBuffer:bufW offset:0 atIndex:0];
  [enc setBuffer:bufX offset:0 atIndex:1];
  [enc setBuffer:bufPre offset:0 atIndex:2];
  [enc setBytes:&in_dim length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&out_dim length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&batch_u length:sizeof(uint32_t) atIndex:5];
  const NSUInteger total = static_cast<NSUInteger>(r->batch) * static_cast<NSUInteger>(r->rows);
  const NSUInteger tg = std::min<NSUInteger>(g_pso_matvec.maxTotalThreadsPerThreadgroup, 256);
  [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

  // Bias + act (reads/writes pre, writes post)
  [enc setComputePipelineState:g_pso_bias_act];
  [enc setBuffer:bufPre offset:0 atIndex:0];
  [enc setBuffer:bufPost offset:0 atIndex:1];
  [enc setBuffer:bufBias offset:0 atIndex:2];
  [enc setBytes:&out_dim length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&batch_u length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&act_i length:sizeof(int32_t) atIndex:5];
  [enc setBytes:&bias_u length:sizeof(uint32_t) atIndex:6];
  const NSUInteger tg2 = std::min<NSUInteger>(g_pso_bias_act.maxTotalThreadsPerThreadgroup, 256);
  [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];

  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.error) {
    NSString* desc = cmd.error.localizedDescription;
    return Status::Err(StatusCode::Internal,
                       std::string("Metal resident execute failed: ") +
                           (desc ? [desc UTF8String] : "?"));
  }
  std::memcpy(pre, [bufPre contents], y_bytes);
  std::memcpy(post, [bufPost contents], y_bytes);
  return Status::Ok();
}

Status resident_backward(ResidentDense* r, const float* grad_out, const float* input,
                         const float* pre, float* grad_in, float* grad_w) {
  if (!r || !r->buf_w || !grad_out || !input || !pre || !grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "null resident backward");
  }

  std::lock_guard<std::mutex> lock(g_mu);
  auto st = ensure_runtime();
  if (!st) return st;

  id<MTLDevice> device = welvet_metal_device_handle();
  id<MTLBuffer> bufW = (__bridge id<MTLBuffer>)r->buf_w;

  const NSUInteger y_bytes =
      static_cast<NSUInteger>(r->batch) * static_cast<NSUInteger>(r->rows) * sizeof(float);
  const NSUInteger x_bytes =
      static_cast<NSUInteger>(r->batch) * static_cast<NSUInteger>(r->cols) * sizeof(float);
  const NSUInteger dw_bytes =
      static_cast<NSUInteger>(r->rows) * static_cast<NSUInteger>(r->cols) * sizeof(float);

  id<MTLBuffer> bufGy = [device newBufferWithBytes:grad_out length:y_bytes
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufPre = [device newBufferWithBytes:pre length:y_bytes
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufDPre = [device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufX = [device newBufferWithBytes:input length:x_bytes
                                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufGx = [device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufDW = [device newBufferWithLength:dw_bytes options:MTLResourceStorageModeShared];
  if (!bufGy || !bufPre || !bufDPre || !bufX || !bufGx || !bufDW) {
    return Status::Err(StatusCode::Internal, "resident bwd MTLBuffer alloc failed");
  }

  uint32_t in_dim = static_cast<uint32_t>(r->cols);
  uint32_t out_dim = static_cast<uint32_t>(r->rows);
  uint32_t batch_u = static_cast<uint32_t>(r->batch);
  int32_t act_i = static_cast<int32_t>(r->act);

  id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

  // act′
  [enc setComputePipelineState:g_pso_act_deriv];
  [enc setBuffer:bufGy offset:0 atIndex:0];
  [enc setBuffer:bufPre offset:0 atIndex:1];
  [enc setBuffer:bufDPre offset:0 atIndex:2];
  [enc setBytes:&out_dim length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&batch_u length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&act_i length:sizeof(int32_t) atIndex:5];
  const NSUInteger total_y = static_cast<NSUInteger>(r->batch) * static_cast<NSUInteger>(r->rows);
  NSUInteger tg = std::min<NSUInteger>(g_pso_act_deriv.maxTotalThreadsPerThreadgroup, 256);
  [enc dispatchThreads:MTLSizeMake(total_y, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

  // MatVecT
  [enc setComputePipelineState:g_pso_matvect];
  [enc setBuffer:bufW offset:0 atIndex:0];
  [enc setBuffer:bufDPre offset:0 atIndex:1];
  [enc setBuffer:bufGx offset:0 atIndex:2];
  [enc setBytes:&in_dim length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&out_dim length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&batch_u length:sizeof(uint32_t) atIndex:5];
  const NSUInteger total_x = static_cast<NSUInteger>(r->batch) * static_cast<NSUInteger>(r->cols);
  tg = std::min<NSUInteger>(g_pso_matvect.maxTotalThreadsPerThreadgroup, 256);
  [enc dispatchThreads:MTLSizeMake(total_x, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

  // Outer
  [enc setComputePipelineState:g_pso_outer];
  [enc setBuffer:bufDPre offset:0 atIndex:0];
  [enc setBuffer:bufX offset:0 atIndex:1];
  [enc setBuffer:bufDW offset:0 atIndex:2];
  [enc setBytes:&in_dim length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&out_dim length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&batch_u length:sizeof(uint32_t) atIndex:5];
  const NSUInteger total_w = static_cast<NSUInteger>(r->rows) * static_cast<NSUInteger>(r->cols);
  tg = std::min<NSUInteger>(g_pso_outer.maxTotalThreadsPerThreadgroup, 256);
  [enc dispatchThreads:MTLSizeMake(total_w, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.error) {
    NSString* desc = cmd.error.localizedDescription;
    return Status::Err(StatusCode::Internal,
                       std::string("Metal resident backward failed: ") +
                           (desc ? [desc UTF8String] : "?"));
  }
  std::memcpy(grad_in, [bufGx contents], x_bytes);
  std::memcpy(grad_w, [bufDW contents], dw_bytes);
  return Status::Ok();
}

void resident_release(ResidentDense* r) {
  if (!r) return;
  auto drop = [](void** p) {
    if (p && *p) {
      (void)(__bridge_transfer id<MTLBuffer>)*p;
      *p = nullptr;
    }
  };
  // May be called under g_mu from mount, or unlocked from session_release.
  drop(&r->buf_w);
  drop(&r->buf_bias);
  drop(&r->buf_x);
  drop(&r->buf_pre);
  drop(&r->buf_post);
  r->batch = r->rows = r->cols = r->bias_len = 0;
}

} // namespace welvet::metal
