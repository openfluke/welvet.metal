#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "welvet/metal/training.hpp"
#include "welvet/metal/library.hpp"
#include "welvet/metal/matvec.hpp"
#include "welvet/metal/pipeline.hpp"
#include "metal/internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <map>
#include <vector>

namespace welvet::metal {
namespace {
std::mutex training_mu;
id<MTLCommandQueue> queue = nil;
std::map<std::string, id<MTLComputePipelineState>> pipelines;

Status prepare(const char* name, id<MTLComputePipelineState>* pso) {
  if (auto st = library_ensure_loaded(); !st) return st;
  auto device = welvet_metal_device_handle();
  if (!device) return Status::Err(StatusCode::Unsupported, "Metal device unavailable; CPU execution is forbidden");
  if (!queue) queue = [device newCommandQueue];
  if (!queue) return Status::Err(StatusCode::Internal, "Metal training queue allocation failed");
  auto it = pipelines.find(name);
  if (it != pipelines.end()) { *pso = it->second; return Status::Ok(); }
  Pipeline pipe;
  if (auto st = pipeline_get(name, &pipe); !st) return st;
  *pso = (__bridge_transfer id<MTLComputePipelineState>)pipe.mtl;
  pipelines.emplace(name, *pso);
  return Status::Ok();
}

bool valid_size(std::size_t n) {
  return n > 0 && n <= std::numeric_limits<uint32_t>::max();
}

Status finish(id<MTLCommandBuffer> cmd) {
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted) {
    auto desc = cmd.error.localizedDescription;
    return Status::Err(StatusCode::Internal, std::string("Metal training failed: ") +
                       (desc ? desc.UTF8String : "command did not complete"));
  }
  return Status::Ok();
}

void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, std::size_t n) {
  [enc setComputePipelineState:pso];
  [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(n, std::min<NSUInteger>(256, pso.maxTotalThreadsPerThreadgroup)), 1, 1)];
}

Status sgd(float* weights, const void* grad, std::size_t n, double lr, bool wide) {
  if (!weights || !grad || !valid_size(n) || !std::isfinite(lr)) {
    return Status::Err(StatusCode::InvalidArgument, "Metal SGD: invalid arguments");
  }
  if (std::abs(lr) > std::numeric_limits<float>::max() ||
      (lr != 0.0 && std::abs(lr) < std::numeric_limits<float>::min())) return Status::Err(StatusCode::Unsupported, "Metal SGD: learning rate exceeds soft-double range");
  std::lock_guard<std::mutex> lock(training_mu);
  id<MTLComputePipelineState> pso;
  if (auto st = prepare(wide ? "welvet_sgd_f32_f64_grad" : "welvet_sgd_f32", &pso); !st) return st;
  auto device = welvet_metal_device_handle();
  const auto bytes = n * sizeof(float);
  auto w = [device newBufferWithBytes:weights length:bytes options:MTLResourceStorageModeShared];
  auto g = [device newBufferWithBytes:grad length:bytes * (wide ? 2 : 1) options:MTLResourceStorageModeShared];
  if (!w || !g) return Status::Err(StatusCode::Internal, "Metal SGD buffer allocation failed");
  float rate[2];
  split_double(lr, &rate[0], &rate[1]);
  const uint32_t count = static_cast<uint32_t>(n);
  auto cmd = [queue commandBuffer];
  auto enc = [cmd computeCommandEncoder];
  if (!cmd || !enc) return Status::Err(StatusCode::Internal, "Metal SGD encoder allocation failed");
  [enc setBuffer:w offset:0 atIndex:0];
  [enc setBuffer:g offset:0 atIndex:1];
  [enc setBytes:rate length:sizeof(rate) atIndex:2];
  [enc setBytes:&count length:sizeof(count) atIndex:3];
  dispatch(enc, pso, n);
  [enc endEncoding];
  if (auto st = finish(cmd); !st) return st;
  std::memcpy(weights, w.contents, bytes);
  return Status::Ok();
}
} // namespace

Status sgd_f32(float* weights, const float* grad, std::size_t n, double lr) {
  return sgd(weights, grad, n, lr, false);
}

Status sgd_f32_f64_grad(float* weights, const double* grad, std::size_t n, double lr) {
  if (!grad || !valid_size(n)) return Status::Err(StatusCode::InvalidArgument, "Metal SGD: gradient size");
  std::vector<float> xy(n * 2);
  // Representation conversion only; multiplication and subtraction run in the kernel.
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(grad[i]) || std::abs(grad[i]) > std::numeric_limits<float>::max() ||
        (grad[i] != 0.0 && std::abs(grad[i]) < std::numeric_limits<float>::min())) {
      return Status::Err(StatusCode::Unsupported, "Metal SGD: gradient exceeds soft-double range");
    }
    split_double(grad[i], &xy[i * 2], &xy[i * 2 + 1]);
  }
  return sgd(weights, xy.data(), n, lr, true);
}

Status mse_f32(const float* pred, const float* target, std::size_t n, double* loss, float* grad) {
  if (!pred || !target || (!loss && !grad) || !valid_size(n)) {
    return Status::Err(StatusCode::InvalidArgument, "Metal MSE: invalid arguments");
  }
  std::lock_guard<std::mutex> lock(training_mu);
  id<MTLComputePipelineState> loss_pso = nil, grad_pso = nil;
  if (loss) { if (auto st = prepare("welvet_mse_loss", &loss_pso); !st) return st; }
  if (grad) { if (auto st = prepare("welvet_mse_grad", &grad_pso); !st) return st; }
  auto device = welvet_metal_device_handle();
  const auto bytes = n * sizeof(float);
  auto p = [device newBufferWithBytes:pred length:bytes options:MTLResourceStorageModeShared];
  auto t = [device newBufferWithBytes:target length:bytes options:MTLResourceStorageModeShared];
  auto l = loss ? [device newBufferWithLength:2 * sizeof(float) options:MTLResourceStorageModeShared] : nil;
  auto g = grad ? [device newBufferWithLength:bytes options:MTLResourceStorageModeShared] : nil;
  if (!p || !t || (loss && !l) || (grad && !g)) return Status::Err(StatusCode::Internal, "Metal MSE buffer allocation failed");
  const uint32_t count = static_cast<uint32_t>(n);
  auto cmd = [queue commandBuffer];
  auto enc = [cmd computeCommandEncoder];
  if (!cmd || !enc) return Status::Err(StatusCode::Internal, "Metal MSE encoder allocation failed");
  [enc setBuffer:p offset:0 atIndex:0];
  [enc setBuffer:t offset:0 atIndex:1];
  [enc setBytes:&count length:sizeof(count) atIndex:3];
  if (loss) {
    [enc setBuffer:l offset:0 atIndex:2];
    dispatch(enc, loss_pso, 1);
  }
  if (grad) {
    [enc setBuffer:g offset:0 atIndex:2];
    dispatch(enc, grad_pso, n);
  }
  [enc endEncoding];
  if (auto st = finish(cmd); !st) return st;
  if (loss) {
    const auto* xy = static_cast<const float*>(l.contents);
    *loss = static_cast<double>(xy[0]) + static_cast<double>(xy[1]); // ABI scalar export
  }
  if (grad) std::memcpy(grad, g.contents, bytes);
  return Status::Ok();
}

Status accumulate_f32(float* dst, const float* src, std::size_t n) {
  if (!dst || !src || !valid_size(n)) return Status::Err(StatusCode::InvalidArgument, "Metal accumulate: invalid arguments");
  std::lock_guard<std::mutex> lock(training_mu);
  id<MTLComputePipelineState> pso;
  if (auto st = prepare("welvet_accumulate_f32", &pso); !st) return st;
  auto device = welvet_metal_device_handle();
  const auto bytes = n * sizeof(float);
  auto d = [device newBufferWithBytes:dst length:bytes options:MTLResourceStorageModeShared];
  auto s = [device newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared];
  if (!d || !s) return Status::Err(StatusCode::Internal, "Metal accumulate buffer allocation failed");
  const uint32_t count = static_cast<uint32_t>(n);
  auto cmd = [queue commandBuffer];
  auto enc = [cmd computeCommandEncoder];
  if (!cmd || !enc) return Status::Err(StatusCode::Internal, "Metal accumulate encoder allocation failed");
  [enc setBuffer:d offset:0 atIndex:0];
  [enc setBuffer:s offset:0 atIndex:1];
  [enc setBytes:&count length:sizeof(count) atIndex:2];
  dispatch(enc, pso, n);
  [enc endEncoding];
  if (auto st = finish(cmd); !st) return st;
  std::memcpy(dst, d.contents, bytes);
  return Status::Ok();
}
} // namespace welvet::metal
