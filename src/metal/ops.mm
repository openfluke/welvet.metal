#include "metal/internal.hpp"
#include "welvet/metal/library.hpp"
#include "welvet/metal/ops.hpp"
#include "welvet/metal/pipeline.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <mutex>
namespace welvet::metal {
Status run(const char *kernel, std::initializer_list<Arg> args, std::initializer_list<int32_t> ints,
           std::initializer_list<float> reals, std::size_t threads) {
  if (args.size() > 8 || ints.size() > 64 || reals.size() > 64 || !threads)
    return Status::Err(StatusCode::InvalidArgument, "Metal op: invalid argument count");
  static std::mutex mu;
  static id<MTLCommandQueue> queue = nil;
  static std::map<std::string, id<MTLComputePipelineState>> pipelines;
  std::lock_guard<std::mutex> lock(mu);
  if (auto st = library_ensure_loaded(); !st)
    return st;
  auto device = welvet_metal_device_handle();
  if (!device)
    return Status::Err(StatusCode::Unsupported,
                       "Metal device unavailable; CPU execution is forbidden");
  if (!queue)
    queue = [device newCommandQueue];
  auto &pso = pipelines[kernel];
  if (!pso) {
    Pipeline p;
    if (auto st = pipeline_get(kernel, &p); !st)
      return st;
    pso = (__bridge_transfer id<MTLComputePipelineState>)p.mtl;
  }
  std::array<id<MTLBuffer>, 8> buffers;
  auto cmd = [queue commandBuffer];
  auto enc = [cmd computeCommandEncoder];
  if (!cmd || !enc)
    return Status::Err(StatusCode::Internal, "Metal command allocation failed");
  unsigned idx = 0;
  for (const auto &arg : args) {
    if (arg.bytes && !arg.input)
      return Status::Err(StatusCode::InvalidArgument, "Metal op: null buffer");
    buffers[idx] = arg.bytes ? [device newBufferWithBytes:arg.input
                                                   length:arg.bytes
                                                  options:MTLResourceStorageModeShared]
                             : [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    if (!buffers[idx])
      return Status::Err(StatusCode::Internal, "Metal buffer allocation failed");
    [enc setBuffer:buffers[idx] offset:0 atIndex:idx];
    ++idx;
  }
  auto dummy = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
  for (; idx < 8; ++idx)
    [enc setBuffer:dummy offset:0 atIndex:idx];
  int32_t ci[64] = {};
  float cr[64] = {};
  std::copy(ints.begin(), ints.end(), ci);
  std::copy(reals.begin(), reals.end(), cr);
  [enc setBytes:ci length:sizeof(ci) atIndex:8];
  [enc setBytes:cr length:sizeof(cr) atIndex:9];
  [enc setComputePipelineState:pso];
  [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(
                                            threads, std::min<NSUInteger>(
                                                         256, pso.maxTotalThreadsPerThreadgroup)),
                                        1, 1)];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted)
    return Status::Err(StatusCode::Internal,
                       std::string("Metal op failed: ") + kernel + ": " +
                           (cmd.error ? cmd.error.localizedDescription.UTF8String : "unknown"));
  idx = 0;
  for (const auto &arg : args) {
    if (arg.output && arg.bytes)
      std::memcpy(arg.output, buffers[idx].contents, arg.bytes);
    ++idx;
  }
  return Status::Ok();
}
} // namespace welvet::metal
