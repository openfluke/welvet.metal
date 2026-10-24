#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "welvet/metal/library.hpp"
#include "welvet/metal/pipeline.hpp"
#include "metal/internal.hpp"

namespace welvet::metal {

Status pipeline_get(std::string_view kernel_name, Pipeline* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null Pipeline");
  if (kernel_name.empty()) return Status::Err(StatusCode::InvalidArgument, "empty kernel");

  auto st = library_ensure_loaded();
  if (!st) return st;

  id<MTLDevice> device = welvet_metal_device_handle();
  id<MTLLibrary> lib = welvet_metal_library_handle();
  if (!device || !lib) {
    return Status::Err(StatusCode::Internal, "Metal library not ready");
  }

  NSString* name = [[NSString alloc] initWithBytes:kernel_name.data()
                                            length:kernel_name.size()
                                          encoding:NSUTF8StringEncoding];
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  if (!fn) {
    return Status::Err(StatusCode::Unsupported,
                       std::string("Metal kernel not found: ") + std::string(kernel_name));
  }

  NSError* err = nil;
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    std::string msg = "pipeline create failed";
    if (err) msg += std::string(": ") + [[err localizedDescription] UTF8String];
    return Status::Err(StatusCode::Internal, msg);
  }

  out->mtl = (__bridge_retained void*)pso;
  out->name = std::string(kernel_name);
  return Status::Ok();
}

void pipeline_release(Pipeline* p) {
  if (!p || !p->mtl) return;
  (void)(__bridge_transfer id<MTLComputePipelineState>)p->mtl;
  p->mtl = nullptr;
  p->name.clear();
}

} // namespace welvet::metal
