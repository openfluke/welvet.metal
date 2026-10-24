#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "welvet/metal/buffer.hpp"
#include "metal/internal.hpp"

#include <cstring>

namespace welvet::metal {

Status buffer_alloc(std::size_t bytes, Buffer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null Buffer");
  if (bytes == 0) return Status::Err(StatusCode::InvalidArgument, "zero-size buffer");

  id<MTLDevice> device = welvet_metal_device_handle();
  if (!device) return Status::Err(StatusCode::Unsupported, "Metal device unavailable");

  id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!buf) return Status::Err(StatusCode::Internal, "MTLBuffer alloc failed");

  out->mtl = (__bridge_retained void*)buf;
  out->host = [buf contents];
  out->bytes = bytes;
  out->device_owned = true;
  return Status::Ok();
}

Status buffer_upload(Buffer* buf, const void* host, std::size_t bytes) {
  if (!buf || !buf->mtl) return Status::Err(StatusCode::InvalidArgument, "null buffer");
  if (!host) return Status::Err(StatusCode::InvalidArgument, "null host");
  if (bytes > buf->bytes) return Status::Err(StatusCode::InvalidArgument, "upload overflow");

  id<MTLBuffer> mtl = (__bridge id<MTLBuffer>)buf->mtl;
  std::memcpy([mtl contents], host, bytes);
  return Status::Ok();
}

Status buffer_download(const Buffer& buf, void* host, std::size_t bytes) {
  if (!buf.mtl) return Status::Err(StatusCode::InvalidArgument, "null buffer");
  if (!host) return Status::Err(StatusCode::InvalidArgument, "null host");
  if (bytes > buf.bytes) return Status::Err(StatusCode::InvalidArgument, "download overflow");

  id<MTLBuffer> mtl = (__bridge id<MTLBuffer>)buf.mtl;
  std::memcpy(host, [mtl contents], bytes);
  return Status::Ok();
}

void buffer_release(Buffer* buf) {
  if (!buf || !buf->mtl) return;
  (void)(__bridge_transfer id<MTLBuffer>)buf->mtl;
  buf->mtl = nullptr;
  buf->host = nullptr;
  buf->bytes = 0;
}

} // namespace welvet::metal
