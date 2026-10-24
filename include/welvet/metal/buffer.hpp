#pragma once

#include <cstddef>
#include <cstdint>

#include "welvet/status.hpp"

namespace welvet::metal {

// Opaque MTLBuffer handle owned by the engine. Host pointer may be nullptr
// until a synchronized download is requested.
struct Buffer {
  void* mtl = nullptr; // id<MTLBuffer> under ARC (retained)
  void* host = nullptr;
  std::size_t bytes = 0;
  bool device_owned = true;
};

Status buffer_alloc(std::size_t bytes, Buffer* out);
Status buffer_upload(Buffer* buf, const void* host, std::size_t bytes);
Status buffer_download(const Buffer& buf, void* host, std::size_t bytes);
void buffer_release(Buffer* buf);

} // namespace welvet::metal
