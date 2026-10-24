#pragma once

#include <string>
#include <string_view>

#include "welvet/status.hpp"

namespace welvet::metal {

struct Pipeline {
  void* mtl = nullptr; // id<MTLComputePipelineState>
  std::string name;
};

// Compile a named kernel from the bundled / runtime-compiled shader library.
Status pipeline_get(std::string_view kernel_name, Pipeline* out);
void pipeline_release(Pipeline* p);

} // namespace welvet::metal
