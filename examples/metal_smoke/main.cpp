#include "welvet/gaps.hpp"
#include "welvet/metal/buffer.hpp"
#include "welvet/metal/device.hpp"
#include "welvet/metal/library.hpp"
#include "welvet/metal/pipeline.hpp"
#include "welvet_c.h"

#include <cstdio>
#include <iostream>
#include <vector>

int main() {
  std::printf("welvet.metal %s\n", welvet_version());

  auto st = welvet::metal::init_default_device();
  if (!st) {
    std::fprintf(stderr, "init: %s\n", st.message.c_str());
    return 1;
  }
  std::printf("runtime: %s\n", welvet::metal::runtime_identity().c_str());

  std::cout << "devices:\n";
  for (const auto& d : welvet::metal::list_devices()) {
    std::cout << "  " << d.name << " ws=" << d.recommended_max_working_set
              << (d.low_power ? " low-power" : "") << "\n";
  }

  st = welvet::metal::library_ensure_loaded();
  if (!st) {
    std::fprintf(stderr, "library: %s\n", st.message.c_str());
    return 1;
  }
  std::printf("shaders: %s\n", welvet::metal::library_source_path().c_str());

  welvet::metal::Pipeline pipe;
  st = welvet::metal::pipeline_get("welvet_smoke_saxpy", &pipe);
  if (!st) {
    std::fprintf(stderr, "pipeline: %s\n", st.message.c_str());
    return 1;
  }
  std::printf("pipeline: %s ok\n", pipe.name.c_str());
  welvet::metal::pipeline_release(&pipe);

  // Shared buffer round-trip smoke (no dispatch yet — dispatch lands with Dense).
  constexpr std::size_t n = 4;
  welvet::metal::Buffer buf;
  st = welvet::metal::buffer_alloc(n * sizeof(float), &buf);
  if (!st) {
    std::fprintf(stderr, "buffer: %s\n", st.message.c_str());
    return 1;
  }
  const float host_in[n] = {1, 2, 3, 4};
  float host_out[n] = {};
  st = welvet::metal::buffer_upload(&buf, host_in, sizeof(host_in));
  if (!st) {
    std::fprintf(stderr, "upload: %s\n", st.message.c_str());
    return 1;
  }
  st = welvet::metal::buffer_download(buf, host_out, sizeof(host_out));
  welvet::metal::buffer_release(&buf);
  if (!st) {
    std::fprintf(stderr, "download: %s\n", st.message.c_str());
    return 1;
  }
  std::printf("buffer round-trip: [%g, %g, %g, %g]\n", host_out[0], host_out[1], host_out[2],
              host_out[3]);

  std::cout << "\n";
  welvet::gaps::print_census(std::cout);
  return (host_out[0] == 1.f && host_out[3] == 4.f) ? 0 : 1;
}
