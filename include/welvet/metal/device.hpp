#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "welvet/status.hpp"

namespace welvet::metal {

struct DeviceInfo {
  std::string name;
  std::string registry_id; // stable-ish identity for evidence logs
  bool low_power = false;
  bool headless = false;
  uint64_t recommended_max_working_set = 0;
};

// Process-wide default device (first discrete/high-power preferred).
Status init_default_device();
bool available();
Status default_device_info(DeviceInfo* out);
std::vector<DeviceInfo> list_devices();

// Library / pipeline compile identity for provenance in w2a logs.
std::string runtime_identity();

} // namespace welvet::metal
