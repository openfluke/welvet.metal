#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>

#include "welvet/metal/device.hpp"
#include "metal/internal.hpp"

#include <mutex>
#include <sstream>

namespace {

std::mutex g_device_mu;
id<MTLDevice> g_device = nil;
std::string g_identity;

welvet::metal::DeviceInfo info_from(id<MTLDevice> d) {
  welvet::metal::DeviceInfo i;
  if (!d) return i;
  i.name = d.name ? [d.name UTF8String] : "(unnamed)";
#if TARGET_OS_OSX
  i.registry_id = std::to_string(static_cast<unsigned long long>(d.registryID));
  i.low_power = d.isLowPower;
  i.headless = d.isHeadless;
#endif
  i.recommended_max_working_set = d.recommendedMaxWorkingSetSize;
  return i;
}

id<MTLDevice> pick_device() {
#if TARGET_OS_OSX
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
  id<MTLDevice> best = nil;
  for (id<MTLDevice> d in devices) {
    if (!best) {
      best = d;
      continue;
    }
    if (best.isLowPower && !d.isLowPower) best = d;
  }
  if (best) return best;
#endif
  return MTLCreateSystemDefaultDevice();
}

} // namespace

id<MTLDevice> welvet_metal_device_handle(void) {
  std::lock_guard<std::mutex> lock(g_device_mu);
  if (!g_device) {
    g_device = pick_device();
    if (g_device) {
      auto info = info_from(g_device);
      std::ostringstream os;
      os << "metal:" << info.name;
      if (!info.registry_id.empty()) os << ":reg=" << info.registry_id;
      os << ":ws=" << info.recommended_max_working_set;
      g_identity = os.str();
    }
  }
  return g_device;
}

namespace welvet::metal {

Status init_default_device() {
  if (welvet_metal_device_handle()) return Status::Ok();
  return Status::Err(StatusCode::Unsupported, "no Metal device available");
}

bool available() { return welvet_metal_device_handle() != nil; }

Status default_device_info(DeviceInfo* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null DeviceInfo");
  id<MTLDevice> d = welvet_metal_device_handle();
  if (!d) return Status::Err(StatusCode::Unsupported, "no Metal device available");
  *out = info_from(d);
  return Status::Ok();
}

std::vector<DeviceInfo> list_devices() {
  std::vector<DeviceInfo> out;
#if TARGET_OS_OSX
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
  for (id<MTLDevice> d in devices) {
    out.push_back(info_from(d));
  }
#else
  if (id<MTLDevice> d = MTLCreateSystemDefaultDevice()) {
    out.push_back(info_from(d));
  }
#endif
  return out;
}

std::string runtime_identity() {
  (void)welvet_metal_device_handle();
  std::lock_guard<std::mutex> lock(g_device_mu);
  return g_identity.empty() ? std::string("metal:unavailable") : g_identity;
}

} // namespace welvet::metal
