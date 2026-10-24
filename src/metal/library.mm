#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "welvet/metal/library.hpp"
#include "metal/internal.hpp"

#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

namespace {

std::mutex g_lib_mu;
id<MTLLibrary> g_lib = nil;
std::string g_path;

std::string default_shader_path() {
#ifdef WELVET_SHADER_DIR
  return std::string(WELVET_SHADER_DIR) + "/welvet_dense.metal";
#else
  return "shaders/welvet_dense.metal";
#endif
}

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

welvet::Status compile_at(const std::string& path) {
  id<MTLDevice> device = welvet_metal_device_handle();
  if (!device) {
    return welvet::Status::Err(welvet::StatusCode::Unsupported, "Metal device unavailable");
  }

  std::string src = read_file(path);
  if (src.empty()) {
    return welvet::Status::Err(welvet::StatusCode::Internal,
                               "cannot read Metal shader source: " + path);
  }

  NSError* err = nil;
  NSString* ns = [[NSString alloc] initWithBytes:src.data()
                                          length:src.size()
                                        encoding:NSUTF8StringEncoding];
  MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
  // Soft-double / Go f64-parity requires exact IEEE ops — never fast-math.
  opts.fastMathEnabled = NO;
  id<MTLLibrary> lib = [device newLibraryWithSource:ns options:opts error:&err];
  if (!lib) {
    std::string msg = "Metal compile failed";
    if (err) msg += std::string(": ") + [[err localizedDescription] UTF8String];
    return welvet::Status::Err(welvet::StatusCode::Internal, msg);
  }
  g_lib = lib;
  g_path = path;
  return welvet::Status::Ok();
}

} // namespace

id<MTLLibrary> welvet_metal_library_handle(void) {
  // Caller must library_ensure_loaded() first (avoids recursive g_lib_mu lock).
  std::lock_guard<std::mutex> lock(g_lib_mu);
  return g_lib;
}

namespace welvet::metal {

Status library_ensure_loaded() {
  std::lock_guard<std::mutex> lock(g_lib_mu);
  if (g_lib) return Status::Ok();
  return compile_at(default_shader_path());
}

bool library_ready() {
  std::lock_guard<std::mutex> lock(g_lib_mu);
  return g_lib != nil;
}

std::string library_source_path() {
  std::lock_guard<std::mutex> lock(g_lib_mu);
  return g_path.empty() ? default_shader_path() : g_path;
}

Status library_reload(std::string_view optional_override_path) {
  std::lock_guard<std::mutex> lock(g_lib_mu);
  g_lib = nil;
  const std::string path = optional_override_path.empty() ? default_shader_path()
                                                          : std::string(optional_override_path);
  return compile_at(path);
}

} // namespace welvet::metal
