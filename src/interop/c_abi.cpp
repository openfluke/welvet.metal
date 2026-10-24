#include "welvet_c.h"

#include "welvet/dense.hpp"
#include "welvet/gaps.hpp"
#include "welvet/metal/device.hpp"
#include "welvet/status.hpp"

#include <cstring>
#include <string>

namespace {

welvet_status to_c(const welvet::Status& st) {
  welvet_status out{};
  out.code = static_cast<int32_t>(st.code);
  std::strncpy(out.message, st.message.c_str(), sizeof(out.message) - 1);
  out.message[sizeof(out.message) - 1] = '\0';
  return out;
}

} // namespace

extern "C" {

const char* welvet_version(void) {
  // Same formula as welvet.cpp / Go: 0.{round(earned)} until 100 → 1.0.0.
  // Engine board starts at 0 — Metal host scaffold does not earn engine points yet.
  return "0.00";
}

const char* welvet_runtime_identity(void) {
  static thread_local std::string id;
  id = welvet::metal::runtime_identity();
  return id.c_str();
}

welvet_status welvet_metal_init(void) {
  return to_c(welvet::metal::init_default_device());
}

int32_t welvet_metal_available(void) {
  return welvet::metal::available() ? 1 : 0;
}

int32_t welvet_gaps_count(void) {
  return static_cast<int32_t>(welvet::gaps::count_gaps());
}

int32_t welvet_gaps_done(void) {
  return static_cast<int32_t>(welvet::gaps::count_done());
}

welvet_status welvet_dense_forward_f32(int32_t batch, int32_t in, int32_t out, int32_t act,
                                       int32_t dtype, int32_t /*backend*/, const float* W_init,
                                       const double* /*bias*/, size_t /*bias_len*/, const float* x,
                                       float* pre, float* post) {
  (void)batch;
  (void)x;
  (void)pre;
  (void)post;
  welvet::dense::Layer layer;
  auto st = welvet::dense::new_layer(in, out, static_cast<welvet::Activation>(act),
                                     static_cast<welvet::DType>(dtype), W_init, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 xin, pre_t, post_t;
  return to_c(welvet::dense::forward_f32(layer, xin, &pre_t, &post_t));
}

} // extern "C"
