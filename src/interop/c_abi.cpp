#include "welvet_c.h"

#include "welvet/dense_session.hpp"
#include "welvet/gaps.hpp"
#include "welvet/metal/device.hpp"
#include "welvet/metal/matvec.hpp"
#include "welvet/status.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

welvet_status to_c(const welvet::Status& st) {
  welvet_status out{};
  out.code = static_cast<int32_t>(st.code);
  std::snprintf(out.message, sizeof(out.message), "%s", st.message.c_str());
  return out;
}

} // namespace

extern "C" {

const char* welvet_version(void) {
  // VERSIONING.md: 0.01 — Dense FormatNone×34 Metal forward exact_bit vs Go.
  return "0.01";
}

const char* welvet_runtime_identity(void) {
  static thread_local std::string id;
  id = welvet::metal::runtime_identity();
  return id.c_str();
}

welvet_status welvet_metal_init(void) {
  auto st = welvet::metal::init_default_device();
  if (!st) return to_c(st);
  return to_c(welvet::metal::require_dense_compute());
}

int32_t welvet_metal_available(void) { return welvet::metal::available() ? 1 : 0; }

int32_t welvet_gaps_count(void) { return static_cast<int32_t>(welvet::gaps::count_gaps()); }

int32_t welvet_gaps_done(void) { return static_cast<int32_t>(welvet::gaps::count_done()); }

welvet_status welvet_dense_mount(int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype,
                                 const float* W_init, const double* bias, size_t bias_len,
                                 welvet_dense_session** sess) {
  if (!sess || !W_init) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "invalid argument");
    return st;
  }
  welvet::dense::Session* s = nullptr;
  auto st = welvet::dense::session_mount(batch, in, out, static_cast<welvet::Activation>(act),
                                         static_cast<welvet::DType>(dtype), W_init, bias, bias_len,
                                         &s);
  if (!st) return to_c(st);
  *sess = reinterpret_cast<welvet_dense_session*>(s);
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_dense_execute(welvet_dense_session* s, const float* x, float* pre,
                                   float* post) {
  return to_c(welvet::dense::session_execute(reinterpret_cast<welvet::dense::Session*>(s), x, pre,
                                             post));
}

void welvet_dense_release(welvet_dense_session* s) {
  welvet::dense::session_release(reinterpret_cast<welvet::dense::Session*>(s));
}

welvet_status welvet_dense_forward_f32(int32_t batch, int32_t in, int32_t out, int32_t act,
                                       int32_t dtype, int32_t /*backend*/, const float* W_init,
                                       const double* bias, size_t bias_len, const float* x,
                                       float* pre, float* post) {
  welvet_dense_session* s = nullptr;
  auto st = welvet_dense_mount(batch, in, out, act, dtype, W_init, bias, bias_len, &s);
  if (st.code != 0) return st;
  st = welvet_dense_execute(s, x, pre, post);
  welvet_dense_release(s);
  return st;
}

} // extern "C"
