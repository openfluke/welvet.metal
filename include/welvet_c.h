#pragma once

/* Stable C ABI for welvet.metal — used by w2a go_v_metal and future NAC. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(WELVET_BUILD_SHARED)
#  ifdef WELVET_EXPORTS
#    define WELVET_API __declspec(dllexport)
#  else
#    define WELVET_API __declspec(dllimport)
#  endif
#else
#  define WELVET_API __attribute__((visibility("default")))
#endif

typedef struct welvet_status {
  int32_t code; /* 0=ok, 1=invalid, 2=unsupported, 3=gap, 4=internal */
  char message[256];
} welvet_status;

/* Engine identity — version follows Go-mirrored board (0.{round(earned)}). */
WELVET_API const char* welvet_version(void);
WELVET_API const char* welvet_runtime_identity(void); /* Metal device + build */

WELVET_API welvet_status welvet_metal_init(void);
WELVET_API int32_t welvet_metal_available(void);
WELVET_API int32_t welvet_gaps_count(void);
WELVET_API int32_t welvet_gaps_done(void);

/* One-shot Dense Metal forward (mount+execute+release internally). */
WELVET_API welvet_status welvet_dense_forward_f32(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype, int32_t backend,
    const float* W_init, const double* bias, size_t bias_len, const float* x, float* pre,
    float* post);

/* Resident session: mount once, execute many (w2a mount vs exec timing). */
typedef struct welvet_dense_session welvet_dense_session;

WELVET_API welvet_status welvet_dense_mount(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype,
    const float* W_init, const double* bias, size_t bias_len, welvet_dense_session** sess);

WELVET_API welvet_status welvet_dense_execute(welvet_dense_session* s, const float* x, float* pre,
                                              float* post);

WELVET_API void welvet_dense_release(welvet_dense_session* s);

#ifdef __cplusplus
}
#endif
