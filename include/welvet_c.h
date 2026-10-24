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

/* MHA ModeSelf forward — WVM1 wire. Q/K/V/O Dense on Metal; attn ALU host f64 (Go/cpp baseline). */
WELVET_API welvet_status welvet_mha_forward_f32(
    int32_t batch, int32_t seq, int32_t d_model, int32_t num_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t max_seq_len, int32_t mask, int32_t pos, int32_t dtype, int32_t backend,
    const float* Q, const float* K, const float* V, const float* O, const float* x, float* pre,
    float* post);

/* SwiGLU forward — WVS1 wire. Gate/Up/Down Dense on Metal; SiLU⊙ host f64 (Go/cpp baseline). */
WELVET_API welvet_status welvet_swiglu_forward_f32(
    int32_t batch, int32_t seq, int32_t input_dim, int32_t intermediate_dim, int32_t dtype,
    int32_t backend, const float* Gate, const float* Up, const float* Down, const float* x,
    float* pre, float* post);

/* RMSNorm forward — WVR1 wire. γ FormatNone×34 + host f64 RMS (Go/cpp exact_bit). */
WELVET_API welvet_status welvet_rmsnorm_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma, const float* x, float* pre, float* post);

/* LayerNorm forward — WVL1 wire. γ/β FormatNone×34 + host f64 mean/var (Go/cpp exact_bit). */
WELVET_API welvet_status welvet_layernorm_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, double eps, int32_t dtype, int32_t backend,
    const float* gamma, const float* beta, const float* x, float* pre, float* post);

/* Softmax forward — WSM1 wire. All Kind variants; host stable Softmax (Go/cpp exact_bit). */
WELVET_API welvet_status welvet_softmax_forward_f32(
    int32_t n, int32_t dim, int32_t seq_len, int32_t kind, int32_t rows, int32_t cols,
    double temperature, double entmax_alpha, int32_t gumbel_seed, int32_t dtype, int32_t backend,
    int32_t rank, const int32_t* shape, const uint8_t* mask, int32_t mask_len, const int32_t* hier,
    int32_t hier_len, const float* x, float* pre, float* post);

/* Embedding gather — WEM1 wire. Vocab×Emb FormatNone×34 decode_row_f64 (Go/cpp exact_bit). */
WELVET_API welvet_status welvet_embedding_forward_f32(
    int32_t batch, int32_t vocab, int32_t emb, int32_t seq_len, int32_t in_rank, int32_t in_last,
    int32_t act, int32_t dtype, int32_t backend, const float* W, const float* x, float* pre,
    float* post);

/* Sequential Dense chain — WSQ1 wire. packed: Depth×[Dim×Dim], x: [batch*seq*dim]
 * (seq=1 → [batch*dim]). pre/post: same layout (last child). Chains Metal soft-double Dense. */
WELVET_API welvet_status welvet_sequential_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* Residual y=F(x)+x — WRE1 wire. packed: Depth×[Dim×Dim] Dense F, x: [batch*seq*dim]
 * (seq=1 → [batch*dim]). pre: last F child pre; post: skip-add. F via Metal soft-double Dense. */
WELVET_API welvet_status welvet_residual_forward_f32(
    int32_t batch, int32_t seq, int32_t dim, int32_t depth, int32_t child_act, int32_t dtype,
    int32_t backend, const float* packed, const float* x, float* pre, float* post);

/* seqmix Kind name — WSX1 smoke. kind: 0=attention 1=ssm 2=linear_attn 3=conv_mix.
 * Writes NUL-terminated name into buf (max buf_len). */
WELVET_API welvet_status welvet_seqmix_kind_name(int32_t kind, char* buf, int32_t buf_len);

/* Mixer Kind tags: mha→0, mamba→1, gdn→2 (matches seqmix.Kind iota). */
WELVET_API int32_t welvet_seqmix_mha_kind(void);
WELVET_API int32_t welvet_seqmix_mamba_kind(void);
WELVET_API int32_t welvet_seqmix_gdn_kind(void);

/* CNN1/2/3 — WVC1/2/3 wire. Host im2col → Metal soft-double Dense → loom. */
WELVET_API welvet_status welvet_cnn1_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len, int32_t kernel, int32_t stride,
    int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W, const double* bias,
    size_t bias_len, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_cnn2_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t height, int32_t width, int32_t kernel,
    int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend, const float* W,
    const double* bias, size_t bias_len, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_cnn3_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t depth, int32_t height, int32_t width,
    int32_t kernel, int32_t stride, int32_t padding, int32_t act, int32_t dtype, int32_t backend,
    const float* W, const double* bias, size_t bias_len, const float* x, float* pre, float* post);

/* ConvT1/2/3 — WVT1/2/3 wire. Host f64 scatter + Activate (Go/cpp exact_bit). No bias. */
WELVET_API welvet_status welvet_convt1_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len, int32_t kernel, int32_t stride,
    int32_t padding, int32_t output_padding, int32_t act, int32_t dtype, int32_t backend,
    const float* W, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_convt2_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t height, int32_t width, int32_t kernel,
    int32_t stride, int32_t padding, int32_t output_padding, int32_t act, int32_t dtype,
    int32_t backend, const float* W, const float* x, float* pre, float* post);

WELVET_API welvet_status welvet_convt3_forward_f32(
    int32_t batch, int32_t in_c, int32_t filters, int32_t depth, int32_t height, int32_t width,
    int32_t kernel, int32_t stride, int32_t padding, int32_t output_padding, int32_t act,
    int32_t dtype, int32_t backend, const float* W, const float* x, float* pre, float* post);

#ifdef __cplusplus
}
#endif
