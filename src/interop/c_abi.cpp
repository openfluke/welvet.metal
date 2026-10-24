#include "welvet_c.h"

#include "welvet/dense_session.hpp"
#include "welvet/gaps.hpp"
#include "welvet/metal/device.hpp"
#include "welvet/metal/matvec.hpp"
#include "welvet/mha.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/softmax.hpp"
#include "welvet/status.hpp"
#include "welvet/swiglu.hpp"
#include "welvet/tensor.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

welvet_status welvet_mha_forward_f32(int32_t batch, int32_t seq, int32_t d_model, int32_t num_heads,
                                     int32_t num_kv_heads, int32_t head_dim, int32_t max_seq_len,
                                     int32_t mask, int32_t pos, int32_t dtype, int32_t /*backend*/,
                                     const float* Q, const float* K, const float* V, const float* O,
                                     const float* x, float* pre, float* post) {
  if (batch <= 0 || seq <= 0 || d_model <= 0 || num_heads <= 0 || !Q || !K || !V || !O || !x ||
      !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "mha: invalid argument");
    return st;
  }

  welvet::mha::Config cfg;
  cfg.d_model = d_model;
  cfg.num_heads = num_heads;
  cfg.num_kv_heads = num_kv_heads;
  cfg.head_dim = head_dim;
  cfg.max_seq_len = max_seq_len;
  cfg.mask = static_cast<welvet::mha::MaskKind>(mask);
  cfg.pos = static_cast<welvet::mha::PosKind>(pos);
  cfg.mode = welvet::mha::AttnMode::Self;
  cfg.causal = (cfg.mask == welvet::mha::MaskKind::Causal);

  welvet::mha::Layer layer;
  auto st = welvet::mha::new_configured(cfg, static_cast<welvet::DType>(dtype), Q, K, V, O, &layer);
  if (!st) return to_c(st);

  welvet::TensorF32 input;
  input.shape = {batch, seq, d_model};
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * d_model));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::mha::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_swiglu_forward_f32(int32_t batch, int32_t seq, int32_t input_dim,
                                        int32_t intermediate_dim, int32_t dtype, int32_t /*backend*/,
                                        const float* Gate, const float* Up, const float* Down,
                                        const float* x, float* pre, float* post) {
  if (batch <= 0 || seq <= 0 || input_dim <= 0 || intermediate_dim <= 0 || !Gate || !Up || !Down ||
      !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "swiglu: invalid argument");
    return st;
  }

  welvet::swiglu::Config cfg;
  cfg.input_dim = input_dim;
  cfg.intermediate_dim = intermediate_dim;

  welvet::swiglu::Layer layer;
  auto st = welvet::swiglu::new_configured(cfg, static_cast<welvet::DType>(dtype), Gate, Up, Down,
                                           &layer);
  if (!st) return to_c(st);

  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, input_dim};
  } else {
    input.shape = {batch, seq, input_dim};
  }
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * input_dim));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::swiglu::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_rmsnorm_forward_f32(int32_t batch, int32_t seq, int32_t dim, double eps,
                                         int32_t dtype, int32_t /*backend*/, const float* gamma,
                                         const float* x, float* pre, float* post) {
  if (batch <= 0 || seq <= 0 || dim <= 0 || !gamma || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "rmsnorm: invalid argument");
    return st;
  }

  welvet::rmsnorm::Config cfg;
  cfg.dim = dim;
  cfg.eps = eps;

  welvet::rmsnorm::Layer layer;
  auto st = welvet::rmsnorm::new_configured(cfg, static_cast<welvet::DType>(dtype), gamma, &layer);
  if (!st) return to_c(st);

  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, dim};
  } else {
    input.shape = {batch, seq, dim};
  }
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * dim));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::rmsnorm::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_layernorm_forward_f32(int32_t batch, int32_t seq, int32_t dim, double eps,
                                           int32_t dtype, int32_t /*backend*/, const float* gamma,
                                           const float* beta, const float* x, float* pre,
                                           float* post) {
  if (batch <= 0 || seq <= 0 || dim <= 0 || !gamma || !beta || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "layernorm: invalid argument");
    return st;
  }

  welvet::layernorm::Config cfg;
  cfg.dim = dim;
  cfg.eps = eps;

  welvet::layernorm::Layer layer;
  auto st = welvet::layernorm::new_configured(cfg, static_cast<welvet::DType>(dtype), gamma, beta,
                                              &layer);
  if (!st) return to_c(st);

  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, dim};
  } else {
    input.shape = {batch, seq, dim};
  }
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * dim));

  welvet::TensorF32 pre_t, post_t;
  st = welvet::layernorm::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_softmax_forward_f32(int32_t n, int32_t dim, int32_t seq_len, int32_t kind,
                                         int32_t rows, int32_t cols, double temperature,
                                         double entmax_alpha, int32_t gumbel_seed, int32_t dtype,
                                         int32_t /*backend*/, int32_t rank, const int32_t* shape,
                                         const uint8_t* mask, int32_t mask_len, const int32_t* hier,
                                         int32_t hier_len, const float* x, float* pre, float* post) {
  if (n <= 0 || dim <= 0 || !x || !pre || !post || rank < 1 || !shape) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "softmax: invalid argument");
    return st;
  }
  welvet::softmax::Config cfg;
  cfg.dim = dim;
  cfg.seq_len = seq_len <= 0 ? 1 : seq_len;
  cfg.rows = rows;
  cfg.cols = cols;
  cfg.temperature = temperature;
  cfg.kind = static_cast<welvet::softmax::Kind>(kind);
  cfg.entmax_alpha = entmax_alpha;
  cfg.gumbel_seed = static_cast<uint32_t>(gumbel_seed);
  if (mask && mask_len > 0) {
    cfg.mask.resize(static_cast<std::size_t>(mask_len));
    for (int32_t i = 0; i < mask_len; ++i) {
      cfg.mask[static_cast<std::size_t>(i)] = mask[i] != 0;
    }
  }
  if (hier && hier_len > 0) {
    cfg.hierarchy_levels.assign(hier, hier + hier_len);
  }
  welvet::softmax::Layer layer;
  auto st = welvet::softmax::new_configured(cfg, static_cast<welvet::DType>(dtype), &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  input.shape.assign(shape, shape + rank);
  input.data.assign(x, x + static_cast<std::size_t>(n));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::softmax::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  if (static_cast<int>(pre_t.data.size()) != n || static_cast<int>(post_t.data.size()) != n) {
    welvet_status err{};
    err.code = 1;
    std::snprintf(err.message, sizeof(err.message), "softmax: unexpected out len");
    return err;
  }
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

} // extern "C"
