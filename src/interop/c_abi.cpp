#include "welvet_c.h"

#include "welvet/dense_session.hpp"
#include "welvet/gaps.hpp"
#include "welvet/metal/device.hpp"
#include "welvet/metal/matvec.hpp"
#include "welvet/mha.hpp"
#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/gdn.hpp"
#include "welvet/kmeans.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mamba.hpp"
#include "welvet/metacognition.hpp"
#include "welvet/parallel.hpp"
#include "welvet/rnn.hpp"
#include "welvet/embedding.hpp"
#include "welvet/residual.hpp"
#include "welvet/seqmix.hpp"
#include "welvet/sequential.hpp"
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

welvet_status welvet_embedding_forward_f32(int32_t batch, int32_t vocab, int32_t emb,
                                           int32_t seq_len, int32_t in_rank, int32_t in_last,
                                           int32_t act, int32_t dtype, int32_t /*backend*/,
                                           const float* W, const float* x, float* pre, float* post) {
  if (batch <= 0 || vocab <= 0 || emb <= 0 || seq_len <= 0 || in_last <= 0 || !W || !x || !pre ||
      !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "embedding: invalid argument");
    return st;
  }
  if (in_rank < 1 || in_rank > 3) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "embedding: bad in_rank");
    return st;
  }
  welvet::embedding::Config cfg;
  cfg.vocab_size = vocab;
  cfg.embedding_dim = emb;
  cfg.seq_len = seq_len;
  welvet::embedding::Layer layer;
  auto st = welvet::embedding::new_configured(cfg, static_cast<welvet::DType>(dtype),
                                              static_cast<welvet::Activation>(act), W, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  if (in_rank == 1) {
    input.shape = {seq_len};
  } else if (in_rank == 2) {
    input.shape = {batch, seq_len};
  } else {
    input.shape = {batch, seq_len, in_last};
  }
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq_len * in_last));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::embedding::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_sequential_forward_f32(int32_t batch, int32_t seq, int32_t dim, int32_t depth,
                                            int32_t child_act, int32_t dtype, int32_t /*backend*/,
                                            const float* packed, const float* x, float* pre,
                                            float* post) {
  if (batch <= 0 || seq <= 0 || dim <= 0 || depth <= 0 || !packed || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "sequential: invalid argument");
    return st;
  }
  welvet::sequential::Config cfg;
  cfg.dim = dim;
  cfg.seq_len = seq;
  cfg.depth = depth;
  welvet::sequential::Layer layer;
  auto st = welvet::sequential::new_configured(cfg, static_cast<welvet::DType>(dtype),
                                               static_cast<welvet::Activation>(child_act), packed,
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
  st = welvet::sequential::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_residual_forward_f32(int32_t batch, int32_t seq, int32_t dim, int32_t depth,
                                          int32_t child_act, int32_t dtype, int32_t /*backend*/,
                                          const float* packed, const float* x, float* pre,
                                          float* post) {
  if (batch <= 0 || seq <= 0 || dim <= 0 || depth <= 0 || !packed || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "residual: invalid argument");
    return st;
  }
  welvet::residual::Config cfg;
  cfg.dim = dim;
  cfg.seq_len = seq;
  cfg.depth = depth;
  welvet::residual::Layer layer;
  auto st = welvet::residual::new_configured(cfg, static_cast<welvet::DType>(dtype),
                                             static_cast<welvet::Activation>(child_act), packed,
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
  st = welvet::residual::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_seqmix_kind_name(int32_t kind, char* buf, int32_t buf_len) {
  if (!buf || buf_len <= 0) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "seqmix: bad buf");
    return st;
  }
  const char* name = welvet::seqmix::kind_name(static_cast<welvet::seqmix::Kind>(kind));
  if (!name) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "seqmix: unknown kind %d", static_cast<int>(kind));
    return st;
  }
  std::snprintf(buf, static_cast<std::size_t>(buf_len), "%s", name);
  return to_c(welvet::Status::Ok());
}

int32_t welvet_seqmix_mha_kind(void) {
  return static_cast<int32_t>(welvet::seqmix::mha_mixer_kind());
}
int32_t welvet_seqmix_mamba_kind(void) {
  return static_cast<int32_t>(welvet::seqmix::mamba_kind());
}
int32_t welvet_seqmix_gdn_kind(void) {
  return static_cast<int32_t>(welvet::seqmix::gdn_kind());
}

welvet_status welvet_cnn1_forward_f32(int32_t batch, int32_t in_c, int32_t filters, int32_t seq_len,
                                      int32_t kernel, int32_t stride, int32_t padding, int32_t act,
                                      int32_t dtype, int32_t /*backend*/, const float* W,
                                      const double* bias, size_t bias_len, const float* x,
                                      float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || seq_len <= 0 || kernel <= 0 || !W || !x || !pre ||
      !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "cnn1: invalid argument");
    return st;
  }
  welvet::cnn1::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.seq_len = seq_len;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::cnn1::Layer layer;
  auto st = welvet::cnn1::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  if (bias && bias_len > 0) {
    layer.proj.weights.bias.assign(bias, bias + bias_len);
  }
  welvet::TensorF32 input;
  input.shape = {batch, in_c, seq_len};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * seq_len));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::cnn1::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_cnn2_forward_f32(int32_t batch, int32_t in_c, int32_t filters, int32_t height,
                                      int32_t width, int32_t kernel, int32_t stride, int32_t padding,
                                      int32_t act, int32_t dtype, int32_t /*backend*/,
                                      const float* W, const double* bias, size_t bias_len,
                                      const float* x, float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || height <= 0 || width <= 0 || kernel <= 0 || !W ||
      !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "cnn2: invalid argument");
    return st;
  }
  welvet::cnn2::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.height = height;
  cfg.width = width;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::cnn2::Layer layer;
  auto st = welvet::cnn2::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  if (bias && bias_len > 0) {
    layer.proj.weights.bias.assign(bias, bias + bias_len);
  }
  welvet::TensorF32 input;
  input.shape = {batch, in_c, height, width};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * height * width));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::cnn2::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_cnn3_forward_f32(int32_t batch, int32_t in_c, int32_t filters, int32_t depth,
                                      int32_t height, int32_t width, int32_t kernel, int32_t stride,
                                      int32_t padding, int32_t act, int32_t dtype,
                                      int32_t /*backend*/, const float* W, const double* bias,
                                      size_t bias_len, const float* x, float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || depth <= 0 || height <= 0 || width <= 0 ||
      kernel <= 0 || !W || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "cnn3: invalid argument");
    return st;
  }
  welvet::cnn3::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.depth = depth;
  cfg.height = height;
  cfg.width = width;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::cnn3::Layer layer;
  auto st = welvet::cnn3::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  if (bias && bias_len > 0) {
    layer.proj.weights.bias.assign(bias, bias + bias_len);
  }
  welvet::TensorF32 input;
  input.shape = {batch, in_c, depth, height, width};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * depth * height * width));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::cnn3::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_convt1_forward_f32(int32_t batch, int32_t in_c, int32_t filters,
                                        int32_t seq_len, int32_t kernel, int32_t stride,
                                        int32_t padding, int32_t output_padding, int32_t act,
                                        int32_t dtype, int32_t /*backend*/, const float* W,
                                        const float* x, float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || seq_len <= 0 || kernel <= 0 || !W || !x || !pre ||
      !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "convt1: invalid argument");
    return st;
  }
  welvet::convt1::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.seq_len = seq_len;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.output_padding = output_padding;
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::convt1::Layer layer;
  auto st = welvet::convt1::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  input.shape = {batch, in_c, seq_len};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * seq_len));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::convt1::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_convt2_forward_f32(int32_t batch, int32_t in_c, int32_t filters,
                                        int32_t height, int32_t width, int32_t kernel,
                                        int32_t stride, int32_t padding, int32_t output_padding,
                                        int32_t act, int32_t dtype, int32_t /*backend*/,
                                        const float* W, const float* x, float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || height <= 0 || width <= 0 || kernel <= 0 || !W ||
      !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "convt2: invalid argument");
    return st;
  }
  welvet::convt2::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.height = height;
  cfg.width = width;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.output_padding = output_padding;
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::convt2::Layer layer;
  auto st = welvet::convt2::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  input.shape = {batch, in_c, height, width};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * height * width));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::convt2::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_convt3_forward_f32(int32_t batch, int32_t in_c, int32_t filters, int32_t depth,
                                        int32_t height, int32_t width, int32_t kernel,
                                        int32_t stride, int32_t padding, int32_t output_padding,
                                        int32_t act, int32_t dtype, int32_t /*backend*/,
                                        const float* W, const float* x, float* pre, float* post) {
  if (batch <= 0 || in_c <= 0 || filters <= 0 || depth <= 0 || height <= 0 || width <= 0 ||
      kernel <= 0 || !W || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "convt3: invalid argument");
    return st;
  }
  welvet::convt3::Config cfg;
  cfg.in_channels = in_c;
  cfg.filters = filters;
  cfg.depth = depth;
  cfg.height = height;
  cfg.width = width;
  cfg.kernel = kernel;
  cfg.stride = stride;
  cfg.padding = padding;
  cfg.output_padding = output_padding;
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::convt3::Layer layer;
  auto st = welvet::convt3::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  input.shape = {batch, in_c, depth, height, width};
  input.data.assign(x, x + static_cast<std::size_t>(batch * in_c * depth * height * width));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::convt3::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_rnn_forward_f32(int32_t batch, int32_t input_size, int32_t hidden_size,
                                     int32_t seq_len, int32_t dtype, int32_t /*backend*/,
                                     const float* packed, const float* x, float* pre, float* post) {
  if (batch <= 0 || input_size <= 0 || hidden_size <= 0 || seq_len <= 0 || !packed || !x || !pre ||
      !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "rnn: invalid argument");
    return st;
  }
  welvet::rnn::Config cfg;
  cfg.input_size = input_size;
  cfg.hidden_size = hidden_size;
  cfg.seq_len = seq_len;
  welvet::rnn::Layer layer;
  auto st = welvet::rnn::new_configured(cfg, static_cast<welvet::DType>(dtype), packed, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  input.shape = {batch, seq_len, input_size};
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq_len * input_size));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::rnn::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_lstm_forward_f32(int32_t batch, int32_t input_size, int32_t hidden_size,
                                      int32_t seq_len, int32_t dtype, int32_t /*backend*/,
                                      const float* packed, const float* x, float* pre, float* post) {
  if (batch <= 0 || input_size <= 0 || hidden_size <= 0 || seq_len <= 0 || !packed || !x || !pre ||
      !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "lstm: invalid argument");
    return st;
  }
  welvet::lstm::Config cfg;
  cfg.input_size = input_size;
  cfg.hidden_size = hidden_size;
  cfg.seq_len = seq_len;
  welvet::lstm::Layer layer;
  auto st = welvet::lstm::new_configured(cfg, static_cast<welvet::DType>(dtype), packed, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  input.shape = {batch, seq_len, input_size};
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq_len * input_size));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::lstm::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_gdn_forward_f32(int32_t batch, int32_t seq, int32_t hidden,
                                     int32_t num_key_heads, int32_t num_value_heads,
                                     int32_t key_head_dim, int32_t value_head_dim,
                                     int32_t conv_kernel, int32_t /*dtype*/, int32_t /*backend*/,
                                     const float* packed, const float* x, float* pre, float* post) {
  if (batch <= 0 || seq <= 0 || hidden <= 0 || !packed || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "gdn: invalid argument");
    return st;
  }
  welvet::gdn::Config cfg;
  cfg.hidden_size = hidden;
  cfg.num_key_heads = num_key_heads;
  cfg.num_value_heads = num_value_heads;
  cfg.key_head_dim = key_head_dim;
  cfg.value_head_dim = value_head_dim;
  cfg.conv_kernel = conv_kernel;
  welvet::gdn::Layer layer;
  auto st = welvet::gdn::new_configured(cfg, packed, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  input.shape = {batch, seq, hidden};
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * hidden));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::gdn::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_mamba_forward_f32(int32_t batch, int32_t seq, int32_t d_model, int32_t d_state,
                                       int32_t expand, int32_t dtype, int32_t /*backend*/,
                                       const float* packed, const float* x, float* pre,
                                       float* post) {
  if (batch <= 0 || seq <= 0 || d_model <= 0 || d_state <= 0 || !packed || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "mamba: invalid argument");
    return st;
  }
  welvet::mamba::Config cfg;
  cfg.d_model = d_model;
  cfg.d_state = d_state;
  cfg.expand = expand;
  cfg.seq_len = seq;
  welvet::mamba::Layer layer;
  auto st = welvet::mamba::new_configured(cfg, static_cast<welvet::DType>(dtype), packed, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  input.shape = {batch, seq, d_model};
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * d_model));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::mamba::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_kmeans_forward_f32(int32_t n, int32_t feature_dim, int32_t num_clusters,
                                        int32_t output_mode, int32_t act, double temperature,
                                        int32_t dtype, int32_t /*backend*/, int32_t rank,
                                        const int32_t* shape, const float* W, const float* x,
                                        float* pre, float* post) {
  if (n <= 0 || feature_dim <= 0 || num_clusters <= 0 || !W || !x || !pre || !post || rank < 1 ||
      !shape) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "kmeans: invalid argument");
    return st;
  }
  welvet::kmeans::Config cfg;
  cfg.feature_dim = feature_dim;
  cfg.num_clusters = num_clusters;
  cfg.temperature = temperature;
  cfg.output_mode = static_cast<welvet::kmeans::OutputMode>(output_mode);
  cfg.activation = static_cast<welvet::Activation>(act);
  welvet::kmeans::Layer layer;
  auto st = welvet::kmeans::new_configured(cfg, static_cast<welvet::DType>(dtype), W, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  input.shape.assign(shape, shape + rank);
  input.data.assign(x, x + static_cast<std::size_t>(n));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::kmeans::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_parallel_forward_f32(int32_t batch, int32_t seq, int32_t dim, int32_t out_feat,
                                          int32_t branches, int32_t combine, int32_t sparse_k,
                                          int32_t dtype, int32_t /*backend*/, double disagree_beta,
                                          const float* packed, const float* x, float* pre,
                                          float* post) {
  if (batch <= 0 || seq <= 0 || dim <= 0 || out_feat <= 0 || branches <= 0 || !packed || !x ||
      !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "parallel: invalid argument");
    return st;
  }
  if (combine < 0 || combine > 6) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "parallel: bad combine");
    return st;
  }
  welvet::parallel::Config cfg;
  cfg.dim = dim;
  cfg.out_feat = out_feat;
  cfg.branches = branches;
  cfg.combine = static_cast<welvet::parallel::Combine>(combine);
  cfg.seq_len = seq == 1 ? 0 : seq;
  cfg.sparse_k = sparse_k;
  cfg.disagree_beta = disagree_beta;
  welvet::parallel::Layer layer;
  auto st =
      welvet::parallel::new_configured(cfg, static_cast<welvet::DType>(dtype), packed, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, dim};
  } else {
    input.shape = {batch, seq, dim};
  }
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * dim));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::parallel::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_metacognition_forward_f32(int32_t batch, int32_t seq, int32_t dim, int32_t dtype,
                                               int32_t /*backend*/, const float* packed,
                                               const float* x, float* pre, float* post) {
  if (batch <= 0 || seq <= 0 || dim <= 0 || !packed || !x || !pre || !post) {
    welvet_status st{};
    st.code = 1;
    std::snprintf(st.message, sizeof(st.message), "metacognition: invalid argument");
    return st;
  }
  welvet::metacognition::Config cfg;
  cfg.dim = dim;
  cfg.seq_len = seq == 1 ? 0 : seq;
  welvet::metacognition::Layer layer;
  auto st =
      welvet::metacognition::new_configured(cfg, static_cast<welvet::DType>(dtype), packed, &layer);
  if (!st) return to_c(st);
  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, dim};
  } else {
    input.shape = {batch, seq, dim};
  }
  input.data.assign(x, x + static_cast<std::size_t>(batch * seq * dim));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::metacognition::forward(layer, input, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

} // extern "C"
