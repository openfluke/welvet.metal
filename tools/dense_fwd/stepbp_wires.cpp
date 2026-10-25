// STB1/MBP1 StepBP + MeshBP wires (from welvet.cpp dense_fwd).

#include "volume_fwd.hpp"

#include "welvet/architecture.hpp"
#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/layer.hpp"
#include "welvet/runtime/line.hpp"
#include "welvet/runtime/step.hpp"
#include "welvet/stack.hpp"
#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/embedding.hpp"
#include "welvet/gdn.hpp"
#include "welvet/kmeans.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mamba.hpp"
#include "welvet/metacognition.hpp"
#include "welvet/mha.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/softmax.hpp"
#include "welvet/swiglu.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool read_exact(void* p, size_t n) { return std::fread(p, 1, n, stdin) == n; }
template <typename T>
static bool read_exact(T* p, size_t n) {
  return read_exact(static_cast<void*>(p), n * sizeof(T));
}
static bool write_exact(const void* p, size_t n) { return std::fwrite(p, 1, n, stdout) == n; }
template <typename T>
static bool write_exact(const T* p, size_t n) {
  return write_exact(static_cast<const void*>(p), n * sizeof(T));
}
static bool write_magic(const char* m) { return write_exact(m, 4); }

static int prod_shape(const std::vector<int32_t>& sh) {
  int p = 1;
  for (int32_t d : sh) p *= static_cast<int>(d);
  return p;
}


// Retain the child wire so every benchmark iteration starts from identical
// weights and configuration, including recursively nested Stack children.
class ChildReader {
public:
  std::vector<unsigned char> bytes;
  explicit ChildReader(const std::vector<unsigned char>* replay = nullptr) : replay_(replay) {}
  template <typename T> bool read(T* p, size_t n) {
    const size_t size = n * sizeof(T);
    if (replay_) {
      if (offset_ > replay_->size() || size > replay_->size() - offset_) return false;
      if (size) std::memcpy(p, replay_->data() + offset_, size);
      offset_ += size;
      return true;
    }
    if (!read_exact(p, n)) return false;
    if (size) {
      const auto* begin = reinterpret_cast<const unsigned char*>(p);
      bytes.insert(bytes.end(), begin, begin + size);
    }
    return true;
  }
private:
  const std::vector<unsigned char>* replay_;
  size_t offset_ = 0;
};

static bool read_child(ChildReader& reader, welvet::stack::Child* ch) {
  int32_t kind = 0;
  if (!reader.read(&kind, 1)) return false;
  ch->kind = static_cast<welvet::stack::Kind>(kind);
  switch (ch->kind) {
  case welvet::stack::Kind::Dense: {
    int32_t in = 0, out = 0, act = 0, dtype = 1;
    if (!reader.read(&in, 1) || !reader.read(&out, 1) || !reader.read(&act, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    std::vector<float> W(static_cast<std::size_t>(out * in));
    if (!reader.read(W.data(), W.size())) return false;
    uint32_t bias_len = 0;
    if (!reader.read(&bias_len, 1)) return false;
    std::vector<double> bias(bias_len);
    if (bias_len > 0 && !reader.read(bias.data(), bias.size())) return false;
    auto st = welvet::dense::new_layer(in, out, static_cast<welvet::Activation>(act),
                                       static_cast<welvet::DType>(dtype), W.data(), &ch->dense_op);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return false;
    }
    if (bias_len > 0) {
      ch->dense_op.weights.bias = std::move(bias);
    }
    return true;
  }
  case welvet::stack::Kind::RMSNorm: {
    int32_t dim = 0, dtype = 1;
    double eps = 0.0;
    if (!reader.read(&dim, 1) || !reader.read(&dtype, 1) || !reader.read(&eps, 1)) return false;
    std::vector<float> g(static_cast<std::size_t>(dim));
    if (!reader.read(g.data(), g.size())) return false;
    welvet::rmsnorm::Config cfg;
    cfg.dim = dim;
    cfg.eps = eps;
    auto st = welvet::rmsnorm::new_configured(cfg, static_cast<welvet::DType>(dtype), g.data(),
                                              &ch->rmsnorm_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::LayerNorm: {
    int32_t dim = 0, dtype = 1;
    double eps = 0.0;
    if (!reader.read(&dim, 1) || !reader.read(&dtype, 1) || !reader.read(&eps, 1)) return false;
    std::vector<float> g(static_cast<std::size_t>(dim)), b(static_cast<std::size_t>(dim));
    if (!reader.read(g.data(), g.size()) || !reader.read(b.data(), b.size())) return false;
    welvet::layernorm::Config cfg;
    cfg.dim = dim;
    cfg.eps = eps;
    auto st = welvet::layernorm::new_configured(cfg, static_cast<welvet::DType>(dtype), g.data(),
                                                b.data(), &ch->layernorm_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::SwiGLU: {
    int32_t in = 0, inter = 0, dtype = 1;
    if (!reader.read(&in, 1) || !reader.read(&inter, 1) || !reader.read(&dtype, 1)) return false;
    const int ng = inter * in;
    const int nu = inter * in;
    const int nd = in * inter;
    std::vector<float> packed(static_cast<std::size_t>(ng + nu + nd));
    if (!reader.read(packed.data(), packed.size())) return false;
    welvet::swiglu::Config cfg;
    cfg.input_dim = in;
    cfg.intermediate_dim = inter;
    auto st = welvet::swiglu::new_configured(cfg, static_cast<welvet::DType>(dtype), packed.data(),
                                             packed.data() + ng, packed.data() + ng + nu,
                                             &ch->swiglu_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::Softmax: {
    int32_t dim = 0, dtype = 1, sk = 0;
    double temperature = 1.0;
    if (!reader.read(&dim, 1) || !reader.read(&dtype, 1) || !reader.read(&sk, 1) ||
        !reader.read(&temperature, 1)) {
      return false;
    }
    welvet::softmax::Config cfg;
    cfg.dim = dim;
    cfg.kind = static_cast<welvet::softmax::Kind>(sk);
    cfg.temperature = temperature;
    auto st = welvet::softmax::new_configured(cfg, static_cast<welvet::DType>(dtype), &ch->softmax_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::Embedding: {
    int32_t vocab = 0, emb = 0, seq = 0, act = -1, dtype = 1;
    if (!reader.read(&vocab, 1) || !reader.read(&emb, 1) || !reader.read(&seq, 1) ||
        !reader.read(&act, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    std::vector<float> W(static_cast<std::size_t>(vocab * emb));
    if (!reader.read(W.data(), W.size())) return false;
    welvet::embedding::Config cfg;
    cfg.vocab_size = vocab;
    cfg.embedding_dim = emb;
    cfg.seq_len = seq;
    auto st = welvet::embedding::new_configured(cfg, static_cast<welvet::DType>(dtype),
                                                static_cast<welvet::Activation>(act), W.data(),
                                                &ch->embedding_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::MHA: {
    int32_t seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
    int32_t mask = 0, pos = 0, dtype = 1;
    if (!reader.read(&seq, 1) || !reader.read(&d_model, 1) || !reader.read(&num_heads, 1) ||
        !reader.read(&num_kv, 1) || !reader.read(&head_dim, 1) || !reader.read(&msl, 1) ||
        !reader.read(&mask, 1) || !reader.read(&pos, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    const int hd = head_dim > 0 ? head_dim : d_model / num_heads;
    const int kv_heads = num_kv > 0 ? num_kv : num_heads;
    const int q_dim = num_heads * hd;
    const int kv_dim = kv_heads * hd;
    std::vector<float> Q(static_cast<std::size_t>(q_dim * d_model));
    std::vector<float> K(static_cast<std::size_t>(kv_dim * d_model));
    std::vector<float> V(static_cast<std::size_t>(kv_dim * d_model));
    std::vector<float> O(static_cast<std::size_t>(d_model * q_dim));
    if (!reader.read(Q.data(), Q.size()) || !reader.read(K.data(), K.size()) ||
        !reader.read(V.data(), V.size()) || !reader.read(O.data(), O.size())) {
      return false;
    }
    welvet::mha::Config cfg = welvet::mha::decoder_causal(d_model, num_heads, kv_heads);
    cfg.max_seq_len = msl > 0 ? msl : seq;
    cfg.mask = static_cast<welvet::mha::MaskKind>(mask);
    cfg.pos = static_cast<welvet::mha::PosKind>(pos);
    if (head_dim > 0) cfg.head_dim = head_dim;
    auto st = welvet::mha::new_configured(cfg, static_cast<welvet::DType>(dtype), Q.data(), K.data(),
                                          V.data(), O.data(), &ch->mha_op);
    if (!st) return false;
    (void)seq;
    return true;
  }
  case welvet::stack::Kind::CNN1: {
    int32_t in_ch = 0, filters = 0, seq = 0, kernel = 0, stride = 0, pad = 0, act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) || !reader.read(&seq, 1) ||
        !reader.read(&kernel, 1) || !reader.read(&stride, 1) || !reader.read(&pad, 1) ||
        !reader.read(&act, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::cnn1::Config cfg;
    cfg.in_channels = in_ch;
    cfg.filters = filters;
    cfg.seq_len = seq;
    cfg.kernel = kernel;
    cfg.stride = stride;
    cfg.padding = pad;
    cfg.activation = static_cast<welvet::Activation>(act);
    const int nw = filters * in_ch * kernel;
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st = welvet::cnn1::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(),
                                           &ch->cnn1_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::Sequential:
  case welvet::stack::Kind::Residual: {
    int32_t dim = 0, seq = 0, depth = 0, child_act = -1, dtype = 1;
    if (!reader.read(&dim, 1) || !reader.read(&seq, 1) || !reader.read(&depth, 1) ||
        !reader.read(&child_act, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    std::vector<float> W(static_cast<std::size_t>(depth * dim * dim));
    if (!reader.read(W.data(), W.size())) return false;
    if (ch->kind == welvet::stack::Kind::Sequential) {
      welvet::sequential::Config cfg;
      cfg.dim = dim;
      cfg.seq_len = seq;
      cfg.depth = depth;
      auto st = welvet::sequential::new_configured(cfg, static_cast<welvet::DType>(dtype),
                                                   static_cast<welvet::Activation>(child_act),
                                                   W.data(), &ch->sequential_op);
      if (!st) return false;
    } else {
      welvet::residual::Config cfg;
      cfg.dim = dim;
      cfg.seq_len = seq;
      cfg.depth = depth;
      auto st = welvet::residual::new_configured(cfg, static_cast<welvet::DType>(dtype),
                                                 static_cast<welvet::Activation>(child_act),
                                                 W.data(), &ch->residual_op);
      if (!st) return false;
    }
    return true;
  }
  case welvet::stack::Kind::Mamba: {
    int32_t d_model = 0, d_state = 0, expand = 0, seq = 0, dtype = 1;
    if (!reader.read(&d_model, 1) || !reader.read(&d_state, 1) || !reader.read(&expand, 1) ||
        !reader.read(&seq, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::mamba::Config cfg;
    cfg.d_model = d_model;
    cfg.d_state = d_state;
    cfg.expand = expand;
    cfg.seq_len = seq;
    const int nw = welvet::mamba::weight_count(cfg);
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st =
        welvet::mamba::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->mamba_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::Metacognition: {
    int32_t dim = 0, seq = 0, dtype = 1;
    if (!reader.read(&dim, 1) || !reader.read(&seq, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::metacognition::Config cfg;
    cfg.dim = dim;
    cfg.seq_len = seq == 1 ? 0 : seq;
    const int nw = welvet::metacognition::weight_count(cfg);
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st = welvet::metacognition::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(),
                                                    &ch->metacognition_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::GDN: {
    int32_t hidden = 0, seq = 0, nk = 0, nv = 0, hd_k = 0, hd_v = 0, conv_k = 0, dtype = 1;
    if (!reader.read(&hidden, 1) || !reader.read(&seq, 1) || !reader.read(&nk, 1) ||
        !reader.read(&nv, 1) || !reader.read(&hd_k, 1) || !reader.read(&hd_v, 1) ||
        !reader.read(&conv_k, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    (void)seq;
    (void)dtype; // GDN f32-only
    welvet::gdn::Config cfg;
    cfg.hidden_size = hidden;
    cfg.num_key_heads = nk;
    cfg.num_value_heads = nv;
    cfg.key_head_dim = hd_k;
    cfg.value_head_dim = hd_v;
    cfg.conv_kernel = conv_k;
    const int nw = welvet::gdn::weight_count(cfg);
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st = welvet::gdn::new_configured(cfg, W.data(), &ch->gdn_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::KMeans: {
    int32_t k = 0, d = 0, seq = 0, output_mode = 0, act = -1, dtype = 1;
    double temperature = 1.0;
    if (!reader.read(&k, 1) || !reader.read(&d, 1) || !reader.read(&seq, 1) ||
        !reader.read(&output_mode, 1) || !reader.read(&act, 1) || !reader.read(&temperature, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    (void)seq;
    welvet::kmeans::Config cfg;
    cfg.num_clusters = k;
    cfg.feature_dim = d;
    cfg.temperature = temperature;
    cfg.output_mode = static_cast<welvet::kmeans::OutputMode>(output_mode);
    cfg.activation = static_cast<welvet::Activation>(act);
    const int nw = k * d;
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st =
        welvet::kmeans::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->kmeans_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::CNN2: {
    int32_t in_ch = 0, filters = 0, h = 0, w = 0, kernel = 0, stride = 0, pad = 0, act = 0,
            dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) || !reader.read(&h, 1) ||
        !reader.read(&w, 1) || !reader.read(&kernel, 1) || !reader.read(&stride, 1) ||
        !reader.read(&pad, 1) || !reader.read(&act, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::cnn2::Config cfg;
    cfg.in_channels = in_ch;
    cfg.filters = filters;
    cfg.height = h;
    cfg.width = w;
    cfg.kernel = kernel;
    cfg.stride = stride;
    cfg.padding = pad;
    cfg.activation = static_cast<welvet::Activation>(act);
    const int nw = filters * in_ch * kernel * kernel;
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st = welvet::cnn2::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(),
                                           &ch->cnn2_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::CNN3: {
    int32_t in_ch = 0, filters = 0, d = 0, h = 0, w = 0, kernel = 0, stride = 0, pad = 0, act = 0,
            dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) || !reader.read(&d, 1) ||
        !reader.read(&h, 1) || !reader.read(&w, 1) || !reader.read(&kernel, 1) ||
        !reader.read(&stride, 1) || !reader.read(&pad, 1) || !reader.read(&act, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::cnn3::Config cfg;
    cfg.in_channels = in_ch;
    cfg.filters = filters;
    cfg.depth = d;
    cfg.height = h;
    cfg.width = w;
    cfg.kernel = kernel;
    cfg.stride = stride;
    cfg.padding = pad;
    cfg.activation = static_cast<welvet::Activation>(act);
    const int nw = filters * in_ch * kernel * kernel * kernel;
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st = welvet::cnn3::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(),
                                           &ch->cnn3_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::NestedStack: {
    int32_t n_nested = 0;
    if (!reader.read(&n_nested, 1) || n_nested <= 0) return false;
    ch->nested.clear();
    ch->nested.reserve(static_cast<std::size_t>(n_nested));
    for (int32_t i = 0; i < n_nested; ++i) {
      welvet::stack::Child nested;
      if (!read_child(reader, &nested)) return false;
      ch->nested.push_back(std::move(nested));
    }
    return true;
  }
  case welvet::stack::Kind::Parallel: {
    // combine, dim, out_feat, seq, sparse_k, disagree_beta f64, n_branches, branches…, has_gate, [gate Dense]
    int32_t combine = 1, dim = 0, out_feat = 0, seq = 0, sparse_k = 0, n_br = 0, has_gate = 0;
    double disagree_beta = 0.0;
    if (!reader.read(&combine, 1) || !reader.read(&dim, 1) || !reader.read(&out_feat, 1) ||
        !reader.read(&seq, 1) || !reader.read(&sparse_k, 1) || !reader.read(&disagree_beta, 1) ||
        !reader.read(&n_br, 1) || n_br <= 0) {
      return false;
    }
    ch->parallel_nest.combine = combine;
    ch->parallel_nest.dim = dim;
    ch->parallel_nest.out_feat = out_feat;
    ch->parallel_nest.seq_len = seq;
    ch->parallel_nest.sparse_k = sparse_k;
    ch->parallel_nest.disagree_beta = disagree_beta;
    ch->nested.clear();
    ch->nested.reserve(static_cast<std::size_t>(n_br));
    for (int32_t i = 0; i < n_br; ++i) {
      welvet::stack::Child nested;
      if (!read_child(reader, &nested)) return false;
      ch->nested.push_back(std::move(nested));
    }
    if (!reader.read(&has_gate, 1)) return false;
    ch->parallel_nest.has_gate = has_gate;
    if (has_gate) {
      welvet::stack::Child gate_ch;
      if (!read_child(reader, &gate_ch) || gate_ch.kind != welvet::stack::Kind::Dense) return false;
      ch->dense_op = std::move(gate_ch.dense_op);
    }
    return true;
  }
  case welvet::stack::Kind::ConvT1: {
    int32_t in_ch = 0, filters = 0, seq = 0, kernel = 0, stride = 0, pad = 0, out_pad = 0,
            act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) || !reader.read(&seq, 1) ||
        !reader.read(&kernel, 1) || !reader.read(&stride, 1) || !reader.read(&pad, 1) ||
        !reader.read(&out_pad, 1) || !reader.read(&act, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::convt1::Config cfg;
    cfg.in_channels = in_ch;
    cfg.filters = filters;
    cfg.seq_len = seq;
    cfg.kernel = kernel;
    cfg.stride = stride;
    cfg.padding = pad;
    cfg.output_padding = out_pad;
    cfg.activation = static_cast<welvet::Activation>(act);
    const int nw = filters * in_ch * kernel;
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st = welvet::convt1::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(),
                                             &ch->convt1_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::ConvT2: {
    int32_t in_ch = 0, filters = 0, h = 0, w = 0, kernel = 0, stride = 0, pad = 0, out_pad = 0,
            act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) || !reader.read(&h, 1) ||
        !reader.read(&w, 1) || !reader.read(&kernel, 1) || !reader.read(&stride, 1) ||
        !reader.read(&pad, 1) || !reader.read(&out_pad, 1) || !reader.read(&act, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::convt2::Config cfg;
    cfg.in_channels = in_ch;
    cfg.filters = filters;
    cfg.height = h;
    cfg.width = w;
    cfg.kernel = kernel;
    cfg.stride = stride;
    cfg.padding = pad;
    cfg.output_padding = out_pad;
    cfg.activation = static_cast<welvet::Activation>(act);
    const int nw = filters * in_ch * kernel * kernel;
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st = welvet::convt2::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(),
                                             &ch->convt2_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::ConvT3: {
    int32_t in_ch = 0, filters = 0, d = 0, h = 0, w = 0, kernel = 0, stride = 0, pad = 0,
            out_pad = 0, act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) || !reader.read(&d, 1) ||
        !reader.read(&h, 1) || !reader.read(&w, 1) || !reader.read(&kernel, 1) ||
        !reader.read(&stride, 1) || !reader.read(&pad, 1) || !reader.read(&out_pad, 1) ||
        !reader.read(&act, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::convt3::Config cfg;
    cfg.in_channels = in_ch;
    cfg.filters = filters;
    cfg.depth = d;
    cfg.height = h;
    cfg.width = w;
    cfg.kernel = kernel;
    cfg.stride = stride;
    cfg.padding = pad;
    cfg.output_padding = out_pad;
    cfg.activation = static_cast<welvet::Activation>(act);
    const int nw = filters * in_ch * kernel * kernel * kernel;
    std::vector<float> W(static_cast<std::size_t>(nw));
    if (!reader.read(W.data(), W.size())) return false;
    auto st = welvet::convt3::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(),
                                             &ch->convt3_op);
    if (!st) return false;
    return true;
  }
  case welvet::stack::Kind::RNN:
  case welvet::stack::Kind::LSTM: {
    int32_t in = 0, hid = 0, seq = 0, dtype = 1;
    if (!reader.read(&in, 1) || !reader.read(&hid, 1) || !reader.read(&seq, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    if (ch->kind == welvet::stack::Kind::RNN) {
      welvet::rnn::Config cfg;
      cfg.input_size = in;
      cfg.hidden_size = hid;
      cfg.seq_len = seq;
      const int nw = welvet::rnn::weight_count(cfg);
      std::vector<float> W(static_cast<std::size_t>(nw));
      if (!reader.read(W.data(), W.size())) return false;
      auto st =
          welvet::rnn::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->rnn_op);
      if (!st) return false;
    } else {
      welvet::lstm::Config cfg;
      cfg.input_size = in;
      cfg.hidden_size = hid;
      cfg.seq_len = seq;
      const int nw = welvet::lstm::weight_count(cfg);
      std::vector<float> W(static_cast<std::size_t>(nw));
      if (!reader.read(W.data(), W.size())) return false;
      auto st = welvet::lstm::new_configured(cfg, static_cast<welvet::DType>(dtype), W.data(),
                                             &ch->lstm_op);
      if (!st) return false;
    }
    return true;
  }
  default:
    std::fprintf(stderr, "welvet_dense_fwd: STB1 unsupported kind %d\n", kind);
    return false;
  }
}

// Dense mesh-clock train (MBP1/MTW1/MTC1 share stdin layout; out_magic differs).
int run_mesh_dense_train(int bench_iters, welvet::runtime::step::Credit credit,
                         const char* short_hdr, const char* out_magic) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0, ticks = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1) || !read_exact(&ticks, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short %s header\n", short_hdr);
    return 2;
  }
  if (batch <= 0 || ticks <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch/ticks\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in = 0, out = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t in = 0, out = 0, act = 0, dtype = 1;
    if (!read_exact(&in, 1) || !read_exact(&out, 1) || !read_exact(&act, 1) || !read_exact(&dtype, 1)) {
      return 2;
    }
    c.in = in;
    c.out = out;
    c.act = act;
    c.dtype = dtype;
    c.W.resize(static_cast<std::size_t>(out * in));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len > 0 && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in < 0) first_in = in;
    cells.push_back(std::move(c));
  }
  if (first_in < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_out = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_out = it->out;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_out};
  target.data.resize(static_cast<std::size_t>(batch * final_out));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) {
        auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
        cell->meta.is_disabled = true;
        continue;
      }
      welvet::dense::Layer layer;
      auto st = welvet::dense::new_layer(c.in, c.out, static_cast<welvet::Activation>(c.act),
                                         static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
      if (!st) return st;
      if (!c.bias.empty()) layer.weights.bias = c.bias;
      st = welvet::architecture::bind_dense(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                            std::move(layer));
      if (!st) return st;
    }
    return welvet::runtime::step::train_mse(g, input, target, ticks, lr, &loss, credit);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) {
      return 2;
    }
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_magic(out_magic) || !write_exact(&loss, 1)) return 2;
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t o = c.out, i = c.in;
    std::vector<float> Wa(static_cast<std::size_t>(o * i));
    auto est = cell->dense_op.weights.export_f32(Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&o, 1) || !write_exact(&i, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  return 0;
}

int run_meshbp(int bench_iters) {
  return run_mesh_dense_train(bench_iters, welvet::runtime::step::Credit::FamilyBP, "MBP1", "MBPO");
}

int run_meshtween(int bench_iters) {
  return run_mesh_dense_train(bench_iters, welvet::runtime::step::Credit::FamilyTween, "MTW1",
                              "MTWO");
}

int run_meshtweenchain(int bench_iters) {
  return run_mesh_dense_train(bench_iters, welvet::runtime::step::Credit::FamilyBP, "MTC1", "MTCO");
}

int run_meshtweensplit(int bench_iters) {
  return run_mesh_dense_train(bench_iters, welvet::runtime::step::Credit::FamilyTweenSplit, "MTS1",
                              "MTSO");
}

int run_meshtweenalt(int bench_iters) {
  return run_mesh_dense_train(bench_iters, welvet::runtime::step::Credit::FamilyTweenAlt, "MTA1",
                              "MTAO");
}

int run_meshtweensplitfastproxy(int bench_iters) {
  return run_mesh_dense_train(bench_iters, welvet::runtime::step::Credit::FamilyTweenSplitFastProxy,
                              "MTF1", "MTFO");
}

int run_meshtweensplitsparse(int bench_iters) {
  return run_mesh_dense_train(bench_iters, welvet::runtime::step::Credit::FamilyTweenSplitSparse,
                              "MSS1", "MSSO");
}

int run_stepbp(int bench_iters) {
  int32_t n_children = 0, n_ticks = 0, batch = 0;
  double lr = 0.0;
  if (!read_exact(&n_children, 1) || !read_exact(&n_ticks, 1) || !read_exact(&batch, 1) ||
      !read_exact(&lr, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short STB1 header\n");
    return 2;
  }
  if (n_children <= 0 || n_ticks <= 0 || batch <= 0) return 1;

  ChildReader initial;
  welvet::stack::Layer stack;
  stack.meta.type = welvet::LayerType::Stack;
  stack.children.resize(static_cast<std::size_t>(n_children));
  for (int32_t i = 0; i < n_children; ++i) {
    if (!read_child(initial, &stack.children[static_cast<std::size_t>(i)])) {
      std::fprintf(stderr, "welvet_dense_fwd: bad STB1 child %d\n", i);
      return 2;
    }
  }

  int32_t x_ndim = 0, t_ndim = 0;
  if (!read_exact(&x_ndim, 1) || x_ndim <= 0 || x_ndim > 5) return 2;
  std::vector<int32_t> x_shape(static_cast<std::size_t>(x_ndim));
  if (!read_exact(x_shape.data(), x_shape.size())) return 2;
  if (!read_exact(&t_ndim, 1) || t_ndim <= 0 || t_ndim > 5) return 2;
  std::vector<int32_t> t_shape(static_cast<std::size_t>(t_ndim));
  if (!read_exact(t_shape.data(), t_shape.size())) return 2;
  const int x_len = prod_shape(x_shape);
  const int t_len = prod_shape(t_shape);

  std::vector<std::vector<float>> xs(static_cast<std::size_t>(n_ticks));
  std::vector<std::vector<float>> tgts(static_cast<std::size_t>(n_ticks));
  for (int32_t t = 0; t < n_ticks; ++t) {
    xs[static_cast<std::size_t>(t)].resize(static_cast<std::size_t>(x_len));
    tgts[static_cast<std::size_t>(t)].resize(static_cast<std::size_t>(t_len));
    if (!read_exact(xs[static_cast<std::size_t>(t)].data(),
                    xs[static_cast<std::size_t>(t)].size()) ||
        !read_exact(tgts[static_cast<std::size_t>(t)].data(),
                    tgts[static_cast<std::size_t>(t)].size())) {
      return 2;
    }
  }
  (void)batch;

  double loss = 0.0;
  auto once = [&]() -> welvet::Status {
    stack = welvet::stack::Layer{};
    stack.meta.type = welvet::LayerType::Stack;
    stack.children.resize(static_cast<std::size_t>(n_children));
    ChildReader replay(&initial.bytes);
    for (auto& child : stack.children) {
      if (!read_child(replay, &child)) {
        return welvet::Status::Err(welvet::StatusCode::Internal, "bad STB1 replay child");
      }
    }
    welvet::runtime::line::State st;
    loss = 0.0;
    for (int32_t t = 0; t < n_ticks; ++t) {
      welvet::TensorF32 xin;
      xin.shape.assign(x_shape.begin(), x_shape.end());
      xin.data = xs[static_cast<std::size_t>(t)];
      welvet::TensorF32 tgt;
      tgt.shape.assign(t_shape.begin(), t_shape.end());
      tgt.data = tgts[static_cast<std::size_t>(t)];
      double tick_loss = 0.0;
      if (auto s = welvet::runtime::line::train_mse(stack, &st, xin, tgt, lr, &tick_loss); !s) {
        return s;
      }
      loss = tick_loss;
    }
    return welvet::Status::Ok();
  };

  if (bench_iters > 0) {
    // Warmup + timed; reset the fixture before every run.
    for (int i = 0; i < 2; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    if (auto st = once(); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_magic("STBO") || !write_exact(&loss, 1) || !write_exact(&n_children, 1)) return 2;
  for (int32_t i = 0; i < n_children; ++i) {
    const auto& ch = stack.children[static_cast<std::size_t>(i)];
    const int32_t nw = welvet::stack::child_grad_w_size(ch);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    if (nw > 0) {
      auto est = welvet::stack::child_export_f32(ch, Wa.data());
      if (!est) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
        return 1;
      }
    }
    if (!write_exact(&nw, 1)) return 2;
    if (nw > 0 && !write_exact(Wa.data(), Wa.size())) return 2;
  }
  return 0;
}
