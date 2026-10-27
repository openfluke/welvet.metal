#pragma once
// Internal versioned model descriptor codec shared by the native ABI and tools.
#include "welvet/architecture.hpp"
#include <cstdio>
#include <stdexcept>
namespace welvet::interop::wire {
template <class R> struct DepthGuard {
  R &r;
  DepthGuard(R &v) : r(v) {
    if (++r.depth > 64)
      throw std::invalid_argument("model nesting exceeds 64");
  }
  ~DepthGuard() { --r.depth; }
};
template <class R> size_t checked_count(R &r, int64_t n, bool empty = false) {
  if (n < 0 || (!empty && n == 0) || n > (1 << 26) || !r.allow_count(size_t(n)))
    throw std::invalid_argument("invalid model payload size");
  return size_t(n);
}
template <class Reader>
bool read_child(Reader &reader, welvet::stack::Child *ch) {
  DepthGuard<Reader> depth(reader);
  int32_t kind = 0;
  if (!reader.read(&kind, 1))
    return false;
  ch->kind = static_cast<welvet::stack::Kind>(kind);
  switch (ch->kind) {
  case welvet::stack::Kind::None: return true;
  case welvet::stack::Kind::Dense: {
    int32_t in = 0, out = 0, act = 0, dtype = 1;
    if (!reader.read(&in, 1) || !reader.read(&out, 1) ||
        !reader.read(&act, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    std::vector<float> W(checked_count(reader, int64_t(out) * in));
    if (!reader.read(W.data(), W.size()))
      return false;
    uint32_t bias_len = 0;
    if (!reader.read(&bias_len, 1))
      return false;
    std::vector<double> bias(checked_count(reader, bias_len, true));
    if (bias_len > 0 && !reader.read(bias.data(), bias.size()))
      return false;
    auto st = welvet::dense::new_layer(
        in, out, static_cast<welvet::Activation>(act),
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
    if (!reader.read(&dim, 1) || !reader.read(&dtype, 1) ||
        !reader.read(&eps, 1))
      return false;
    std::vector<float> g(checked_count(reader, dim));
    if (!reader.read(g.data(), g.size()))
      return false;
    welvet::rmsnorm::Config cfg;
    cfg.dim = dim;
    cfg.eps = eps;
    auto st = welvet::rmsnorm::new_configured(
        cfg, static_cast<welvet::DType>(dtype), g.data(), &ch->rmsnorm_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::LayerNorm: {
    int32_t dim = 0, dtype = 1;
    double eps = 0.0;
    if (!reader.read(&dim, 1) || !reader.read(&dtype, 1) ||
        !reader.read(&eps, 1))
      return false;
    std::vector<float> g(checked_count(reader, dim)),
        b(checked_count(reader, dim));
    if (!reader.read(g.data(), g.size()) || !reader.read(b.data(), b.size()))
      return false;
    welvet::layernorm::Config cfg;
    cfg.dim = dim;
    cfg.eps = eps;
    auto st = welvet::layernorm::new_configured(
        cfg, static_cast<welvet::DType>(dtype), g.data(), b.data(),
        &ch->layernorm_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::SwiGLU: {
    int32_t in = 0, inter = 0, dtype = 1;
    if (!reader.read(&in, 1) || !reader.read(&inter, 1) ||
        !reader.read(&dtype, 1))
      return false;
    const int ng = inter * in;
    const int nu = inter * in;
    const int nd = in * inter;
    std::vector<float> packed(checked_count(reader, ng + nu + nd));
    if (!reader.read(packed.data(), packed.size()))
      return false;
    welvet::swiglu::Config cfg;
    cfg.input_dim = in;
    cfg.intermediate_dim = inter;
    auto st = welvet::swiglu::new_configured(
        cfg, static_cast<welvet::DType>(dtype), packed.data(),
        packed.data() + ng, packed.data() + ng + nu, &ch->swiglu_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::Softmax: {
    int32_t dim = 0, dtype = 1, sk = 0;
    double temperature = 1.0;
    if (!reader.read(&dim, 1) || !reader.read(&dtype, 1) ||
        !reader.read(&sk, 1) || !reader.read(&temperature, 1)) {
      return false;
    }
    welvet::softmax::Config cfg;
    cfg.dim = dim;
    cfg.kind = static_cast<welvet::softmax::Kind>(sk);
    cfg.temperature = temperature;
    auto st = welvet::softmax::new_configured(
        cfg, static_cast<welvet::DType>(dtype), &ch->softmax_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::Embedding: {
    int32_t vocab = 0, emb = 0, seq = 0, act = -1, dtype = 1;
    if (!reader.read(&vocab, 1) || !reader.read(&emb, 1) ||
        !reader.read(&seq, 1) || !reader.read(&act, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    std::vector<float> W(checked_count(reader, int64_t(vocab) * emb));
    if (!reader.read(W.data(), W.size()))
      return false;
    welvet::embedding::Config cfg;
    cfg.vocab_size = vocab;
    cfg.embedding_dim = emb;
    cfg.seq_len = seq;
    auto st = welvet::embedding::new_configured(
        cfg, static_cast<welvet::DType>(dtype),
        static_cast<welvet::Activation>(act), W.data(), &ch->embedding_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::MHA: {
    int32_t seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0,
            msl = 0;
    int32_t mask = 0, pos = 0, dtype = 1;
    if (!reader.read(&seq, 1) || !reader.read(&d_model, 1) ||
        !reader.read(&num_heads, 1) || !reader.read(&num_kv, 1) ||
        !reader.read(&head_dim, 1) || !reader.read(&msl, 1) ||
        !reader.read(&mask, 1) || !reader.read(&pos, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    if (num_heads <= 0 || d_model <= 0 || num_kv < 0 || head_dim < 0)
      return false;
    const int hd = head_dim > 0 ? head_dim : d_model / num_heads;
    const int kv_heads = num_kv > 0 ? num_kv : num_heads;
    const int q_dim = num_heads * hd;
    const int kv_dim = kv_heads * hd;
    std::vector<float> Q(checked_count(reader, q_dim * d_model));
    std::vector<float> K(checked_count(reader, kv_dim * d_model));
    std::vector<float> V(checked_count(reader, kv_dim * d_model));
    std::vector<float> O(checked_count(reader, d_model * q_dim));
    if (!reader.read(Q.data(), Q.size()) || !reader.read(K.data(), K.size()) ||
        !reader.read(V.data(), V.size()) || !reader.read(O.data(), O.size())) {
      return false;
    }
    welvet::mha::Config cfg =
        welvet::mha::decoder_causal(d_model, num_heads, kv_heads);
    cfg.max_seq_len = msl > 0 ? msl : seq;
    cfg.mask = static_cast<welvet::mha::MaskKind>(mask);
    cfg.pos = static_cast<welvet::mha::PosKind>(pos);
    if (head_dim > 0)
      cfg.head_dim = head_dim;
    auto st = welvet::mha::new_configured(
        cfg, static_cast<welvet::DType>(dtype), Q.data(), K.data(), V.data(),
        O.data(), &ch->mha_op);
    if (!st)
      return false;
    (void)seq;
    return true;
  }
  case welvet::stack::Kind::CNN1: {
    int32_t in_ch = 0, filters = 0, seq = 0, kernel = 0, stride = 0, pad = 0,
            act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) ||
        !reader.read(&seq, 1) || !reader.read(&kernel, 1) ||
        !reader.read(&stride, 1) || !reader.read(&pad, 1) ||
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
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::cnn1::new_configured(
        cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->cnn1_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::Sequential:
  case welvet::stack::Kind::Residual: {
    int32_t dim = 0, seq = 0, depth = 0, child_act = -1, dtype = 1;
    if (!reader.read(&dim, 1) || !reader.read(&seq, 1) ||
        !reader.read(&depth, 1) || !reader.read(&child_act, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    std::vector<float> W(checked_count(reader, int64_t(depth) * dim * dim));
    if (!reader.read(W.data(), W.size()))
      return false;
    if (ch->kind == welvet::stack::Kind::Sequential) {
      welvet::sequential::Config cfg;
      cfg.dim = dim;
      cfg.seq_len = seq;
      cfg.depth = depth;
      auto st = welvet::sequential::new_configured(
          cfg, static_cast<welvet::DType>(dtype),
          static_cast<welvet::Activation>(child_act), W.data(),
          &ch->sequential_op);
      if (!st)
        return false;
    } else {
      welvet::residual::Config cfg;
      cfg.dim = dim;
      cfg.seq_len = seq;
      cfg.depth = depth;
      auto st = welvet::residual::new_configured(
          cfg, static_cast<welvet::DType>(dtype),
          static_cast<welvet::Activation>(child_act), W.data(),
          &ch->residual_op);
      if (!st)
        return false;
    }
    return true;
  }
  case welvet::stack::Kind::Mamba: {
    int32_t d_model = 0, d_state = 0, expand = 0, seq = 0, dtype = 1;
    if (!reader.read(&d_model, 1) || !reader.read(&d_state, 1) ||
        !reader.read(&expand, 1) || !reader.read(&seq, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::mamba::Config cfg;
    cfg.d_model = d_model;
    cfg.d_state = d_state;
    cfg.expand = expand;
    cfg.seq_len = seq;
    const int nw = welvet::mamba::weight_count(cfg);
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::mamba::new_configured(
        cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->mamba_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::Metacognition: {
    int32_t dim = 0, seq = 0, dtype = 1;
    if (!reader.read(&dim, 1) || !reader.read(&seq, 1) ||
        !reader.read(&dtype, 1)) {
      return false;
    }
    welvet::metacognition::Config cfg;
    cfg.dim = dim;
    cfg.seq_len = seq == 1 ? 0 : seq;
    const int nw = welvet::metacognition::weight_count(cfg);
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::metacognition::new_configured(
        cfg, static_cast<welvet::DType>(dtype), W.data(),
        &ch->metacognition_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::GDN: {
    int32_t hidden = 0, seq = 0, nk = 0, nv = 0, hd_k = 0, hd_v = 0, conv_k = 0,
            dtype = 1;
    if (!reader.read(&hidden, 1) || !reader.read(&seq, 1) ||
        !reader.read(&nk, 1) || !reader.read(&nv, 1) ||
        !reader.read(&hd_k, 1) || !reader.read(&hd_v, 1) ||
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
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::gdn::new_configured(cfg, W.data(), &ch->gdn_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::KMeans: {
    int32_t k = 0, d = 0, seq = 0, output_mode = 0, act = -1, dtype = 1;
    double temperature = 1.0;
    if (!reader.read(&k, 1) || !reader.read(&d, 1) || !reader.read(&seq, 1) ||
        !reader.read(&output_mode, 1) || !reader.read(&act, 1) ||
        !reader.read(&temperature, 1) || !reader.read(&dtype, 1)) {
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
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::kmeans::new_configured(
        cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->kmeans_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::CNN2: {
    int32_t in_ch = 0, filters = 0, h = 0, w = 0, kernel = 0, stride = 0,
            pad = 0, act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) ||
        !reader.read(&h, 1) || !reader.read(&w, 1) ||
        !reader.read(&kernel, 1) || !reader.read(&stride, 1) ||
        !reader.read(&pad, 1) || !reader.read(&act, 1) ||
        !reader.read(&dtype, 1)) {
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
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::cnn2::new_configured(
        cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->cnn2_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::CNN3: {
    int32_t in_ch = 0, filters = 0, d = 0, h = 0, w = 0, kernel = 0, stride = 0,
            pad = 0, act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) ||
        !reader.read(&d, 1) || !reader.read(&h, 1) || !reader.read(&w, 1) ||
        !reader.read(&kernel, 1) || !reader.read(&stride, 1) ||
        !reader.read(&pad, 1) || !reader.read(&act, 1) ||
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
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::cnn3::new_configured(
        cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->cnn3_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::NestedStack: {
    int32_t n_nested = 0;
    if (!reader.read(&n_nested, 1) || n_nested <= 0 || n_nested > 4096)
      return false;
    ch->nested.clear();
    ch->nested.reserve(checked_count(reader, n_nested));
    for (int32_t i = 0; i < n_nested; ++i) {
      welvet::stack::Child nested;
      if (!read_child(reader, &nested))
        return false;
      ch->nested.push_back(std::move(nested));
    }
    return true;
  }
  case welvet::stack::Kind::Parallel: {
    // combine, dim, out_feat, seq, sparse_k, disagree_beta f64, n_branches,
    // branches…, has_gate, [gate Dense]
    int32_t combine = 1, dim = 0, out_feat = 0, seq = 0, sparse_k = 0, n_br = 0,
            has_gate = 0;
    double disagree_beta = 0.0;
    if (!reader.read(&combine, 1) || !reader.read(&dim, 1) ||
        !reader.read(&out_feat, 1) || !reader.read(&seq, 1) ||
        !reader.read(&sparse_k, 1) || !reader.read(&disagree_beta, 1) ||
        !reader.read(&n_br, 1) || n_br <= 0 || n_br > 4096) {
      return false;
    }
    ch->parallel_nest.combine = combine;
    ch->parallel_nest.dim = dim;
    ch->parallel_nest.out_feat = out_feat;
    ch->parallel_nest.seq_len = seq;
    ch->parallel_nest.sparse_k = sparse_k;
    ch->parallel_nest.disagree_beta = disagree_beta;
    ch->nested.clear();
    ch->nested.reserve(checked_count(reader, n_br));
    for (int32_t i = 0; i < n_br; ++i) {
      welvet::stack::Child nested;
      if (!read_child(reader, &nested))
        return false;
      ch->nested.push_back(std::move(nested));
    }
    if (!reader.read(&has_gate, 1))
      return false;
    ch->parallel_nest.has_gate = has_gate;
    if (has_gate) {
      welvet::stack::Child gate_ch;
      if (!read_child(reader, &gate_ch) ||
          gate_ch.kind != welvet::stack::Kind::Dense)
        return false;
      ch->dense_op = std::move(gate_ch.dense_op);
    }
    return true;
  }
  case welvet::stack::Kind::ConvT1: {
    int32_t in_ch = 0, filters = 0, seq = 0, kernel = 0, stride = 0, pad = 0,
            out_pad = 0, act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) ||
        !reader.read(&seq, 1) || !reader.read(&kernel, 1) ||
        !reader.read(&stride, 1) || !reader.read(&pad, 1) ||
        !reader.read(&out_pad, 1) || !reader.read(&act, 1) ||
        !reader.read(&dtype, 1)) {
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
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::convt1::new_configured(
        cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->convt1_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::ConvT2: {
    int32_t in_ch = 0, filters = 0, h = 0, w = 0, kernel = 0, stride = 0,
            pad = 0, out_pad = 0, act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) ||
        !reader.read(&h, 1) || !reader.read(&w, 1) ||
        !reader.read(&kernel, 1) || !reader.read(&stride, 1) ||
        !reader.read(&pad, 1) || !reader.read(&out_pad, 1) ||
        !reader.read(&act, 1) || !reader.read(&dtype, 1)) {
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
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::convt2::new_configured(
        cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->convt2_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::ConvT3: {
    int32_t in_ch = 0, filters = 0, d = 0, h = 0, w = 0, kernel = 0, stride = 0,
            pad = 0, out_pad = 0, act = 0, dtype = 1;
    if (!reader.read(&in_ch, 1) || !reader.read(&filters, 1) ||
        !reader.read(&d, 1) || !reader.read(&h, 1) || !reader.read(&w, 1) ||
        !reader.read(&kernel, 1) || !reader.read(&stride, 1) ||
        !reader.read(&pad, 1) || !reader.read(&out_pad, 1) ||
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
    std::vector<float> W(checked_count(reader, nw));
    if (!reader.read(W.data(), W.size()))
      return false;
    auto st = welvet::convt3::new_configured(
        cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->convt3_op);
    if (!st)
      return false;
    return true;
  }
  case welvet::stack::Kind::RNN:
  case welvet::stack::Kind::LSTM: {
    int32_t in = 0, hid = 0, seq = 0, dtype = 1;
    if (!reader.read(&in, 1) || !reader.read(&hid, 1) ||
        !reader.read(&seq, 1) || !reader.read(&dtype, 1)) {
      return false;
    }
    if (ch->kind == welvet::stack::Kind::RNN) {
      welvet::rnn::Config cfg;
      cfg.input_size = in;
      cfg.hidden_size = hid;
      cfg.seq_len = seq;
      const int nw = welvet::rnn::weight_count(cfg);
      std::vector<float> W(checked_count(reader, nw));
      if (!reader.read(W.data(), W.size()))
        return false;
      auto st = welvet::rnn::new_configured(
          cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->rnn_op);
      if (!st)
        return false;
    } else {
      welvet::lstm::Config cfg;
      cfg.input_size = in;
      cfg.hidden_size = hid;
      cfg.seq_len = seq;
      const int nw = welvet::lstm::weight_count(cfg);
      std::vector<float> W(checked_count(reader, nw));
      if (!reader.read(W.data(), W.size()))
        return false;
      auto st = welvet::lstm::new_configured(
          cfg, static_cast<welvet::DType>(dtype), W.data(), &ch->lstm_op);
      if (!st)
        return false;
    }
    return true;
  }
  case welvet::stack::Kind::View: {
    int32_t ndim = 0;
    if (!reader.read(&ndim, 1) || ndim <= 0 || ndim > 8)
      return false;
    ch->view_shape.resize(checked_count(reader, ndim));
    for (int32_t i = 0; i < ndim; ++i) {
      int32_t d = 0;
      if (!reader.read(&d, 1) || d <= 0)
        return false;
      ch->view_shape[static_cast<std::size_t>(i)] = d;
    }
    return true;
  }
  case welvet::stack::Kind::Flatten: {
    int32_t feat = 0;
    if (!reader.read(&feat, 1) || feat < 0)
      return false;
    ch->flatten_feat = feat;
    return true;
  }
  default:
    std::fprintf(stderr, "welvet_dense_fwd: STB1 unsupported kind %d\n", kind);
    return false;
  }
}

} // namespace welvet::interop::wire
