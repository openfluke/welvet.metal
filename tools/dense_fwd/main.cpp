// Binary wire tool for w2a go_v_metal — Dense FormatNone forward + timing.
//
// Magics (same wire as welvet.cpp dense slice):
//   WVD3 — Dense forward (WVD2/WVD1 accepted)
//   stdin → stdout
//   --bench=N → WVB2: mount_ns + exec_ns (N warm executes) + WVO1
//               (legacy WVB1 was mount+execute fused; WVB2 splits them)

#include "welvet/gaps.hpp"
#include "welvet/metal/device.hpp"
#include "welvet_c.h"
#include "welvet/architecture.hpp"
#include "welvet/dense.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "volume_fwd.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

static bool read_exact(void* p, size_t n) {
  return std::fread(p, 1, n, stdin) == n;
}
template <typename T>
static bool read_exact(T* p, size_t n) {
  return read_exact(static_cast<void*>(p), n * sizeof(T));
}
static bool write_exact(const void* p, size_t n) {
  return std::fwrite(p, 1, n, stdout) == n;
}
template <typename T>
static bool write_exact(const T* p, size_t n) {
  return write_exact(static_cast<const void*>(p), n * sizeof(T));
}

static int run_dense(bool v1, bool v2, bool v3, int bench_iters) {
  int32_t batch = 0, in = 0, out = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in, 1) || !read_exact(&out, 1) || !read_exact(&act, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short header\n");
    return 2;
  }
  if ((v2 || v3) && !read_exact(&dtype, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short dtype\n");
    return 2;
  }
  if (v3 && !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short backend\n");
    return 2;
  }
  if (batch <= 0 || in <= 0 || out <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad dims\n");
    return 2;
  }

  std::vector<float> W(static_cast<std::size_t>(out * in));
  if (!read_exact(W.data(), W.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short weights\n");
    return 2;
  }

  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short bias_len\n");
    return 2;
  }
  std::vector<double> bias(bias_len);
  if (bias_len > 0 && !read_exact(bias.data(), bias.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short bias\n");
    return 2;
  }

  std::vector<float> x(static_cast<std::size_t>(batch * in));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short input\n");
    return 2;
  }

  std::vector<float> pre(static_cast<std::size_t>(batch * out));
  std::vector<float> post(static_cast<std::size_t>(batch * out));
  const double* bias_ptr = bias_len ? bias.data() : nullptr;

  if (bench_iters > 0) {
    // ---- mount (decode dtype → soft-double + upload W/bias once) ----
    welvet_dense_session* sess = nullptr;
    const auto tm0 = std::chrono::steady_clock::now();
    welvet_status st =
        welvet_dense_mount(batch, in, out, act, dtype, W.data(), bias_ptr, bias_len, &sess);
    const auto tm1 = std::chrono::steady_clock::now();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: mount: %s\n", st.message);
      return 1;
    }
    const int64_t mount_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();

    // warmup execute
    for (int i = 0; i < 5; ++i) {
      st = welvet_dense_execute(sess, x.data(), pre.data(), post.data());
      if (st.code != 0) {
        welvet_dense_release(sess);
        std::fprintf(stderr, "welvet_dense_fwd: warmup: %s\n", st.message);
        return 1;
      }
    }

    // ---- warm execute loop ----
    const auto te0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      st = welvet_dense_execute(sess, x.data(), pre.data(), post.data());
      if (st.code != 0) {
        welvet_dense_release(sess);
        std::fprintf(stderr, "welvet_dense_fwd: exec: %s\n", st.message);
        return 1;
      }
    }
    const auto te1 = std::chrono::steady_clock::now();
    const int64_t exec_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(te1 - te0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);

    welvet_dense_release(sess);

    // WVB2: mount_ns, exec_ns (total for N iters), iters
    if (!write_exact("WVB2", 4) || !write_exact(&mount_ns, 1) || !write_exact(&exec_ns, 1) ||
        !write_exact(&iters32, 1)) {
      return 2;
    }
  } else {
    welvet_status st = welvet_dense_forward_f32(batch, in, out, act, dtype, backend, W.data(),
                                                bias_ptr, bias_len, x.data(), pre.data(),
                                                post.data());
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  if (!write_exact("WVO1", 4) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(post.data(), post.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: write failed\n");
    return 2;
  }
  return 0;
}

static int run_mha(int bench_iters) {
  int32_t batch = 0, seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
  int32_t mask = 0, pos = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&d_model, 1) ||
      !read_exact(&num_heads, 1) || !read_exact(&num_kv, 1) || !read_exact(&head_dim, 1) ||
      !read_exact(&msl, 1) || !read_exact(&mask, 1) || !read_exact(&pos, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVM1 header\n");
    return 2;
  }
  if (num_heads <= 0 || d_model <= 0 || d_model % num_heads != 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad mha dims\n");
    return 2;
  }
  const int hd = head_dim > 0 ? head_dim : d_model / num_heads;
  const int kv_heads = num_kv > 0 ? num_kv : num_heads;
  const int q_dim = num_heads * hd;
  const int kv_dim = kv_heads * hd;

  std::vector<float> Q(static_cast<std::size_t>(q_dim * d_model));
  std::vector<float> K(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> V(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> O(static_cast<std::size_t>(d_model * q_dim));
  if (!read_exact(Q.data(), Q.size()) || !read_exact(K.data(), K.size()) ||
      !read_exact(V.data(), V.size()) || !read_exact(O.data(), O.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVM1 weights\n");
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * d_model));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVM1 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * q_dim));
  std::vector<float> post(static_cast<std::size_t>(batch * seq * d_model));

  auto once = [&]() -> welvet_status {
    return welvet_mha_forward_f32(batch, seq, d_model, num_heads, num_kv, head_dim, msl, mask, pos,
                                  dtype, backend, Q.data(), K.data(), V.data(), O.data(), x.data(),
                                  pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
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
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WVMO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_embedding(int bench_iters) {
  int32_t batch = 0, vocab = 0, emb = 0, seq = 0, in_rank = 2, in_last = 1, act = -1, dtype = 1,
          backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&vocab, 1) || !read_exact(&emb, 1) ||
      !read_exact(&seq, 1) || !read_exact(&in_rank, 1) || !read_exact(&in_last, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WEM1 header\n");
    return 2;
  }
  if (batch <= 0 || vocab <= 0 || emb <= 0 || seq <= 0 || in_last <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad embedding dims\n");
    return 2;
  }
  std::vector<float> W(static_cast<std::size_t>(vocab * emb));
  if (!read_exact(W.data(), W.size())) return 2;
  const int x_n = batch * seq * in_last;
  std::vector<float> x(static_cast<std::size_t>(x_n));
  if (!read_exact(x.data(), x.size())) return 2;
  const int out_n = (in_rank == 1 ? seq : batch * seq) * emb;
  std::vector<float> pre(static_cast<std::size_t>(out_n));
  std::vector<float> post(pre.size());

  auto once = [&]() -> welvet_status {
    return welvet_embedding_forward_f32(batch, vocab, emb, seq, in_rank, in_last, act, dtype,
                                        backend, W.data(), x.data(), pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
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
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WEMO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_sequential(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, depth = 0, child_act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&depth, 1) || !read_exact(&child_act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WSQ1 header\n");
    return 2;
  }
  if (batch <= 0 || seq <= 0 || dim <= 0 || depth <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad sequential dims\n");
    return 2;
  }
  const int wn = depth * dim * dim;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());

  auto once = [&]() -> welvet_status {
    return welvet_sequential_forward_f32(batch, seq, dim, depth, child_act, dtype, backend,
                                         packed.data(), x.data(), pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
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
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WSQO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_residual(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, depth = 0, child_act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&depth, 1) || !read_exact(&child_act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WRE1 header\n");
    return 2;
  }
  if (batch <= 0 || seq <= 0 || dim <= 0 || depth <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad residual dims\n");
    return 2;
  }
  const int wn = depth * dim * dim;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());

  auto once = [&]() -> welvet_status {
    return welvet_residual_forward_f32(batch, seq, dim, depth, child_act, dtype, backend,
                                       packed.data(), x.data(), pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
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
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WREO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_softmax(int bench_iters) {
  int32_t n = 0, dim = 0, seq = 0, kind = 0, rows = 0, cols = 0, gumbel_seed = 0, dtype = 1,
          backend = 0, rank = 0, mask_len = 0, hier_len = 0;
  double temperature = 1.0, entmax_alpha = 0.0;
  if (!read_exact(&n, 1) || !read_exact(&dim, 1) || !read_exact(&seq, 1) || !read_exact(&kind, 1) ||
      !read_exact(&rows, 1) || !read_exact(&cols, 1) || !read_exact(&temperature, 1) ||
      !read_exact(&entmax_alpha, 1) || !read_exact(&gumbel_seed, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1) || !read_exact(&rank, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WSM1 header\n");
    return 2;
  }
  if (rank < 1 || rank > 8) {
    std::fprintf(stderr, "welvet_dense_fwd: bad softmax rank\n");
    return 2;
  }
  std::vector<int32_t> shape(static_cast<std::size_t>(rank));
  if (!read_exact(shape.data(), shape.size())) return 2;
  if (!read_exact(&mask_len, 1)) return 2;
  std::vector<uint8_t> mask(static_cast<std::size_t>(mask_len > 0 ? mask_len : 0));
  if (mask_len > 0 && !read_exact(mask.data(), mask.size())) return 2;
  if (!read_exact(&hier_len, 1)) return 2;
  std::vector<int32_t> hier(static_cast<std::size_t>(hier_len > 0 ? hier_len : 0));
  if (hier_len > 0 && !read_exact(hier.data(), hier.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(n));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(n));
  std::vector<float> post(pre.size());

  auto once = [&]() -> welvet_status {
    return welvet_softmax_forward_f32(
        n, dim, seq, kind, rows, cols, temperature, entmax_alpha, gumbel_seed, dtype, backend, rank,
        shape.data(), mask_len > 0 ? mask.data() : nullptr, mask_len,
        hier_len > 0 ? hier.data() : nullptr, hier_len, x.data(), pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
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
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WSMO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_layernorm(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&eps, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVL1 header\n");
    return 2;
  }
  if (batch <= 0 || seq <= 0 || dim <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad layernorm dims\n");
    return 2;
  }

  std::vector<float> gamma(static_cast<std::size_t>(dim));
  std::vector<float> beta(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size()) || !read_exact(beta.data(), beta.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVL1 affine\n");
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVL1 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * dim));
  std::vector<float> post(static_cast<std::size_t>(batch * seq * dim));

  auto once = [&]() -> welvet_status {
    return welvet_layernorm_forward_f32(batch, seq, dim, eps, dtype, backend, gamma.data(),
                                        beta.data(), x.data(), pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
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
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WVLO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_rmsnorm(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&eps, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVR1 header\n");
    return 2;
  }
  if (batch <= 0 || seq <= 0 || dim <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad rmsnorm dims\n");
    return 2;
  }

  std::vector<float> gamma(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVR1 gamma\n");
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVR1 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * dim));
  std::vector<float> post(static_cast<std::size_t>(batch * seq * dim));

  auto once = [&]() -> welvet_status {
    return welvet_rmsnorm_forward_f32(batch, seq, dim, eps, dtype, backend, gamma.data(), x.data(),
                                      pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
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
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WVRO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_swiglu(int bench_iters) {
  int32_t batch = 0, seq = 0, input_dim = 0, inter = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&input_dim, 1) ||
      !read_exact(&inter, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVS1 header\n");
    return 2;
  }
  if (batch <= 0 || seq <= 0 || input_dim <= 0 || inter <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad swiglu dims\n");
    return 2;
  }

  std::vector<float> Gate(static_cast<std::size_t>(inter * input_dim));
  std::vector<float> Up(static_cast<std::size_t>(inter * input_dim));
  std::vector<float> Down(static_cast<std::size_t>(input_dim * inter));
  if (!read_exact(Gate.data(), Gate.size()) || !read_exact(Up.data(), Up.size()) ||
      !read_exact(Down.data(), Down.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVS1 weights\n");
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * input_dim));
  if (!read_exact(x.data(), x.size())) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVS1 input\n");
    return 2;
  }
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * inter));
  std::vector<float> post(static_cast<std::size_t>(batch * seq * input_dim));

  auto once = [&]() -> welvet_status {
    return welvet_swiglu_forward_f32(batch, seq, input_dim, inter, dtype, backend, Gate.data(),
                                     Up.data(), Down.data(), x.data(), pre.data(), post.data());
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
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
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WVSO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int write_cnn_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WVCO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int bench_or_once_cnn(int bench_iters, const std::function<welvet_status()>& once,
                             std::vector<float>& /*pre*/, std::vector<float>& /*post*/) {
  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      welvet_status st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
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
    welvet_status st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }
  return 0;
}

static int run_cnn1(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0;
  int32_t act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&seq, 1) || !read_exact(&kernel, 1) || !read_exact(&stride, 1) ||
      !read_exact(&padding, 1) || !read_exact(&act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC1 header\n");
    return 2;
  }
  const int patch = in_c * kernel;
  const int s = stride <= 0 ? 1 : stride;
  const int out_len = (seq + 2 * padding - kernel) / s + 1;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * seq));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_len));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_cnn1_forward_f32(batch, in_c, filters, seq, kernel, stride, padding, act, dtype,
                                   backend, W.data(), bias.data(), bias.size(), x.data(),
                                   pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_cnn_out(pre, post);
}

static int run_cnn2(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0,
          padding = 0;
  int32_t act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&height, 1) || !read_exact(&width, 1) || !read_exact(&kernel, 1) ||
      !read_exact(&stride, 1) || !read_exact(&padding, 1) || !read_exact(&act, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC2 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_h = (height + 2 * padding - kernel) / s + 1;
  const int out_w = (width + 2 * padding - kernel) / s + 1;
  const int patch = in_c * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_h * out_w));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_cnn2_forward_f32(batch, in_c, filters, height, width, kernel, stride, padding,
                                   act, dtype, backend, W.data(), bias.data(), bias.size(),
                                   x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_cnn_out(pre, post);
}

static int run_cnn3(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, depth = 0, height = 0, width = 0, kernel = 0,
          stride = 0, padding = 0;
  int32_t act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&depth, 1) || !read_exact(&height, 1) || !read_exact(&width, 1) ||
      !read_exact(&kernel, 1) || !read_exact(&stride, 1) || !read_exact(&padding, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVC3 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_d = (depth + 2 * padding - kernel) / s + 1;
  const int out_h = (height + 2 * padding - kernel) / s + 1;
  const int out_w = (width + 2 * padding - kernel) / s + 1;
  const int patch = in_c * kernel * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * depth * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_d * out_h * out_w));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_cnn3_forward_f32(batch, in_c, filters, depth, height, width, kernel, stride,
                                   padding, act, dtype, backend, W.data(), bias.data(),
                                   bias.size(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_cnn_out(pre, post);
}

static int write_convt_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WVTO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int write_rnn_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WRNO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int write_lstm_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WLSO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_rnn(int bench_iters) {
  int32_t batch = 0, input_size = 0, hidden_size = 0, seq_len = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&input_size, 1) || !read_exact(&hidden_size, 1) ||
      !read_exact(&seq_len, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WRN1 header\n");
    return 2;
  }
  const int wn = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq_len * input_size));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * seq_len * hidden_size));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_rnn_forward_f32(batch, input_size, hidden_size, seq_len, dtype, backend,
                                  packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_rnn_out(pre, post);
}

static int run_lstm(int bench_iters) {
  int32_t batch = 0, input_size = 0, hidden_size = 0, seq_len = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&input_size, 1) || !read_exact(&hidden_size, 1) ||
      !read_exact(&seq_len, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WLS1 header\n");
    return 2;
  }
  const int gate = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
  const int wn = 4 * gate;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq_len * input_size));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * seq_len * 5 * hidden_size));
  std::vector<float> post(static_cast<std::size_t>(batch * seq_len * hidden_size));
  auto once = [&]() -> welvet_status {
    return welvet_lstm_forward_f32(batch, input_size, hidden_size, seq_len, dtype, backend,
                                   packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_lstm_out(pre, post);
}

static int write_gdn_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WGDO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_gdn(int bench_iters) {
  int32_t batch = 0, seq = 0, hidden = 0, nk = 0, nv = 0, hd_k = 0, hd_v = 0, conv_k = 0;
  int32_t dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&hidden, 1) ||
      !read_exact(&nk, 1) || !read_exact(&nv, 1) || !read_exact(&hd_k, 1) || !read_exact(&hd_v, 1) ||
      !read_exact(&conv_k, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WGD1 header\n");
    return 2;
  }
  int conv_eff = conv_k;
  if (conv_eff < 1) conv_eff = 4;
  const int key_d = nk * hd_k;
  const int val_d = nv * hd_v;
  const int conv_d = key_d * 2 + val_d;
  const int wn = conv_d * hidden + val_d * hidden + nv * hidden + nv * hidden + hidden * val_d +
                 conv_d * conv_eff + nv + nv + hd_v;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * hidden));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());
  auto once = [&]() -> welvet_status {
    return welvet_gdn_forward_f32(batch, seq, hidden, nk, nv, hd_k, hd_v, conv_k, dtype, backend,
                                  packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_gdn_out(pre, post);
}

static int write_mamba_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WMBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_mamba(int bench_iters) {
  int32_t batch = 0, seq = 0, d_model = 0, d_state = 0, expand = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&d_model, 1) ||
      !read_exact(&d_state, 1) || !read_exact(&expand, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WMB1 header\n");
    return 2;
  }
  int expand_eff = expand;
  if (expand_eff <= 0) expand_eff = 2;
  const int inner = expand_eff * d_model;
  const int wn = inner * (3 * d_model + 2);
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * d_model));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());
  auto once = [&]() -> welvet_status {
    return welvet_mamba_forward_f32(batch, seq, d_model, d_state, expand, dtype, backend,
                                    packed.data(), x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_mamba_out(pre, post);
}

static int write_kmeans_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WKMO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int write_pl_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WPLO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int write_mc_out(const std::vector<float>& pre, const std::vector<float>& post) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  if (!write_exact("WMCO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size())) {
    return 2;
  }
  return 0;
}

static int run_metacognition(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WMC1 header\n");
    return 2;
  }
  const int wn = dim * dim;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(x.size());
  std::vector<float> post(x.size());
  auto once = [&]() -> welvet_status {
    return welvet_metacognition_forward_f32(batch, seq, dim, dtype, backend, packed.data(),
                                            x.data(), pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_mc_out(pre, post);
}

static int run_parallel(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, out_feat = 0, branches = 0, combine = 0, sparse_k = 0;
  int32_t dtype = 1, backend = 0;
  double disagree_beta = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&out_feat, 1) || !read_exact(&branches, 1) || !read_exact(&combine, 1) ||
      !read_exact(&sparse_k, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1) ||
      !read_exact(&disagree_beta, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WPL1 header\n");
    return 2;
  }
  int gate_n = 0;
  if (combine == 3) gate_n = branches * dim; // filter
  const int wn = branches * out_feat * dim + gate_n;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int out_d = out_feat;
  if (combine == 0) out_d = branches * out_feat; // concat
  std::vector<float> pre(x.size());
  std::vector<float> post(static_cast<std::size_t>(batch * seq * out_d));
  auto once = [&]() -> welvet_status {
    return welvet_parallel_forward_f32(batch, seq, dim, out_feat, branches, combine, sparse_k,
                                       dtype, backend, disagree_beta, packed.data(), x.data(),
                                       pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_pl_out(pre, post);
}

static int run_kmeans(int bench_iters) {
  int32_t n = 0, feature_dim = 0, num_clusters = 0, output_mode = 0, act = -1, dtype = 1,
          backend = 0, rank = 0;
  double temperature = 1.0;
  if (!read_exact(&n, 1) || !read_exact(&feature_dim, 1) || !read_exact(&num_clusters, 1) ||
      !read_exact(&output_mode, 1) || !read_exact(&act, 1) || !read_exact(&temperature, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1) || !read_exact(&rank, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WKM1 header\n");
    return 2;
  }
  if (rank < 1 || rank > 8) {
    std::fprintf(stderr, "welvet_dense_fwd: bad kmeans rank\n");
    return 2;
  }
  std::vector<int32_t> shape(static_cast<std::size_t>(rank));
  if (!read_exact(shape.data(), shape.size())) return 2;
  std::vector<float> W(static_cast<std::size_t>(num_clusters * feature_dim));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(n));
  if (!read_exact(x.data(), x.size())) return 2;
  int batch = 1;
  for (int i = 0; i < rank - 1; ++i) batch *= shape[static_cast<std::size_t>(i)];
  const int out_dim = output_mode == 1 ? feature_dim : num_clusters;
  std::vector<float> pre(static_cast<std::size_t>(batch * out_dim));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_kmeans_forward_f32(n, feature_dim, num_clusters, output_mode, act, temperature,
                                     dtype, backend, rank, shape.data(), W.data(), x.data(),
                                     pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_kmeans_out(pre, post);
}

static int run_convt1(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0,
          out_pad = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&seq, 1) || !read_exact(&kernel, 1) || !read_exact(&stride, 1) ||
      !read_exact(&padding, 1) || !read_exact(&out_pad, 1) || !read_exact(&act, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVT1 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_len = (seq - 1) * s - 2 * padding + kernel + out_pad;
  const int patch = in_c * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * seq));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_len));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_convt1_forward_f32(batch, in_c, filters, seq, kernel, stride, padding, out_pad,
                                     act, dtype, backend, W.data(), x.data(), pre.data(),
                                     post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_convt_out(pre, post);
}

static int run_convt2(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0,
          padding = 0, out_pad = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&height, 1) || !read_exact(&width, 1) || !read_exact(&kernel, 1) ||
      !read_exact(&stride, 1) || !read_exact(&padding, 1) || !read_exact(&out_pad, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVT2 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_h = (height - 1) * s - 2 * padding + kernel + out_pad;
  const int out_w = (width - 1) * s - 2 * padding + kernel + out_pad;
  const int patch = in_c * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_h * out_w));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_convt2_forward_f32(batch, in_c, filters, height, width, kernel, stride, padding,
                                     out_pad, act, dtype, backend, W.data(), x.data(), pre.data(),
                                     post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_convt_out(pre, post);
}

static int run_convt3(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, depth = 0, height = 0, width = 0, kernel = 0,
          stride = 0, padding = 0, out_pad = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&depth, 1) || !read_exact(&height, 1) || !read_exact(&width, 1) ||
      !read_exact(&kernel, 1) || !read_exact(&stride, 1) || !read_exact(&padding, 1) ||
      !read_exact(&out_pad, 1) || !read_exact(&act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVT3 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_d = (depth - 1) * s - 2 * padding + kernel + out_pad;
  const int out_h = (height - 1) * s - 2 * padding + kernel + out_pad;
  const int out_w = (width - 1) * s - 2 * padding + kernel + out_pad;
  const int patch = in_c * kernel * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * depth * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  std::vector<float> pre(static_cast<std::size_t>(batch * filters * out_d * out_h * out_w));
  std::vector<float> post(pre.size());
  auto once = [&]() -> welvet_status {
    return welvet_convt3_forward_f32(batch, in_c, filters, depth, height, width, kernel, stride,
                                     padding, out_pad, act, dtype, backend, W.data(), x.data(),
                                     pre.data(), post.data());
  };
  if (int rc = bench_or_once_cnn(bench_iters, once, pre, post); rc != 0) return rc;
  return write_convt_out(pre, post);
}

static int run_seqmix(int /*bench_iters*/) {
  int32_t n = 0;
  if (!read_exact(&n, 1) || n < 0 || n > 64) {
    std::fprintf(stderr, "welvet_dense_fwd: short/bad WSX1 header\n");
    return 2;
  }
  std::vector<int32_t> kinds(static_cast<std::size_t>(n));
  if (n > 0 && !read_exact(kinds.data(), kinds.size())) return 2;
  if (!write_exact("WSXO", 4) || !write_exact(&n, 1)) return 2;
  for (int32_t k : kinds) {
    char name[64]{};
    welvet_status st = welvet_seqmix_kind_name(k, name, static_cast<int32_t>(sizeof(name)));
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: seqmix kind %d: %s\n", static_cast<int>(k),
                   st.message);
      return 2;
    }
    const int32_t len = static_cast<int32_t>(std::strlen(name));
    if (!write_exact(&len, 1) || !write_exact(name, static_cast<std::size_t>(len))) return 2;
  }
  const int32_t tags[3] = {welvet_seqmix_mha_kind(), welvet_seqmix_mamba_kind(),
                           welvet_seqmix_gdn_kind()};
  const int32_t n_tags = 3;
  if (!write_exact(&n_tags, 1) || !write_exact(tags, 3)) return 2;
  return 0;
}


static void usage() {
  std::fprintf(stderr,
               "welvet_dense_fwd (welvet.metal)\n"
               "  --version | --identity | --gaps | --help\n"
               "  --bench=N → WVB2 mount_ns+exec_ns for volume wires\n"
               "  stdin WVD3→WVO1 | WVM1→WVMO | … | WVF1/WMV1/…/MCG1→WVFO\n"
               "       WDB1→WDBO | WVN1→WVNO (Dense NormalBP)\n"
               "       WRB1→WRBO | WRV1→WRVO (RMSNorm NormalBP)\n"
               "       WLB1→WLBO | WLV1→WLVO (LayerNorm NormalBP)\n"
               "       WSB1→WSBO | WSV1→WSVO (SwiGLU NormalBP)\n"
               "       WXB1→WXBO | WXV1→WXVO (Softmax NormalBP)\n"
               "       WEB1→WEBO | WEV1→WEVO (Embedding NormalBP)\n"
               "       WHB1→WHBO | WHV1→WHVO (MHA NormalBP)\n");
}

int main(int argc, char** argv) {
  int bench_iters = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--version" || a == "-v") {
      std::printf("%s\n", welvet_version());
      return 0;
    }
    if (a == "--identity") {
      auto st = welvet_metal_init();
      if (st.code != 0) {
        std::fprintf(stderr, "%s\n", st.message);
        return 1;
      }
      std::printf("%s\n", welvet_runtime_identity());
      return 0;
    }
    if (a == "--gaps") {
      welvet::gaps::print_census(std::cout);
      return 0;
    }
    if (a == "--help" || a == "-h") {
      usage();
      return 0;
    }
    if (a.rfind("--bench=", 0) == 0) {
      bench_iters = std::atoi(a.c_str() + 8);
      continue;
    }
    std::fprintf(stderr, "unknown arg %s\n", a.c_str());
    usage();
    return 2;
  }

  char magic[4];
  if (!read_exact(magic, 4)) {
    usage();
    return 2;
  }
  (void)welvet_metal_init();
  if (std::memcmp(magic, "WVM1", 4) == 0) {
    return run_mha(bench_iters);
  }
  if (std::memcmp(magic, "WVS1", 4) == 0) {
    return run_swiglu(bench_iters);
  }
  if (std::memcmp(magic, "WVR1", 4) == 0) {
    return run_rmsnorm(bench_iters);
  }
  if (std::memcmp(magic, "WVL1", 4) == 0) {
    return run_layernorm(bench_iters);
  }
  if (std::memcmp(magic, "WSM1", 4) == 0) {
    return run_softmax(bench_iters);
  }
  if (std::memcmp(magic, "WEM1", 4) == 0) {
    return run_embedding(bench_iters);
  }
  if (std::memcmp(magic, "WSQ1", 4) == 0) {
    return run_sequential(bench_iters);
  }
  if (std::memcmp(magic, "WRE1", 4) == 0) {
    return run_residual(bench_iters);
  }
  if (std::memcmp(magic, "WSX1", 4) == 0) {
    return run_seqmix(bench_iters);
  }
  if (std::memcmp(magic, "WVC1", 4) == 0) {
    return run_cnn1(bench_iters);
  }
  if (std::memcmp(magic, "WVC2", 4) == 0) {
    return run_cnn2(bench_iters);
  }
  if (std::memcmp(magic, "WVC3", 4) == 0) {
    return run_cnn3(bench_iters);
  }
  if (std::memcmp(magic, "WVT1", 4) == 0) {
    return run_convt1(bench_iters);
  }
  if (std::memcmp(magic, "WVT2", 4) == 0) {
    return run_convt2(bench_iters);
  }
  if (std::memcmp(magic, "WVT3", 4) == 0) {
    return run_convt3(bench_iters);
  }
  if (std::memcmp(magic, "WRN1", 4) == 0) {
    return run_rnn(bench_iters);
  }
  if (std::memcmp(magic, "WLS1", 4) == 0) {
    return run_lstm(bench_iters);
  }
  if (std::memcmp(magic, "WGD1", 4) == 0) {
    return run_gdn(bench_iters);
  }
  if (std::memcmp(magic, "WMB1", 4) == 0) {
    return run_mamba(bench_iters);
  }
  if (std::memcmp(magic, "WKM1", 4) == 0) {
    return run_kmeans(bench_iters);
  }
  if (std::memcmp(magic, "WPL1", 4) == 0) {
    return run_parallel(bench_iters);
  }
  if (std::memcmp(magic, "WMC1", 4) == 0) {
    return run_metacognition(bench_iters);
  }
  if (std::memcmp(magic, "WDB1", 4) == 0) {
    return run_dense_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WVN1", 4) == 0) {
    return run_volume_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WRB1", 4) == 0) {
    return run_rmsnorm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WRV1", 4) == 0) {
    return run_volume_rmsnorm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WLB1", 4) == 0) {
    return run_layernorm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WLV1", 4) == 0) {
    return run_volume_layernorm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WSB1", 4) == 0) {
    return run_swiglu_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WSV1", 4) == 0) {
    return run_volume_swiglu_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WXB1", 4) == 0) {
    return run_softmax_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WXV1", 4) == 0) {
    return run_volume_softmax_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WEB1", 4) == 0) {
    return run_embedding_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WEV1", 4) == 0) {
    return run_volume_embedding_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WHB1", 4) == 0) {
    return run_mha_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WHV1", 4) == 0) {
    return run_volume_mha_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "C1B1", 4) == 0) {
    return run_cnn1_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "C2B1", 4) == 0) {
    return run_cnn2_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "C3B1", 4) == 0) {
    return run_cnn3_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "C1V1", 4) == 0) {
    return run_volume_cnn1_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "C2V1", 4) == 0) {
    return run_volume_cnn2_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "C3V1", 4) == 0) {
    return run_volume_cnn3_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "T1B1", 4) == 0) {
    return run_convt1_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "T2B1", 4) == 0) {
    return run_convt2_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "T3B1", 4) == 0) {
    return run_convt3_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "T1V1", 4) == 0) {
    return run_volume_convt1_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "T2V1", 4) == 0) {
    return run_volume_convt2_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "T3V1", 4) == 0) {
    return run_volume_convt3_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "RNB1", 4) == 0) {
    return run_rnn_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "RNV1", 4) == 0) {
    return run_volume_rnn_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "LSB1", 4) == 0) {
    return run_lstm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "LSV1", 4) == 0) {
    return run_volume_lstm_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "SQB1", 4) == 0) {
    return run_sequential_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "SQV1", 4) == 0) {
    return run_volume_sequential_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "REB1", 4) == 0) {
    return run_residual_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "REV1", 4) == 0) {
    return run_volume_residual_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "MBB1", 4) == 0) {
    return run_mamba_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "MBV1", 4) == 0) {
    return run_volume_mamba_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "MCB1", 4) == 0) {
    return run_metacognition_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "MCV1", 4) == 0) {
    return run_volume_metacognition_normalbp(bench_iters);
  }
  if (std::memcmp(magic,"NBL1",4)==0)return run_normalbp_child(bench_iters);
  if (std::memcmp(magic,"KMB1",4)==0)return run_kmeans_normalbp(bench_iters);
  if (std::memcmp(magic,"KMTV",4)==0)return run_volume_kmeans_normalbp(bench_iters,VolumeTrainMode::Tween);
  if (std::memcmp(magic,"KMTC",4)==0)return run_volume_kmeans_normalbp(bench_iters,VolumeTrainMode::TweenChain);
  if (std::memcmp(magic,"PNTV",4)==0)return run_volume_parallel_normalbp(bench_iters,VolumeTrainMode::Tween);
  if (std::memcmp(magic,"PNTC",4)==0)return run_volume_parallel_normalbp(bench_iters,VolumeTrainMode::TweenChain);
  if (std::memcmp(magic,"KMV1",4)==0)return run_volume_kmeans_normalbp(bench_iters);
  if (std::memcmp(magic,"PNV1",4)==0)return run_volume_parallel_normalbp(bench_iters);
  if (std::memcmp(magic,"PNB1",4)==0)return run_parallel_normalbp(bench_iters);
  if (std::memcmp(magic,"PPB1",4)==0)return run_parallel_poly_train(bench_iters);
  if (std::memcmp(magic, "GDB1", 4) == 0) {
    return run_gdn_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "GDV1", 4) == 0) {
    return run_volume_gdn_normalbp(bench_iters);
  }
  if (std::memcmp(magic,"TAV1",4)==0) return run_volume_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"TSP1",4)==0) return run_volume_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"TRA1",4)==0) return run_volume_rmsnorm_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"TRP1",4)==0) return run_volume_rmsnorm_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"TLA1",4)==0) return run_volume_layernorm_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"TLP1",4)==0) return run_volume_layernorm_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"TGA1",4)==0) return run_volume_swiglu_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"TGP1",4)==0) return run_volume_swiglu_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"TXA1",4)==0) return run_volume_softmax_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"TXP1",4)==0) return run_volume_softmax_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"TEA1",4)==0) return run_volume_embedding_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"TEP1",4)==0) return run_volume_embedding_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"THA1",4)==0) return run_volume_mha_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"THP1",4)==0) return run_volume_mha_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"C1TA",4)==0) return run_volume_cnn1_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"C1TP",4)==0) return run_volume_cnn1_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"C2TA",4)==0) return run_volume_cnn2_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"C2TP",4)==0) return run_volume_cnn2_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"C3TA",4)==0) return run_volume_cnn3_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"C3TP",4)==0) return run_volume_cnn3_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"T1TA",4)==0) return run_volume_convt1_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"T1TP",4)==0) return run_volume_convt1_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"T2TA",4)==0) return run_volume_convt2_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"T2TP",4)==0) return run_volume_convt2_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"T3TA",4)==0) return run_volume_convt3_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"T3TP",4)==0) return run_volume_convt3_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"RNTA",4)==0) return run_volume_rnn_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"RNTP",4)==0) return run_volume_rnn_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"LSTA",4)==0) return run_volume_lstm_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"LSTP",4)==0) return run_volume_lstm_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"SQTA",4)==0) return run_volume_sequential_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"SQTP",4)==0) return run_volume_sequential_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"RETA",4)==0) return run_volume_residual_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"RETP",4)==0) return run_volume_residual_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"MBTA",4)==0) return run_volume_mamba_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"MBTP",4)==0) return run_volume_mamba_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"MCTA",4)==0) return run_volume_metacognition_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"MCTP",4)==0) return run_volume_metacognition_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"GDTA",4)==0) return run_volume_gdn_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"GDTP",4)==0) return run_volume_gdn_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"SKTA",4)==0) return run_volume_stack_normalbp(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"SKTP",4)==0) return run_volume_stack_normalbp(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"TSB1",4)==0) return run_tween_leaf(bench_iters,VolumeTrainMode::TweenSplit);
  if (std::memcmp(magic,"TAB1",4)==0) return run_tween_leaf(bench_iters,VolumeTrainMode::TweenAlt);
  if (std::memcmp(magic,"STW1",4)==0) return run_stepbp(bench_iters,welvet::runtime::line::Credit::FamilyTween,"STWO");
  if (std::memcmp(magic,"STC1",4)==0) return run_stepbp(bench_iters,welvet::runtime::line::Credit::FamilyBP,"STCO");
  if (std::memcmp(magic,"STS1",4)==0) return run_stepbp(bench_iters,welvet::runtime::line::Credit::FamilyTweenSplit,"STSO");
  if (std::memcmp(magic,"STA1",4)==0) return run_stepbp(bench_iters,welvet::runtime::line::Credit::FamilyTweenAlt,"STAO");
  if (std::memcmp(magic,"MTW1",4)==0) return run_meshtween(bench_iters);
  if (std::memcmp(magic,"MTC1",4)==0) return run_meshtweenchain(bench_iters);
  if (std::memcmp(magic, "TCB1", 4) == 0) return run_tween_leaf(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "TCV1", 4) == 0) return run_volume_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "TRC1", 4) == 0) return run_volume_rmsnorm_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "TLC1", 4) == 0) return run_volume_layernorm_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "TSC1", 4) == 0) return run_volume_swiglu_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "TXC1", 4) == 0) return run_volume_softmax_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "TEC1", 4) == 0) return run_volume_embedding_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "THC1", 4) == 0) return run_volume_mha_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "C1TC", 4) == 0) return run_volume_cnn1_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "C2TC", 4) == 0) return run_volume_cnn2_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "C3TC", 4) == 0) return run_volume_cnn3_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "T1TC", 4) == 0) return run_volume_convt1_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "T2TC", 4) == 0) return run_volume_convt2_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "T3TC", 4) == 0) return run_volume_convt3_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "RNTC", 4) == 0) return run_volume_rnn_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "LSTC", 4) == 0) return run_volume_lstm_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "SQTC", 4) == 0) return run_volume_sequential_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "RETC", 4) == 0) return run_volume_residual_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "MBTC", 4) == 0) return run_volume_mamba_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "MCTC", 4) == 0) return run_volume_metacognition_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "GDTC", 4) == 0) return run_volume_gdn_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "SKTC", 4) == 0) return run_volume_stack_normalbp(bench_iters, VolumeTrainMode::TweenChain);
  if (std::memcmp(magic, "TWB1", 4) == 0) return run_tween_leaf(bench_iters);
  if (std::memcmp(magic, "TWV1", 4) == 0) return run_volume_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "TRV1", 4) == 0) return run_volume_rmsnorm_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "TLV1", 4) == 0) return run_volume_layernorm_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "TSV1", 4) == 0) return run_volume_swiglu_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "TXV1", 4) == 0) return run_volume_softmax_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "TEV1", 4) == 0) return run_volume_embedding_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "THV1", 4) == 0) return run_volume_mha_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "C1TV", 4) == 0) return run_volume_cnn1_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "C2TV", 4) == 0) return run_volume_cnn2_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "C3TV", 4) == 0) return run_volume_cnn3_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "T1TV", 4) == 0) return run_volume_convt1_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "T2TV", 4) == 0) return run_volume_convt2_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "T3TV", 4) == 0) return run_volume_convt3_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "RNTV", 4) == 0) return run_volume_rnn_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "LSTV", 4) == 0) return run_volume_lstm_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "SQTV", 4) == 0) return run_volume_sequential_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "RETV", 4) == 0) return run_volume_residual_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "MBTV", 4) == 0) return run_volume_mamba_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "MCTV", 4) == 0) return run_volume_metacognition_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "GDTV", 4) == 0) return run_volume_gdn_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "SKTV", 4) == 0) return run_volume_stack_normalbp(bench_iters, VolumeTrainMode::Tween);
  if (std::memcmp(magic, "STB1", 4) == 0) {
    return run_stepbp(bench_iters);
  }
  if (std::memcmp(magic,"MGB1",4)==0) return run_mesh_generic_bp(bench_iters);
  if (std::memcmp(magic, "MBP1", 4) == 0) {
    return run_meshbp(bench_iters);
  }
  if (std::memcmp(magic, "STKB", 4) == 0) {
    return run_stack_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "STKV", 4) == 0) {
    return run_volume_stack_normalbp(bench_iters);
  }
  if (std::memcmp(magic, "WVF1", 4) == 0) {
    return run_volume_dense_fwd(bench_iters);
  }
  if (std::memcmp(magic, "WMV1", 4) == 0) {
    return run_volume_mha_fwd(bench_iters);
  }
  if (std::memcmp(magic, "WRF1", 4) == 0) {
    return run_volume_rmsnorm_fwd(bench_iters);
  }
  if (std::memcmp(magic, "WLF1", 4) == 0) {
    return run_volume_layernorm_fwd(bench_iters);
  }
  if (std::memcmp(magic, "WSF1", 4) == 0) {
    return run_volume_softmax_fwd(bench_iters);
  }
  if (std::memcmp(magic, "WPF1", 4) == 0) {
    return run_volume_parallel_fwd(bench_iters);
  }
  if (std::memcmp(magic, "WKF1", 4) == 0) {
    return run_volume_kmeans_fwd(bench_iters);
  }
  if (std::memcmp(magic, "WSG1", 4) == 0) {
    return run_volume_swiglu_fwd(bench_iters);
  }
  if (std::memcmp(magic, "SQG1", 4) == 0) {
    return run_volume_sequential_fwd(bench_iters);
  }
  if (std::memcmp(magic, "REG1", 4) == 0) {
    return run_volume_residual_fwd(bench_iters);
  }
  if (std::memcmp(magic, "RNG1", 4) == 0) {
    return run_volume_rnn_fwd(bench_iters);
  }
  if (std::memcmp(magic, "LSG1", 4) == 0) {
    return run_volume_lstm_fwd(bench_iters);
  }
  if (std::memcmp(magic, "C1G1", 4) == 0) {
    return run_volume_cnn1_fwd(bench_iters);
  }
  if (std::memcmp(magic, "C2G1", 4) == 0) {
    return run_volume_cnn2_fwd(bench_iters);
  }
  if (std::memcmp(magic, "C3G1", 4) == 0) {
    return run_volume_cnn3_fwd(bench_iters);
  }
  if (std::memcmp(magic, "T1G1", 4) == 0) {
    return run_volume_convt1_fwd(bench_iters);
  }
  if (std::memcmp(magic, "T2G1", 4) == 0) {
    return run_volume_convt2_fwd(bench_iters);
  }
  if (std::memcmp(magic, "T3G1", 4) == 0) {
    return run_volume_convt3_fwd(bench_iters);
  }
  if (std::memcmp(magic, "WEG1", 4) == 0) {
    return run_volume_embedding_fwd(bench_iters);
  }
  if (std::memcmp(magic, "MBG1", 4) == 0) {
    return run_volume_mamba_fwd(bench_iters);
  }
  if (std::memcmp(magic, "GDG1", 4) == 0) {
    return run_volume_gdn_fwd(bench_iters);
  }
  if (std::memcmp(magic, "MCG1", 4) == 0) {
    return run_volume_metacognition_fwd(bench_iters);
  }
  const bool v3 = std::memcmp(magic, "WVD3", 4) == 0;
  const bool v2 = std::memcmp(magic, "WVD2", 4) == 0;
  const bool v1 = std::memcmp(magic, "WVD1", 4) == 0;
  if (!v1 && !v2 && !v3) {
    std::fprintf(stderr, "welvet_dense_fwd: bad magic %.4s\n", magic);
    return 2;
  }
  return run_dense(v1, v2, v3, bench_iters);
}
