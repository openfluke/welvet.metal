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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

static void usage() {
  std::fprintf(stderr,
               "welvet_dense_fwd (welvet.metal)\n"
               "  --version | --identity | --gaps | --help\n"
               "  --bench=N\n"
               "  stdin WVD3→WVO1 | WVM1→WVMO | WVS1→WVSO | WVR1→WVRO\n");
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
  const bool v3 = std::memcmp(magic, "WVD3", 4) == 0;
  const bool v2 = std::memcmp(magic, "WVD2", 4) == 0;
  const bool v1 = std::memcmp(magic, "WVD1", 4) == 0;
  if (!v1 && !v2 && !v3) {
    std::fprintf(stderr, "welvet_dense_fwd: bad magic %.4s\n", magic);
    return 2;
  }
  return run_dense(v1, v2, v3, bench_iters);
}
