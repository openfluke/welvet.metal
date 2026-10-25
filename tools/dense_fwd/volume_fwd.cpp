// Volume forwards: bind every cell once, then volume_dense (WVB2 splits mount vs exec).

#include "volume_fwd.hpp"

#include "welvet/architecture.hpp"
#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/weights.hpp"
#include "welvet/kmeans.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/mha.hpp"
#include "welvet/parallel.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/softmax.hpp"
#include "welvet/swiglu.hpp"
#include "welvet/sequential.hpp"
#include "welvet/rnn.hpp"
#include "welvet/residual.hpp"
#include "welvet/metacognition.hpp"
#include "welvet/mamba.hpp"
#include "welvet/lstm.hpp"
#include "welvet/gdn.hpp"
#include "welvet/embedding.hpp"
#include "welvet/convt3.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt1.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn1.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace {

bool read_exact(void* p, size_t n) { return std::fread(p, 1, n, stdin) == n; }
template <typename T>
bool read_exact(T* p, size_t n) {
  return read_exact(static_cast<void*>(p), n * sizeof(T));
}
bool write_exact(const void* p, size_t n) { return std::fwrite(p, 1, n, stdout) == n; }
template <typename T>
bool write_exact(const T* p, size_t n) {
  return write_exact(static_cast<const void*>(p), n * sizeof(T));
}

int write_vf_out(const std::vector<float>& out) {
  const int32_t n = static_cast<int32_t>(out.size());
  if (!write_exact("WVFO", 4) || !write_exact(&n, 1) || !write_exact(out.data(), out.size())) {
    return 2;
  }
  return 0;
}

int finish_volume(int bench_iters, int64_t mount_ns, const std::function<welvet::Status()>& once,
                  welvet::runtime::forward::Result& result) {
  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto te0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto te1 = std::chrono::steady_clock::now();
    const int64_t exec_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(te1 - te0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB2", 4) || !write_exact(&mount_ns, 1) || !write_exact(&exec_ns, 1) ||
        !write_exact(&iters32, 1)) {
      return 2;
    }
  } else {
    auto st = once();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  return write_vf_out(result.output.data);
}

bool read_grid_hdr(int32_t* depth, int32_t* rows, int32_t* cols, int32_t* lpc, int32_t* batch,
                   int32_t* backend, int32_t* n_links, const char* tag) {
  if (!read_exact(depth, 1) || !read_exact(rows, 1) || !read_exact(cols, 1) || !read_exact(lpc, 1) ||
      !read_exact(batch, 1) || !read_exact(backend, 1) || !read_exact(n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short %s header\n", tag);
    return false;
  }
  if (*batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return false;
  }
  return true;
}

bool apply_links(welvet::architecture::Grid& g, const std::vector<int32_t>& links, int32_t n_links) {
  for (int32_t i = 0; i < n_links; ++i) {
    const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
    auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5], L[6],
                                                    L[7]);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return false;
    }
  }
  return true;
}

} // namespace

int run_volume_dense_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WVF1")) return 2;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in = 0, out = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  cells.reserve(hops.size());
  int first_in = -1;
  for (const auto& coord : hops) {
    Cell c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in, 1) || !read_exact(&c.out, 1) || !read_exact(&c.act, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    c.W.resize(static_cast<std::size_t>(c.out * c.in));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len > 0 && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in < 0) first_in = c.in;
    cells.push_back(std::move(c));
  }
  if (first_in < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::dense::Layer layer;
    auto st = welvet::dense::new_layer(c.in, c.out, static_cast<welvet::Activation>(c.act),
                                       static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    (void)backend;
    if (!c.bias.empty()) layer.weights.bias = c.bias;
    st = welvet::architecture::bind_dense(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                          std::move(layer));
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();

  welvet::runtime::forward::Result result;
  return finish_volume(
      bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_mha_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WMV1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
    int mask = 0, pos = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> Q, K, V, O;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  cells.reserve(hops.size());
  int first_seq = -1, first_d = -1;
  for (const auto& coord : hops) {
    Cell c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.seq, 1) || !read_exact(&c.d_model, 1) || !read_exact(&c.num_heads, 1) ||
        !read_exact(&c.num_kv, 1) || !read_exact(&c.head_dim, 1) || !read_exact(&c.msl, 1) ||
        !read_exact(&c.mask, 1) || !read_exact(&c.pos, 1) || !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int hd = c.head_dim > 0 ? c.head_dim : c.d_model / c.num_heads;
    const int kv_heads = c.num_kv > 0 ? c.num_kv : c.num_heads;
    const int q_dim = c.num_heads * hd;
    const int kv_dim = kv_heads * hd;
    c.Q.resize(static_cast<std::size_t>(q_dim * c.d_model));
    c.K.resize(static_cast<std::size_t>(kv_dim * c.d_model));
    c.V.resize(static_cast<std::size_t>(kv_dim * c.d_model));
    c.O.resize(static_cast<std::size_t>(c.d_model * q_dim));
    if (!read_exact(c.Q.data(), c.Q.size()) || !read_exact(c.K.data(), c.K.size()) ||
        !read_exact(c.V.data(), c.V.size()) || !read_exact(c.O.data(), c.O.size())) {
      return 2;
    }
    if (first_seq < 0) {
      first_seq = c.seq;
      first_d = c.d_model;
    }
    cells.push_back(std::move(c));
  }
  if (first_seq < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_d};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_d));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::mha::Config cfg;
    cfg.d_model = c.d_model;
    cfg.num_heads = c.num_heads;
    cfg.num_kv_heads = c.num_kv;
    cfg.head_dim = c.head_dim;
    cfg.max_seq_len = c.msl;
    cfg.mask = static_cast<welvet::mha::MaskKind>(c.mask);
    cfg.pos = static_cast<welvet::mha::PosKind>(c.pos);
    cfg.mode = welvet::mha::AttnMode::Self;
    cfg.causal = (cfg.mask == welvet::mha::MaskKind::Causal);
    welvet::mha::Layer layer;
    auto st = welvet::mha::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.Q.data(),
                                          c.K.data(), c.V.data(), c.O.data(), &layer);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    st = welvet::architecture::bind_mha(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                        std::move(layer));
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();

  welvet::runtime::forward::Result result;
  return finish_volume(
      bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_rmsnorm_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WRF1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int dim = 0, dtype = 1;
    double eps = 0.0;
    welvet::architecture::Coord coord{};
    std::vector<float> gamma;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_dim = -1;
  for (const auto& coord : hops) {
    Cell c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.dtype, 1) || !read_exact(&c.eps, 1)) return 2;
    c.gamma.resize(static_cast<std::size_t>(c.dim));
    if (!read_exact(c.gamma.data(), c.gamma.size())) return 2;
    if (first_dim < 0) first_dim = c.dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::rmsnorm::Config cfg;
    cfg.dim = c.dim;
    cfg.eps = c.eps;
    welvet::rmsnorm::Layer layer;
    auto st = welvet::rmsnorm::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                              c.gamma.data(), &layer);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    st = welvet::architecture::bind_rmsnorm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                            std::move(layer));
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();

  welvet::runtime::forward::Result result;
  return finish_volume(
      bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_layernorm_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WLF1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int dim = 0, dtype = 1;
    double eps = 0.0;
    welvet::architecture::Coord coord{};
    std::vector<float> gamma, beta;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_dim = -1;
  for (const auto& coord : hops) {
    Cell c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.dtype, 1) || !read_exact(&c.eps, 1)) return 2;
    c.gamma.resize(static_cast<std::size_t>(c.dim));
    c.beta.resize(static_cast<std::size_t>(c.dim));
    if (!read_exact(c.gamma.data(), c.gamma.size()) || !read_exact(c.beta.data(), c.beta.size())) {
      return 2;
    }
    if (first_dim < 0) first_dim = c.dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::layernorm::Config cfg;
    cfg.dim = c.dim;
    cfg.eps = c.eps;
    welvet::layernorm::Layer layer;
    auto st = welvet::layernorm::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                c.gamma.data(), c.beta.data(), &layer);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    st = welvet::architecture::bind_layernorm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                              std::move(layer));
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();

  welvet::runtime::forward::Result result;
  return finish_volume(
      bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_softmax_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WSF1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int dim = 0, dtype = 1;
    welvet::architecture::Coord coord{};
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_dim = -1;
  for (const auto& coord : hops) {
    Cell c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.dtype, 1)) return 2;
    if (first_dim < 0) first_dim = c.dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::softmax::Config cfg;
    cfg.dim = c.dim;
    cfg.seq_len = 1;
    cfg.kind = welvet::softmax::Kind::Standard;
    welvet::softmax::Layer layer;
    auto st = welvet::softmax::new_configured(cfg, static_cast<welvet::DType>(c.dtype), &layer);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    st = welvet::architecture::bind_softmax(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                            std::move(layer));
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();

  welvet::runtime::forward::Result result;
  return finish_volume(
      bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_parallel_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WPF1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int dim = 0, out_feat = 0, branches = 0, dtype = 1, combine = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> packed;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_dim = -1;
  for (const auto& coord : hops) {
    Cell c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.out_feat, 1) || !read_exact(&c.branches, 1) ||
        !read_exact(&c.dtype, 1) || !read_exact(&c.combine, 1)) {
      return 2;
    }
    const int nw = c.branches * c.out_feat * c.dim;
    c.packed.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.packed.data(), c.packed.size())) return 2;
    if (first_dim < 0) first_dim = c.dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::parallel::Config cfg;
    cfg.dim = c.dim;
    cfg.out_feat = c.out_feat;
    cfg.branches = c.branches;
    cfg.combine = static_cast<welvet::parallel::Combine>(c.combine);
    welvet::parallel::Layer layer;
    auto st = welvet::parallel::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                               c.packed.data(), &layer);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    st = welvet::architecture::bind_parallel(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                             std::move(layer));
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();

  welvet::runtime::forward::Result result;
  return finish_volume(
      bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_kmeans_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WKF1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int k = 0, d = 0, seq = 0, output_mode = 0, act = -1, dtype = 1;
    double temperature = 1.0;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_d = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.k, 1) || !read_exact(&c.d, 1) || !read_exact(&c.seq, 1) ||
        !read_exact(&c.output_mode, 1) || !read_exact(&c.act, 1) || !read_exact(&c.temperature, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    c.W.resize(static_cast<std::size_t>(c.k * c.d));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_d < 0) {
      first_d = c.d;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_d < 0) return 1;

  welvet::TensorF32 input;
  if (first_seq == 1) {
    input.shape = {batch, first_d};
  } else {
    input.shape = {batch, first_seq, first_d};
  }
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_d));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::kmeans::Config cfg;
    cfg.num_clusters = c.k;
    cfg.feature_dim = c.d;
    cfg.temperature = c.temperature;
    cfg.output_mode = static_cast<welvet::kmeans::OutputMode>(c.output_mode);
    cfg.activation = static_cast<welvet::Activation>(c.act);
    welvet::kmeans::Layer layer;
    auto st =
        welvet::kmeans::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    st = welvet::architecture::bind_kmeans(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                           std::move(layer));
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();

  welvet::runtime::forward::Result result;
  return finish_volume(
      bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

// WSG1 — Volume SwiGLU forward (cell layout = WSV1 minus target/lr).
int run_volume_swiglu_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WSG1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in = 0, inter = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> Gate, Up, Down;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_in = -1;
  for (const auto& coord : hops) {
    Cell c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in, 1) || !read_exact(&c.inter, 1) || !read_exact(&c.dtype, 1)) return 2;
    c.Gate.resize(static_cast<std::size_t>(c.inter * c.in));
    c.Up.resize(static_cast<std::size_t>(c.inter * c.in));
    c.Down.resize(static_cast<std::size_t>(c.in * c.inter));
    if (!read_exact(c.Gate.data(), c.Gate.size()) || !read_exact(c.Up.data(), c.Up.size()) ||
        !read_exact(c.Down.data(), c.Down.size())) {
      return 2;
    }
    if (first_in < 0) first_in = c.in;
    cells.push_back(std::move(c));
  }
  if (first_in < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::swiglu::Config cfg;
    cfg.input_dim = c.in;
    cfg.intermediate_dim = c.inter;
    welvet::swiglu::Layer layer;
    auto st = welvet::swiglu::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                             c.Gate.data(), c.Up.data(), c.Down.data(), &layer);
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    st = welvet::architecture::bind_swiglu(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                           std::move(layer));
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();

  welvet::runtime::forward::Result result;
  return finish_volume(
      bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_sequential_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "SQG1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int dim = 0, seq = 0, chain = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  cells.reserve(hops.size());
  int first_dim = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.chain, 1) || !read_exact(&c.dtype, 1)) return 2;
    const int nw = c.chain * c.dim * c.dim;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_dim < 0) { first_dim = c.dim; first_seq = c.seq; }
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  if (first_seq == 1) input.shape = {batch, first_dim};
  else input.shape = {batch, first_seq, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
      welvet::sequential::Config cfg;
      cfg.dim = c.dim; cfg.seq_len = c.seq; cfg.depth = c.chain;
      welvet::sequential::Layer layer;
      auto st = welvet::sequential::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                   welvet::Activation::Linear, c.W.data(), &layer);
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
      st = welvet::architecture::bind_sequential(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_residual_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "REG1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int dim = 0, seq = 0, chain = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  cells.reserve(hops.size());
  int first_dim = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.chain, 1) || !read_exact(&c.dtype, 1)) return 2;
    const int nw = c.chain * c.dim * c.dim;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_dim < 0) { first_dim = c.dim; first_seq = c.seq; }
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  if (first_seq == 1) input.shape = {batch, first_dim};
  else input.shape = {batch, first_seq, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
      welvet::residual::Config cfg;
      cfg.dim = c.dim; cfg.seq_len = c.seq; cfg.depth = c.chain;
      welvet::residual::Layer layer;
      auto st = welvet::residual::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                 welvet::Activation::Linear, c.W.data(), &layer);
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
      st = welvet::architecture::bind_residual(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_rnn_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "RNG1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in = 0, hid = 0, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  cells.reserve(hops.size());
  int first_in = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.in, 1) || !read_exact(&c.hid, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.dtype, 1)) return 2;
    const int nw = c.hid * c.in + c.hid * c.hid + c.hid;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in < 0) { first_in = c.in; first_seq = c.seq; }
    cells.push_back(std::move(c));
  }
  if (first_in < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
      welvet::rnn::Config cfg;
      cfg.input_size = c.in; cfg.hidden_size = c.hid; cfg.seq_len = c.seq;
      welvet::rnn::Layer layer;
      auto st = welvet::rnn::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
      st = welvet::architecture::bind_rnn(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_lstm_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "LSG1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in = 0, hid = 0, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  cells.reserve(hops.size());
  int first_in = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.in, 1) || !read_exact(&c.hid, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.dtype, 1)) return 2;
    const int nw = 4 * (c.hid * c.in + c.hid * c.hid + c.hid);
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in < 0) { first_in = c.in; first_seq = c.seq; }
    cells.push_back(std::move(c));
  }
  if (first_in < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
      welvet::lstm::Config cfg;
      cfg.input_size = c.in; cfg.hidden_size = c.hid; cfg.seq_len = c.seq;
      welvet::lstm::Layer layer;
      auto st = welvet::lstm::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
      st = welvet::architecture::bind_lstm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_metacognition_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "MCG1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int dim = 0, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  cells.reserve(hops.size());
  int first_dim = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.dtype, 1)) return 2;
    const int nw = c.dim * c.dim;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_dim < 0) { first_dim = c.dim; first_seq = c.seq; }
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  if (first_seq == 1) input.shape = {batch, first_dim};
  else input.shape = {batch, first_seq, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
      welvet::metacognition::Config cfg;
      cfg.dim = c.dim; cfg.seq_len = c.seq == 1 ? 0 : c.seq;
      welvet::metacognition::Layer layer;
      auto st = welvet::metacognition::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
      st = welvet::architecture::bind_metacognition(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_mamba_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "MBG1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int d_model = 0, d_state = 0, expand = 0, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  cells.reserve(hops.size());
  int first_d = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.d_model, 1) || !read_exact(&c.d_state, 1) || !read_exact(&c.expand, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.dtype, 1)) return 2;
    const int nw = ([&]{ int exp = c.expand <= 0 ? 2 : c.expand; int inner = exp * c.d_model; return inner * (3 * c.d_model + 2); })();
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_d < 0) { first_d = c.d_model; first_seq = c.seq; }
    cells.push_back(std::move(c));
  }
  if (first_d < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_d};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_d));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
      welvet::mamba::Config cfg;
      cfg.d_model = c.d_model; cfg.d_state = c.d_state; cfg.expand = c.expand; cfg.seq_len = c.seq;
      welvet::mamba::Layer layer;
      auto st = welvet::mamba::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
      st = welvet::architecture::bind_mamba(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
      if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_embedding_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WEG1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int vocab = 0, emb = 0, seq = 0, act = -1, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.vocab, 1) || !read_exact(&c.emb, 1) || !read_exact(&c.seq, 1) ||
        !read_exact(&c.act, 1) || !read_exact(&c.dtype, 1)) return 2;
    c.W.resize(static_cast<std::size_t>(c.vocab * c.emb));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_seq < 0) first_seq = c.seq;
    cells.push_back(std::move(c));
  }
  if (first_seq < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq};
  input.data.resize(static_cast<std::size_t>(batch * first_seq));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::embedding::Config cfg;
    cfg.vocab_size = c.vocab; cfg.embedding_dim = c.emb; cfg.seq_len = c.seq;
    welvet::embedding::Layer layer;
    auto st = welvet::embedding::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                static_cast<welvet::Activation>(c.act), c.W.data(), &layer);
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
    st = welvet::architecture::bind_embedding(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_gdn_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "GDG1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int hidden = 0, nk = 0, nv = 0, hd_k = 0, hd_v = 0, conv_k = 4, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_h = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.hidden, 1) || !read_exact(&c.nk, 1) || !read_exact(&c.nv, 1) ||
        !read_exact(&c.hd_k, 1) || !read_exact(&c.hd_v, 1) || !read_exact(&c.conv_k, 1) ||
        !read_exact(&c.seq, 1) || !read_exact(&c.dtype, 1)) return 2;
    welvet::gdn::Config cfg;
    cfg.hidden_size = c.hidden; cfg.num_key_heads = c.nk; cfg.num_value_heads = c.nv;
    cfg.key_head_dim = c.hd_k; cfg.value_head_dim = c.hd_v; cfg.conv_kernel = c.conv_k;
    if (auto st = welvet::gdn::validate(&cfg); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1;
    }
    c.W.resize(static_cast<std::size_t>(welvet::gdn::weight_count(cfg)));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_h < 0) { first_h = c.hidden; first_seq = c.seq; }
    cells.push_back(std::move(c));
  }
  if (first_h < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_h};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_h));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::gdn::Config cfg;
    cfg.hidden_size = c.hidden; cfg.num_key_heads = c.nk; cfg.num_value_heads = c.nv;
    cfg.key_head_dim = c.hd_k; cfg.value_head_dim = c.hd_v; cfg.conv_kernel = c.conv_k;
    welvet::gdn::Layer layer;
    auto st = welvet::gdn::new_configured(cfg, c.W.data(), &layer);
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
    st = welvet::architecture::bind_gdn(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_cnn1_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "C1G1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_in_c = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.seq, 1) ||
        !read_exact(&c.kernel, 1) || !read_exact(&c.stride, 1) || !read_exact(&c.padding, 1) ||
        !read_exact(&c.act, 1) || !read_exact(&c.dtype, 1)) return 2;
    c.W.resize(static_cast<std::size_t>(c.filters * c.in_c * c.kernel));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in_c < 0) { first_in_c = c.in_c; first_seq = c.seq; }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_seq};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_seq));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::cnn1::Config cfg;
    cfg.in_channels = c.in_c; cfg.filters = c.filters; cfg.seq_len = c.seq;
    cfg.kernel = c.kernel; cfg.stride = c.stride; cfg.padding = c.padding;
    cfg.activation = static_cast<welvet::Activation>(c.act);
    welvet::cnn1::Layer layer;
    auto st = welvet::cnn1::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
    if (!c.bias.empty()) layer.proj.weights.bias = c.bias;
    st = welvet::architecture::bind_cnn1(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_cnn2_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "C2G1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0, padding = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_in_c = -1, first_h = -1, first_w = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.height, 1) ||
        !read_exact(&c.width, 1) || !read_exact(&c.kernel, 1) || !read_exact(&c.stride, 1) ||
        !read_exact(&c.padding, 1) || !read_exact(&c.act, 1) || !read_exact(&c.dtype, 1)) return 2;
    c.W.resize(static_cast<std::size_t>(c.filters * c.in_c * c.kernel * c.kernel));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in_c < 0) { first_in_c = c.in_c; first_h = c.height; first_w = c.width; }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_h, first_w};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_h * first_w));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::cnn2::Config cfg;
    cfg.in_channels = c.in_c; cfg.filters = c.filters; cfg.height = c.height; cfg.width = c.width;
    cfg.kernel = c.kernel; cfg.stride = c.stride; cfg.padding = c.padding;
    cfg.activation = static_cast<welvet::Activation>(c.act);
    welvet::cnn2::Layer layer;
    auto st = welvet::cnn2::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
    if (!c.bias.empty()) layer.proj.weights.bias = c.bias;
    st = welvet::architecture::bind_cnn2(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_cnn3_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "C3G1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in_c = 0, filters = 0, d = 0, height = 0, width = 0, kernel = 0, stride = 0, padding = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_in_c = -1, first_d = -1, first_h = -1, first_w = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.d, 1) ||
        !read_exact(&c.height, 1) || !read_exact(&c.width, 1) || !read_exact(&c.kernel, 1) ||
        !read_exact(&c.stride, 1) || !read_exact(&c.padding, 1) || !read_exact(&c.act, 1) ||
        !read_exact(&c.dtype, 1)) return 2;
    c.W.resize(static_cast<std::size_t>(c.filters * c.in_c * c.kernel * c.kernel * c.kernel));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in_c < 0) { first_in_c = c.in_c; first_d = c.d; first_h = c.height; first_w = c.width; }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_d, first_h, first_w};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_d * first_h * first_w));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::cnn3::Config cfg;
    cfg.in_channels = c.in_c; cfg.filters = c.filters; cfg.depth = c.d;
    cfg.height = c.height; cfg.width = c.width; cfg.kernel = c.kernel;
    cfg.stride = c.stride; cfg.padding = c.padding;
    cfg.activation = static_cast<welvet::Activation>(c.act);
    welvet::cnn3::Layer layer;
    auto st = welvet::cnn3::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
    if (!c.bias.empty()) layer.proj.weights.bias = c.bias;
    st = welvet::architecture::bind_cnn3(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_convt1_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "T1G1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0, out_pad = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_in_c = -1, first_seq = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.seq, 1) ||
        !read_exact(&c.kernel, 1) || !read_exact(&c.stride, 1) || !read_exact(&c.padding, 1) ||
        !read_exact(&c.out_pad, 1) || !read_exact(&c.act, 1) || !read_exact(&c.dtype, 1)) return 2;
    c.W.resize(static_cast<std::size_t>(c.filters * c.in_c * c.kernel));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in_c < 0) { first_in_c = c.in_c; first_seq = c.seq; }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_seq};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_seq));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::convt1::Config cfg;
    cfg.in_channels = c.in_c; cfg.filters = c.filters; cfg.seq_len = c.seq;
    cfg.kernel = c.kernel; cfg.stride = c.stride; cfg.padding = c.padding; cfg.output_padding = c.out_pad;
    cfg.activation = static_cast<welvet::Activation>(c.act);
    welvet::convt1::Layer layer;
    auto st = welvet::convt1::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
    st = welvet::architecture::bind_convt1(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_convt2_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "T2G1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0, padding = 0, out_pad = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_in_c = -1, first_h = -1, first_w = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.height, 1) ||
        !read_exact(&c.width, 1) || !read_exact(&c.kernel, 1) || !read_exact(&c.stride, 1) ||
        !read_exact(&c.padding, 1) || !read_exact(&c.out_pad, 1) || !read_exact(&c.act, 1) ||
        !read_exact(&c.dtype, 1)) return 2;
    c.W.resize(static_cast<std::size_t>(c.filters * c.in_c * c.kernel * c.kernel));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in_c < 0) { first_in_c = c.in_c; first_h = c.height; first_w = c.width; }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_h, first_w};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_h * first_w));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::convt2::Config cfg;
    cfg.in_channels = c.in_c; cfg.filters = c.filters; cfg.height = c.height; cfg.width = c.width;
    cfg.kernel = c.kernel; cfg.stride = c.stride; cfg.padding = c.padding; cfg.output_padding = c.out_pad;
    cfg.activation = static_cast<welvet::Activation>(c.act);
    welvet::convt2::Layer layer;
    auto st = welvet::convt2::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
    st = welvet::architecture::bind_convt2(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

int run_volume_convt3_fwd(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "T3G1")) return 2;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct Cell {
    bool enabled = false;
    int in_c = 0, filters = 0, d = 0, height = 0, width = 0, kernel = 0, stride = 0, padding = 0, out_pad = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<Cell> cells;
  int first_in_c = -1, first_d = -1, first_h = -1, first_w = -1;
  for (const auto& coord : hops) {
    Cell c; c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) { cells.push_back(std::move(c)); continue; }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.d, 1) ||
        !read_exact(&c.height, 1) || !read_exact(&c.width, 1) || !read_exact(&c.kernel, 1) ||
        !read_exact(&c.stride, 1) || !read_exact(&c.padding, 1) || !read_exact(&c.out_pad, 1) ||
        !read_exact(&c.act, 1) || !read_exact(&c.dtype, 1)) return 2;
    c.W.resize(static_cast<std::size_t>(c.filters * c.in_c * c.kernel * c.kernel * c.kernel));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in_c < 0) { first_in_c = c.in_c; first_d = c.d; first_h = c.height; first_w = c.width; }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_d, first_h, first_w};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_d * first_h * first_w));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  const auto tm0 = std::chrono::steady_clock::now();
  auto g = welvet::architecture::new_grid(depth, rows, cols, lpc);
  if (!apply_links(g, links, n_links)) return 1;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    welvet::convt3::Config cfg;
    cfg.in_channels = c.in_c; cfg.filters = c.filters; cfg.depth = c.d;
    cfg.height = c.height; cfg.width = c.width; cfg.kernel = c.kernel;
    cfg.stride = c.stride; cfg.padding = c.padding; cfg.output_padding = c.out_pad;
    cfg.activation = static_cast<welvet::Activation>(c.act);
    welvet::convt3::Layer layer;
    auto st = welvet::convt3::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
    st = welvet::architecture::bind_convt3(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l, std::move(layer));
    if (!st) { std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str()); return 1; }
  }
  const auto tm1 = std::chrono::steady_clock::now();
  const int64_t mount_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tm1 - tm0).count();
  welvet::runtime::forward::Result result;
  return finish_volume(bench_iters, mount_ns,
      [&]() { return welvet::runtime::forward::volume_dense(g, input, &result); }, result);
}

#include "welvet/runtime/training.hpp"
#include "welvet_c.h"

// WDB1 — Dense NormalBP leaf
// stdin: batch,in,out,act,dtype,backend ; W ; bias_len,bias ; x ; use_target ; grad_or_target ; lr
// stdout WDBO: pre_n,pre ; post_n,post ; dx_n,dX ; dw_n,dW ; w_n,W_after ; loss
// --bench=N → WVB1 fused ns (mount once; reset W in-place; export once after timed loop)
int run_dense_normalbp(int bench_iters) {
  int32_t batch = 0, in = 0, out = 0, act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in, 1) || !read_exact(&out, 1) || !read_exact(&act, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WDB1 header\n");
    return 2;
  }
  (void)backend;
  std::vector<float> W(static_cast<std::size_t>(out * in));
  if (!read_exact(W.data(), W.size())) return 2;
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len > 0 && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * out));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(static_cast<std::size_t>(batch * out));
  std::vector<float> post(static_cast<std::size_t>(batch * out));
  std::vector<float> dX(static_cast<std::size_t>(batch * in));
  std::vector<float> dW(static_cast<std::size_t>(out * in));
  std::vector<float> W_after(static_cast<std::size_t>(out * in));
  double loss = 0.0;

  welvet::dense::Layer layer;
  auto st0 = welvet::dense::new_layer(in, out, static_cast<welvet::Activation>(act),
                                      static_cast<welvet::DType>(dtype), W.data(), &layer);
  if (!st0) {
    std::fprintf(stderr, "welvet_dense_fwd: %s\n", st0.message.c_str());
    return 1;
  }
  if (bias_len > 0) {
    layer.weights.bias.assign(bias.begin(), bias.end());
  }

  welvet::TensorF32 input;
  input.shape = {batch, in};
  input.data.assign(x.begin(), x.end());

  auto reset_w = [&]() -> welvet::Status {
    auto st = welvet::WeightStore::make(out, in, static_cast<welvet::DType>(dtype), W.data(),
                                        &layer.weights);
    if (!st) return st;
    if (bias_len > 0) {
      layer.weights.bias.assign(bias.begin(), bias.end());
    } else {
      layer.weights.bias.clear();
    }
    layer.invalidate_gpu();
    return welvet::Status::Ok();
  };

  auto step = [&](bool export_out) -> welvet::Status {
    if (auto st = reset_w(); !st) return st;
    welvet::TensorF32 pre_t, post_t;
    if (auto st = welvet::dense::forward_f32(layer, input, &pre_t, &post_t); !st) return st;
    welvet::TensorF32 gy;
    gy.shape = {batch, out};
    gy.data.assign(gyt.begin(), gyt.end());
    loss = 0.0;
    if (use_target) {
      if (auto st = welvet::runtime::training::mse(post_t, gy, &loss); !st) return st;
      welvet::TensorF32 gtmp;
      if (auto st = welvet::runtime::training::mse_grad(post_t, gy, &gtmp); !st) return st;
      gy = std::move(gtmp);
    }
    welvet::TensorF32 dx_t, dw_t;
    if (auto st = welvet::dense::backward_f32(layer, gy, input, pre_t, &dx_t, &dw_t); !st) {
      return st;
    }
    if (lr != 0.0) {
      if (auto st = welvet::dense::apply_grad_sgd(&layer, dw_t, lr); !st) return st;
    }
    if (export_out) {
      if (auto st = layer.weights.export_f32(W_after.data()); !st) return st;
      std::memcpy(pre.data(), pre_t.data.data(), pre_t.data.size() * sizeof(float));
      std::memcpy(post.data(), post_t.data.data(), post_t.data.size() * sizeof(float));
      std::memcpy(dX.data(), dx_t.data.data(), dx_t.data.size() * sizeof(float));
      std::memcpy(dW.data(), dw_t.data.data(), dw_t.data.size() * sizeof(float));
    }
    return welvet::Status::Ok();
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = step(false);
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = step(false);
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
    // One export after timed loop for wire payload (exact_bit vs Go).
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  } else {
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_exact("WDBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WVN1 — Volume Dense NormalBP (same cell layout as WVF1 + target + lr)
int run_volume_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WVN1")) return 2;
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
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    if (!apply_links(g, links, n_links)) {
      return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "WVN1: bad links");
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::dense::Layer layer;
      auto st = welvet::dense::new_layer(c.in, c.out, static_cast<welvet::Activation>(c.act),
                                         static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
      if (!st) return st;
      if (!c.bias.empty()) layer.weights.bias = c.bias;
      st = welvet::architecture::bind_dense(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                            std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
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

  if (!write_exact("WVNO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
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
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WRB1 — RMSNorm NormalBP leaf
int run_rmsnorm_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) || !read_exact(&eps, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WRB1 header\n");
    return 2;
  }
  std::vector<float> gamma(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                       static_cast<std::size_t>(dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  (void)backend;
  std::vector<float> pre(x.size()), post(x.size()), dX(x.size()), dW(static_cast<std::size_t>(dim)),
      gamma_after(static_cast<std::size_t>(dim));
  double loss = 0.0;

  welvet::rmsnorm::Config cfg;
  cfg.dim = dim;
  cfg.eps = eps;
  welvet::rmsnorm::Layer layer;
  auto st0 = welvet::rmsnorm::new_configured(cfg, static_cast<welvet::DType>(dtype), gamma.data(),
                                             &layer);
  if (!st0) {
    std::fprintf(stderr, "welvet_dense_fwd: %s\n", st0.message.c_str());
    return 1;
  }
  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, dim};
  } else {
    input.shape = {batch, seq, dim};
  }
  input.data.assign(x.begin(), x.end());

  auto reset_w = [&]() -> welvet::Status {
    return welvet::WeightStore::make(1, dim, static_cast<welvet::DType>(dtype), gamma.data(),
                                     &layer.gamma);
  };

  auto step = [&](bool export_out) -> welvet::Status {
    if (auto st = reset_w(); !st) return st;
    welvet::TensorF32 pre_t, post_t;
    if (auto st = welvet::rmsnorm::forward(layer, input, &pre_t, &post_t); !st) return st;
    welvet::TensorF32 gy;
    gy.shape = post_t.shape;
    gy.data.assign(gyt.begin(), gyt.end());
    loss = 0.0;
    if (use_target) {
      if (auto st = welvet::runtime::training::mse(post_t, gy, &loss); !st) return st;
      welvet::TensorF32 gtmp;
      if (auto st = welvet::runtime::training::mse_grad(post_t, gy, &gtmp); !st) return st;
      gy = std::move(gtmp);
    }
    welvet::TensorF32 dx_t, dw_t;
    if (auto st = welvet::rmsnorm::backward_f32(layer, gy, input, pre_t, &dx_t, &dw_t); !st) {
      return st;
    }
    if (lr != 0.0) {
      if (auto st = welvet::rmsnorm::apply_grad_sgd(&layer, dw_t, lr); !st) return st;
    }
    if (export_out) {
      if (auto st = layer.gamma.export_f32(gamma_after.data()); !st) return st;
      std::memcpy(pre.data(), pre_t.data.data(), pre_t.data.size() * sizeof(float));
      std::memcpy(post.data(), post_t.data.data(), post_t.data.size() * sizeof(float));
      std::memcpy(dX.data(), dx_t.data.data(), dx_t.data.size() * sizeof(float));
      std::memcpy(dW.data(), dw_t.data.data(), dw_t.data.size() * sizeof(float));
    }
    return welvet::Status::Ok();
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = step(false);
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = step(false);
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
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  } else {
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t g_n = static_cast<int32_t>(gamma_after.size());
  if (!write_exact("WRBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&g_n, 1) ||
      !write_exact(gamma_after.data(), gamma_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WRV1 — Volume RMSNorm NormalBP
int run_volume_rmsnorm_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WRV1")) return 2;
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0;
    int dtype = 1;
    double eps = 0.0;
    welvet::architecture::Coord coord{};
    std::vector<float> gamma;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
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
    int32_t dim = 0, dtype = 1;
    double eps = 0.0;
    if (!read_exact(&dim, 1) || !read_exact(&dtype, 1) || !read_exact(&eps, 1)) return 2;
    c.dim = dim;
    c.dtype = dtype;
    c.eps = eps;
    c.gamma.resize(static_cast<std::size_t>(dim));
    if (!read_exact(c.gamma.data(), c.gamma.size())) return 2;
    if (first_dim < 0) first_dim = dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_dim = it->dim;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    if (!apply_links(g, links, n_links)) {
      return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "WRV1: bad links");
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::rmsnorm::Config cfg;
      cfg.dim = c.dim;
      cfg.eps = c.eps;
      welvet::rmsnorm::Layer layer;
      auto st = welvet::rmsnorm::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                c.gamma.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_rmsnorm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                              std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
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

  if (!write_exact("WRVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t d = c.dim;
    std::vector<float> ga(static_cast<std::size_t>(d));
    auto est = cell->rmsnorm_op.gamma.export_f32(ga.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&d, 1) || !write_exact(ga.data(), ga.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WLB1 — LayerNorm NormalBP leaf
int run_layernorm_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) || !read_exact(&eps, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WLB1 header\n");
    return 2;
  }
  std::vector<float> gamma(static_cast<std::size_t>(dim));
  std::vector<float> beta(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size()) || !read_exact(beta.data(), beta.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                       static_cast<std::size_t>(dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  (void)backend;
  std::vector<float> pre(x.size()), post(x.size()), dX(x.size()),
      dW(static_cast<std::size_t>(2 * dim)), gamma_after(static_cast<std::size_t>(dim)),
      beta_after(static_cast<std::size_t>(dim));
  double loss = 0.0;

  welvet::layernorm::Config cfg;
  cfg.dim = dim;
  cfg.eps = eps;
  welvet::layernorm::Layer layer;
  auto st0 = welvet::layernorm::new_configured(cfg, static_cast<welvet::DType>(dtype), gamma.data(),
                                               beta.data(), &layer);
  if (!st0) {
    std::fprintf(stderr, "welvet_dense_fwd: %s\n", st0.message.c_str());
    return 1;
  }
  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, dim};
  } else {
    input.shape = {batch, seq, dim};
  }
  input.data.assign(x.begin(), x.end());

  auto reset_w = [&]() -> welvet::Status {
    auto dt = static_cast<welvet::DType>(dtype);
    if (auto st = welvet::WeightStore::make(1, dim, dt, gamma.data(), &layer.gamma); !st) return st;
    return welvet::WeightStore::make(1, dim, dt, beta.data(), &layer.beta);
  };

  auto step = [&](bool export_out) -> welvet::Status {
    if (auto st = reset_w(); !st) return st;
    welvet::TensorF32 pre_t, post_t;
    if (auto st = welvet::layernorm::forward(layer, input, &pre_t, &post_t); !st) return st;
    welvet::TensorF32 gy;
    gy.shape = post_t.shape;
    gy.data.assign(gyt.begin(), gyt.end());
    loss = 0.0;
    if (use_target) {
      if (auto st = welvet::runtime::training::mse(post_t, gy, &loss); !st) return st;
      welvet::TensorF32 gtmp;
      if (auto st = welvet::runtime::training::mse_grad(post_t, gy, &gtmp); !st) return st;
      gy = std::move(gtmp);
    }
    welvet::TensorF32 dx_t, dw_t;
    if (auto st = welvet::layernorm::backward_f32(layer, gy, input, pre_t, &dx_t, &dw_t); !st) {
      return st;
    }
    if (lr != 0.0) {
      if (auto st = welvet::layernorm::apply_grad_sgd(&layer, dw_t, lr); !st) return st;
    }
    if (export_out) {
      if (auto st = layer.gamma.export_f32(gamma_after.data()); !st) return st;
      if (auto st = layer.beta.export_f32(beta_after.data()); !st) return st;
      std::memcpy(pre.data(), pre_t.data.data(), pre_t.data.size() * sizeof(float));
      std::memcpy(post.data(), post_t.data.data(), post_t.data.size() * sizeof(float));
      std::memcpy(dX.data(), dx_t.data.data(), dx_t.data.size() * sizeof(float));
      std::memcpy(dW.data(), dw_t.data.data(), dw_t.data.size() * sizeof(float));
    }
    return welvet::Status::Ok();
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = step(false);
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = step(false);
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
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  } else {
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t g_n = static_cast<int32_t>(gamma_after.size());
  const int32_t b_n = static_cast<int32_t>(beta_after.size());
  if (!write_exact("WLBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&g_n, 1) ||
      !write_exact(gamma_after.data(), gamma_after.size()) || !write_exact(&b_n, 1) ||
      !write_exact(beta_after.data(), beta_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WLV1 — Volume LayerNorm NormalBP
int run_volume_layernorm_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WLV1")) return 2;
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0;
    int dtype = 1;
    double eps = 0.0;
    welvet::architecture::Coord coord{};
    std::vector<float> gamma;
    std::vector<float> beta;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
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
    int32_t dim = 0, dtype = 1;
    double eps = 0.0;
    if (!read_exact(&dim, 1) || !read_exact(&dtype, 1) || !read_exact(&eps, 1)) return 2;
    c.dim = dim;
    c.dtype = dtype;
    c.eps = eps;
    c.gamma.resize(static_cast<std::size_t>(dim));
    c.beta.resize(static_cast<std::size_t>(dim));
    if (!read_exact(c.gamma.data(), c.gamma.size()) || !read_exact(c.beta.data(), c.beta.size())) {
      return 2;
    }
    if (first_dim < 0) first_dim = dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_dim = it->dim;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    if (!apply_links(g, links, n_links)) {
      return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "WLV1: bad links");
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::layernorm::Config cfg;
      cfg.dim = c.dim;
      cfg.eps = c.eps;
      welvet::layernorm::Layer layer;
      auto st = welvet::layernorm::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                  c.gamma.data(), c.beta.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_layernorm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                                std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
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

  if (!write_exact("WLVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t d = c.dim;
    std::vector<float> ga(static_cast<std::size_t>(d));
    std::vector<float> ba(static_cast<std::size_t>(d));
    auto eg = cell->layernorm_op.gamma.export_f32(ga.data());
    if (!eg) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", eg.message.c_str());
      return 1;
    }
    auto eb = cell->layernorm_op.beta.export_f32(ba.data());
    if (!eb) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", eb.message.c_str());
      return 1;
    }
    if (!write_exact(&d, 1) || !write_exact(ga.data(), ga.size()) ||
        !write_exact(ba.data(), ba.size())) {
      return 2;
    }
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WSB1 — SwiGLU NormalBP leaf
int run_swiglu_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, in = 0, inter = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&in, 1) || !read_exact(&inter, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WSB1 header\n");
    return 2;
  }
  std::vector<float> Gate(static_cast<std::size_t>(inter * in));
  std::vector<float> Up(static_cast<std::size_t>(inter * in));
  std::vector<float> Down(static_cast<std::size_t>(in * inter));
  if (!read_exact(Gate.data(), Gate.size()) || !read_exact(Up.data(), Up.size()) ||
      !read_exact(Down.data(), Down.size())) {
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                       static_cast<std::size_t>(in));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  (void)backend;
  const int nw = 2 * inter * in + in * inter;
  std::vector<float> pre(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                         static_cast<std::size_t>(inter));
  std::vector<float> post(x.size()), dX(x.size()), dW(static_cast<std::size_t>(nw)),
      W_after(static_cast<std::size_t>(nw));
  double loss = 0.0;

  welvet::swiglu::Config cfg;
  cfg.input_dim = in;
  cfg.intermediate_dim = inter;
  welvet::swiglu::Layer layer;
  auto st0 = welvet::swiglu::new_configured(cfg, static_cast<welvet::DType>(dtype), Gate.data(),
                                            Up.data(), Down.data(), &layer);
  if (!st0) {
    std::fprintf(stderr, "welvet_dense_fwd: %s\n", st0.message.c_str());
    return 1;
  }

  welvet::TensorF32 input;
  if (seq == 1) {
    input.shape = {batch, in};
  } else {
    input.shape = {batch, seq, in};
  }
  input.data.assign(x.begin(), x.end());

  auto reset_w = [&]() -> welvet::Status {
    auto st = welvet::WeightStore::make(inter, in, static_cast<welvet::DType>(dtype), Gate.data(),
                                        &layer.Gate.weights);
    if (!st) return st;
    st = welvet::WeightStore::make(inter, in, static_cast<welvet::DType>(dtype), Up.data(),
                                   &layer.Up.weights);
    if (!st) return st;
    st = welvet::WeightStore::make(in, inter, static_cast<welvet::DType>(dtype), Down.data(),
                                   &layer.Down.weights);
    if (!st) return st;
    layer.Gate.invalidate_gpu();
    layer.Up.invalidate_gpu();
    layer.Down.invalidate_gpu();
    return welvet::Status::Ok();
  };

  auto step = [&](bool export_out) -> welvet::Status {
    if (auto st = reset_w(); !st) return st;
    welvet::TensorF32 pre_t, post_t;
    if (auto st = welvet::swiglu::forward(layer, input, &pre_t, &post_t); !st) return st;
    welvet::TensorF32 gy;
    gy.shape = post_t.shape;
    gy.data.assign(gyt.begin(), gyt.end());
    loss = 0.0;
    if (use_target) {
      if (auto st = welvet::runtime::training::mse(post_t, gy, &loss); !st) return st;
      welvet::TensorF32 gtmp;
      if (auto st = welvet::runtime::training::mse_grad(post_t, gy, &gtmp); !st) return st;
      gy = std::move(gtmp);
    }
    welvet::TensorF32 dx_t, dw_t;
    if (auto st = welvet::swiglu::backward_f32(layer, gy, input, pre_t, &dx_t, &dw_t); !st) {
      return st;
    }
    if (lr != 0.0) {
      if (auto st = welvet::swiglu::apply_grad_sgd(&layer, dw_t, lr); !st) return st;
    }
    if (export_out) {
      if (auto st = welvet::swiglu::export_weights_f32(layer, W_after.data()); !st) return st;
      std::memcpy(pre.data(), pre_t.data.data(), pre_t.data.size() * sizeof(float));
      std::memcpy(post.data(), post_t.data.data(), post_t.data.size() * sizeof(float));
      std::memcpy(dX.data(), dx_t.data.data(), dx_t.data.size() * sizeof(float));
      std::memcpy(dW.data(), dw_t.data.data(), dw_t.data.size() * sizeof(float));
    }
    return welvet::Status::Ok();
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = step(false);
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = step(false);
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
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  } else {
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_exact("WSBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WSV1 — Volume SwiGLU NormalBP
int run_volume_swiglu_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WSV1")) return 2;
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in = 0, inter = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> Gate, Up, Down;
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
    int32_t in = 0, inter = 0, dtype = 1;
    if (!read_exact(&in, 1) || !read_exact(&inter, 1) || !read_exact(&dtype, 1)) return 2;
    c.in = in;
    c.inter = inter;
    c.dtype = dtype;
    c.Gate.resize(static_cast<std::size_t>(inter * in));
    c.Up.resize(static_cast<std::size_t>(inter * in));
    c.Down.resize(static_cast<std::size_t>(in * inter));
    if (!read_exact(c.Gate.data(), c.Gate.size()) || !read_exact(c.Up.data(), c.Up.size()) ||
        !read_exact(c.Down.data(), c.Down.size())) {
      return 2;
    }
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

  int final_in = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_in = it->in;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_in};
  target.data.resize(static_cast<std::size_t>(batch * final_in));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    if (!apply_links(g, links, n_links)) {
      return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "WSV1: bad links");
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::swiglu::Config cfg;
      cfg.input_dim = c.in;
      cfg.intermediate_dim = c.inter;
      welvet::swiglu::Layer layer;
      auto st = welvet::swiglu::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                               c.Gate.data(), c.Up.data(), c.Down.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_swiglu(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                             std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
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

  if (!write_exact("WSVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t nw = 2 * c.inter * c.in + c.in * c.inter;
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::swiglu::export_weights_f32(cell->swiglu_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WXB1 — Softmax NormalBP leaf (weightless dX)
int run_softmax_normalbp(int bench_iters) {
  int32_t n = 0, dim = 0, seq = 0, kind = 0, rows = 0, cols = 0, gumbel_seed = 0, dtype = 1,
          backend = 0, rank = 0, mask_len = 0, hier_len = 0;
  double temperature = 1.0, entmax_alpha = 0.0;
  if (!read_exact(&n, 1) || !read_exact(&dim, 1) || !read_exact(&seq, 1) || !read_exact(&kind, 1) ||
      !read_exact(&rows, 1) || !read_exact(&cols, 1) || !read_exact(&temperature, 1) ||
      !read_exact(&entmax_alpha, 1) || !read_exact(&gumbel_seed, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1) || !read_exact(&rank, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WXB1 header\n");
    return 2;
  }
  if (rank < 1 || rank > 8) {
    std::fprintf(stderr, "welvet_dense_fwd: bad softmax nbp rank\n");
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
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(n));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  (void)backend;
  std::vector<float> pre(static_cast<std::size_t>(n));
  std::vector<float> post(pre.size()), dX(pre.size());
  double loss = 0.0;

  welvet::softmax::Config cfg;
  cfg.dim = dim;
  cfg.seq_len = seq <= 0 ? 1 : seq;
  cfg.rows = rows;
  cfg.cols = cols;
  cfg.temperature = temperature;
  cfg.kind = static_cast<welvet::softmax::Kind>(kind);
  cfg.entmax_alpha = entmax_alpha;
  cfg.gumbel_seed = static_cast<uint32_t>(gumbel_seed);
  if (mask_len > 0) {
    cfg.mask.resize(static_cast<std::size_t>(mask_len));
    for (int32_t i = 0; i < mask_len; ++i) {
      cfg.mask[static_cast<std::size_t>(i)] = mask[static_cast<std::size_t>(i)] != 0;
    }
  }
  if (hier_len > 0) {
    cfg.hierarchy_levels.assign(hier.begin(), hier.end());
  }
  welvet::softmax::Layer layer;
  auto st0 = welvet::softmax::new_configured(cfg, static_cast<welvet::DType>(dtype), &layer);
  if (!st0) {
    std::fprintf(stderr, "welvet_dense_fwd: %s\n", st0.message.c_str());
    return 1;
  }
  welvet::TensorF32 input;
  input.shape.assign(shape.begin(), shape.end());
  input.data.assign(x.begin(), x.end());

  // Weightless: reuse layer; no W reset.
  auto step = [&](bool export_out) -> welvet::Status {
    welvet::TensorF32 pre_t, post_t;
    if (auto st = welvet::softmax::forward(layer, input, &pre_t, &post_t); !st) return st;
    welvet::TensorF32 gy;
    gy.shape = post_t.shape;
    gy.data.assign(gyt.begin(), gyt.end());
    loss = 0.0;
    if (use_target) {
      if (auto st = welvet::runtime::training::mse(post_t, gy, &loss); !st) return st;
      welvet::TensorF32 gtmp;
      if (auto st = welvet::runtime::training::mse_grad(post_t, gy, &gtmp); !st) return st;
      gy = std::move(gtmp);
    }
    welvet::TensorF32 dx_t, dw_t;
    if (auto st = welvet::softmax::backward_f32(layer, gy, input, pre_t, &dx_t, &dw_t); !st) {
      return st;
    }
    if (lr != 0.0) {
      if (auto st = welvet::softmax::apply_grad_sgd(&layer, dw_t, lr); !st) return st;
    }
    if (export_out) {
      if (static_cast<int>(pre_t.data.size()) != n || static_cast<int>(post_t.data.size()) != n ||
          static_cast<int>(dx_t.data.size()) != n) {
        return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "softmax nbp: out len");
      }
      std::memcpy(pre.data(), pre_t.data.data(), pre_t.data.size() * sizeof(float));
      std::memcpy(post.data(), post_t.data.data(), post_t.data.size() * sizeof(float));
      std::memcpy(dX.data(), dx_t.data.data(), dx_t.data.size() * sizeof(float));
    }
    return welvet::Status::Ok();
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = step(false);
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = step(false);
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
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  } else {
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = 0;
  const int32_t w_n = 0;
  if (!write_exact("WXBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(&w_n, 1) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WXV1 — Volume Softmax NormalBP
int run_volume_softmax_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WXV1")) return 2;
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0, seq_len = 1, kind = 0, rows = 0, cols = 0, gumbel_seed = 0, dtype = 1;
    double temperature = 1.0, entmax_alpha = 0.0;
    std::vector<uint8_t> mask;
    std::vector<int32_t> hier;
    welvet::architecture::Coord coord{};
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
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
    int32_t dim = 0, seq = 0, kind = 0, crow = 0, ccol = 0, gumbel_seed = 0, dtype = 1;
    double temperature = 1.0, entmax_alpha = 0.0;
    int32_t mask_len = 0, hier_len = 0;
    if (!read_exact(&dim, 1) || !read_exact(&seq, 1) || !read_exact(&kind, 1) ||
        !read_exact(&crow, 1) || !read_exact(&ccol, 1) || !read_exact(&temperature, 1) ||
        !read_exact(&entmax_alpha, 1) || !read_exact(&gumbel_seed, 1) || !read_exact(&dtype, 1) ||
        !read_exact(&mask_len, 1)) {
      return 2;
    }
    c.dim = dim;
    c.seq_len = seq;
    c.kind = kind;
    c.rows = crow;
    c.cols = ccol;
    c.temperature = temperature;
    c.entmax_alpha = entmax_alpha;
    c.gumbel_seed = gumbel_seed;
    c.dtype = dtype;
    c.mask.resize(static_cast<std::size_t>(mask_len > 0 ? mask_len : 0));
    if (mask_len > 0 && !read_exact(c.mask.data(), c.mask.size())) return 2;
    if (!read_exact(&hier_len, 1)) return 2;
    c.hier.resize(static_cast<std::size_t>(hier_len > 0 ? hier_len : 0));
    if (hier_len > 0 && !read_exact(c.hier.data(), c.hier.size())) return 2;
    if (first_dim < 0) first_dim = dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_dim = it->dim;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    if (!apply_links(g, links, n_links)) {
      return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "WXV1: bad links");
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::softmax::Config cfg;
      cfg.dim = c.dim;
      cfg.seq_len = c.seq_len <= 0 ? 1 : c.seq_len;
      cfg.rows = c.rows;
      cfg.cols = c.cols;
      cfg.temperature = c.temperature;
      cfg.kind = static_cast<welvet::softmax::Kind>(c.kind);
      cfg.entmax_alpha = c.entmax_alpha;
      cfg.gumbel_seed = static_cast<uint32_t>(c.gumbel_seed);
      if (!c.mask.empty()) {
        cfg.mask.resize(c.mask.size());
        for (std::size_t i = 0; i < c.mask.size(); ++i) {
          cfg.mask[i] = c.mask[i] != 0;
        }
      }
      if (!c.hier.empty()) {
        cfg.hierarchy_levels.assign(c.hier.begin(), c.hier.end());
      }
      welvet::softmax::Layer layer;
      auto st = welvet::softmax::new_configured(cfg, static_cast<welvet::DType>(c.dtype), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_softmax(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                              std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
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

  if (!write_exact("WXVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    const int32_t nw = 0; // weightless
    if (!write_exact(&nw, 1)) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WEB1 — Embedding NormalBP leaf
int run_embedding_normalbp(int bench_iters) {
  int32_t batch = 0, vocab = 0, emb = 0, seq = 0, in_rank = 2, in_last = 1, act = -1, dtype = 1,
          backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&vocab, 1) || !read_exact(&emb, 1) ||
      !read_exact(&seq, 1) || !read_exact(&in_rank, 1) || !read_exact(&in_last, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WEB1 header\n");
    return 2;
  }
  std::vector<float> W(static_cast<std::size_t>(vocab * emb));
  if (!read_exact(W.data(), W.size())) return 2;
  const int x_n = batch * seq * in_last;
  std::vector<float> x(static_cast<std::size_t>(x_n));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  const int out_n = (in_rank == 1 ? seq : batch * seq) * emb;
  std::vector<float> gyt(static_cast<std::size_t>(out_n));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  (void)backend;
  std::vector<float> pre(static_cast<std::size_t>(out_n));
  std::vector<float> post(pre.size()), dX(static_cast<std::size_t>(x_n)),
      dW(static_cast<std::size_t>(vocab * emb)), W_after(dW.size());
  double loss = 0.0;

  welvet::embedding::Config cfg;
  cfg.vocab_size = vocab;
  cfg.embedding_dim = emb;
  cfg.seq_len = seq;
  welvet::embedding::Layer layer;
  auto st0 = welvet::embedding::new_configured(cfg, static_cast<welvet::DType>(dtype),
                                               static_cast<welvet::Activation>(act), W.data(),
                                               &layer);
  if (!st0) {
    std::fprintf(stderr, "welvet_dense_fwd: %s\n", st0.message.c_str());
    return 1;
  }
  welvet::TensorF32 input;
  if (in_rank == 1) {
    input.shape = {seq};
  } else if (in_rank == 2) {
    input.shape = {batch, seq};
  } else {
    input.shape = {batch, seq, in_last};
  }
  input.data.assign(x.begin(), x.end());

  auto reset_w = [&]() -> welvet::Status {
    return welvet::WeightStore::make(vocab, emb, static_cast<welvet::DType>(dtype), W.data(),
                                     &layer.weights);
  };

  auto step = [&](bool export_out) -> welvet::Status {
    if (auto st = reset_w(); !st) return st;
    welvet::TensorF32 pre_t, post_t;
    if (auto st = welvet::embedding::forward(layer, input, &pre_t, &post_t); !st) return st;
    welvet::TensorF32 gy;
    gy.shape = post_t.shape;
    gy.data.assign(gyt.begin(), gyt.end());
    loss = 0.0;
    if (use_target) {
      if (auto st = welvet::runtime::training::mse(post_t, gy, &loss); !st) return st;
      welvet::TensorF32 gtmp;
      if (auto st = welvet::runtime::training::mse_grad(post_t, gy, &gtmp); !st) return st;
      gy = std::move(gtmp);
    }
    welvet::TensorF32 dx_t, dw_t;
    if (auto st = welvet::embedding::backward_f32(layer, gy, input, pre_t, &dx_t, &dw_t); !st) {
      return st;
    }
    if (lr != 0.0) {
      if (auto st = welvet::embedding::apply_grad_sgd(&layer, dw_t, lr); !st) return st;
    }
    if (export_out) {
      if (auto st = layer.weights.export_f32(W_after.data()); !st) return st;
      std::memcpy(pre.data(), pre_t.data.data(), pre_t.data.size() * sizeof(float));
      std::memcpy(post.data(), post_t.data.data(), post_t.data.size() * sizeof(float));
      std::memcpy(dX.data(), dx_t.data.data(), dx_t.data.size() * sizeof(float));
      std::memcpy(dW.data(), dw_t.data.data(), dw_t.data.size() * sizeof(float));
    }
    return welvet::Status::Ok();
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = step(false);
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = step(false);
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
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  } else {
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_exact("WEBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WEV1 — Volume Embedding NormalBP
int run_volume_embedding_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WEV1")) return 2;
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int vocab = 0, emb = 0, seq = 0, act = -1, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_seq = -1;
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
    int32_t vocab = 0, emb = 0, seq = 0, act = -1, dtype = 1;
    if (!read_exact(&vocab, 1) || !read_exact(&emb, 1) || !read_exact(&seq, 1) ||
        !read_exact(&act, 1) || !read_exact(&dtype, 1)) {
      return 2;
    }
    c.vocab = vocab;
    c.emb = emb;
    c.seq = seq;
    c.act = act;
    c.dtype = dtype;
    c.W.resize(static_cast<std::size_t>(vocab * emb));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_seq < 0) first_seq = seq;
    cells.push_back(std::move(c));
  }
  if (first_seq < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_seq};
  input.data.resize(static_cast<std::size_t>(batch * first_seq));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_emb = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_emb = it->emb;
      final_seq = it->seq;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_emb};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_emb));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    if (!apply_links(g, links, n_links)) {
      return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "WEV1: bad links");
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::embedding::Config cfg;
      cfg.vocab_size = c.vocab;
      cfg.embedding_dim = c.emb;
      cfg.seq_len = c.seq;
      welvet::embedding::Layer layer;
      auto st = welvet::embedding::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                  static_cast<welvet::Activation>(c.act), c.W.data(),
                                                  &layer);
      if (!st) return st;
      st = welvet::architecture::bind_embedding(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                                std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
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

  if (!write_exact("WEVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t nw = c.vocab * c.emb;
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = cell->embedding_op.weights.export_f32(Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WHB1 — MHA NormalBP leaf (ModeSelf)
int run_mha_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
  int32_t mask = 0, pos = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&d_model, 1) ||
      !read_exact(&num_heads, 1) || !read_exact(&num_kv, 1) || !read_exact(&head_dim, 1) ||
      !read_exact(&msl, 1) || !read_exact(&mask, 1) || !read_exact(&pos, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WHB1 header\n");
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
  const int nw = q_dim * d_model + kv_dim * d_model + kv_dim * d_model + d_model * q_dim;

  std::vector<float> Q(static_cast<std::size_t>(q_dim * d_model));
  std::vector<float> K(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> V(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> O(static_cast<std::size_t>(d_model * q_dim));
  if (!read_exact(Q.data(), Q.size()) || !read_exact(K.data(), K.size()) ||
      !read_exact(V.data(), V.size()) || !read_exact(O.data(), O.size())) {
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * d_model));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  (void)backend;
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * q_dim));
  std::vector<float> post(x.size()), dX(x.size()), dW(static_cast<std::size_t>(nw)),
      W_after(static_cast<std::size_t>(nw));
  double loss = 0.0;

  welvet::mha::Config cfg;
  cfg.d_model = d_model;
  cfg.num_heads = num_heads;
  cfg.num_kv_heads = num_kv;
  cfg.head_dim = head_dim;
  cfg.max_seq_len = msl;
  cfg.mask = static_cast<welvet::mha::MaskKind>(mask);
  cfg.pos = static_cast<welvet::mha::PosKind>(pos);
  cfg.mode = welvet::mha::AttnMode::Self;
  cfg.causal = (cfg.mask == welvet::mha::MaskKind::Causal);
  welvet::mha::Layer layer;
  auto st0 = welvet::mha::new_configured(cfg, static_cast<welvet::DType>(dtype), Q.data(), K.data(),
                                         V.data(), O.data(), &layer);
  if (!st0) {
    std::fprintf(stderr, "welvet_dense_fwd: %s\n", st0.message.c_str());
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, seq, d_model};
  input.data.assign(x.begin(), x.end());

  auto reset_w = [&]() -> welvet::Status {
    auto dt = static_cast<welvet::DType>(dtype);
    if (auto st = welvet::WeightStore::make(q_dim, d_model, dt, Q.data(), &layer.Q.weights); !st) {
      return st;
    }
    if (auto st = welvet::WeightStore::make(kv_dim, d_model, dt, K.data(), &layer.K.weights); !st) {
      return st;
    }
    if (auto st = welvet::WeightStore::make(kv_dim, d_model, dt, V.data(), &layer.V.weights); !st) {
      return st;
    }
    if (auto st = welvet::WeightStore::make(d_model, q_dim, dt, O.data(), &layer.O.weights); !st) {
      return st;
    }
    layer.Q.invalidate_gpu();
    layer.K.invalidate_gpu();
    layer.V.invalidate_gpu();
    layer.O.invalidate_gpu();
    return welvet::Status::Ok();
  };

  auto step = [&](bool export_out) -> welvet::Status {
    if (auto st = reset_w(); !st) return st;
    welvet::TensorF32 pre_t, post_t;
    if (auto st = welvet::mha::forward(layer, input, &pre_t, &post_t); !st) return st;
    welvet::TensorF32 gy;
    gy.shape = post_t.shape;
    gy.data.assign(gyt.begin(), gyt.end());
    loss = 0.0;
    if (use_target) {
      if (auto st = welvet::runtime::training::mse(post_t, gy, &loss); !st) return st;
      welvet::TensorF32 gtmp;
      if (auto st = welvet::runtime::training::mse_grad(post_t, gy, &gtmp); !st) return st;
      gy = std::move(gtmp);
    }
    welvet::TensorF32 dx_t, dw_t;
    if (auto st = welvet::mha::backward_f32(layer, gy, input, pre_t, &dx_t, &dw_t); !st) {
      return st;
    }
    if (lr != 0.0) {
      if (auto st = welvet::mha::apply_grad_sgd(&layer, dw_t, lr); !st) return st;
    }
    if (export_out) {
      if (auto st = welvet::mha::export_weights_f32(layer, W_after.data()); !st) return st;
      std::memcpy(pre.data(), pre_t.data.data(), pre_t.data.size() * sizeof(float));
      std::memcpy(post.data(), post_t.data.data(), post_t.data.size() * sizeof(float));
      std::memcpy(dX.data(), dx_t.data.data(), dx_t.data.size() * sizeof(float));
      std::memcpy(dW.data(), dw_t.data.data(), dw_t.data.size() * sizeof(float));
    }
    return welvet::Status::Ok();
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = step(false);
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = step(false);
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
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  } else {
    if (auto st = step(true); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_exact("WHBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WHV1 — Volume MHA NormalBP
int run_volume_mha_normalbp(int bench_iters) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_grid_hdr(&depth, &rows, &cols, &lpc, &batch, &backend, &n_links, "WHV1")) return 2;
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
    int mask = 0, pos = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> Q, K, V, O;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_seq = -1, first_d = -1;
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
    int32_t seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
    int32_t mask = 0, pos = 0, dtype = 1;
    if (!read_exact(&seq, 1) || !read_exact(&d_model, 1) || !read_exact(&num_heads, 1) ||
        !read_exact(&num_kv, 1) || !read_exact(&head_dim, 1) || !read_exact(&msl, 1) ||
        !read_exact(&mask, 1) || !read_exact(&pos, 1) || !read_exact(&dtype, 1)) {
      return 2;
    }
    c.seq = seq;
    c.d_model = d_model;
    c.num_heads = num_heads;
    c.num_kv = num_kv;
    c.head_dim = head_dim;
    c.msl = msl;
    c.mask = mask;
    c.pos = pos;
    c.dtype = dtype;
    const int hd = head_dim > 0 ? head_dim : d_model / num_heads;
    const int kv_heads = num_kv > 0 ? num_kv : num_heads;
    const int q_dim = num_heads * hd;
    const int kv_dim = kv_heads * hd;
    c.Q.resize(static_cast<std::size_t>(q_dim * d_model));
    c.K.resize(static_cast<std::size_t>(kv_dim * d_model));
    c.V.resize(static_cast<std::size_t>(kv_dim * d_model));
    c.O.resize(static_cast<std::size_t>(d_model * q_dim));
    if (!read_exact(c.Q.data(), c.Q.size()) || !read_exact(c.K.data(), c.K.size()) ||
        !read_exact(c.V.data(), c.V.size()) || !read_exact(c.O.data(), c.O.size())) {
      return 2;
    }
    if (first_seq < 0) {
      first_seq = seq;
      first_d = d_model;
    }
    cells.push_back(std::move(c));
  }
  if (first_seq < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_d};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_d));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_seq = 0, final_d = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_seq = it->seq;
      final_d = it->d_model;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_d};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_d));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    if (!apply_links(g, links, n_links)) {
      return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "WHV1: bad links");
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::mha::Config cfg;
      cfg.d_model = c.d_model;
      cfg.num_heads = c.num_heads;
      cfg.num_kv_heads = c.num_kv;
      cfg.head_dim = c.head_dim;
      cfg.max_seq_len = c.msl;
      cfg.mask = static_cast<welvet::mha::MaskKind>(c.mask);
      cfg.pos = static_cast<welvet::mha::PosKind>(c.pos);
      cfg.mode = welvet::mha::AttnMode::Self;
      cfg.causal = (cfg.mask == welvet::mha::MaskKind::Causal);
      welvet::mha::Layer layer;
      auto st = welvet::mha::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.Q.data(),
                                            c.K.data(), c.V.data(), c.O.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_mha(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                          std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
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

  if (!write_exact("WHVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t nw = welvet::mha::grad_w_size(cell->mha_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::mha::export_weights_f32(cell->mha_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}
