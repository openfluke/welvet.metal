// Volume forwards: bind every cell once, then volume_dense (WVB2 splits mount vs exec).

#include "volume_fwd.hpp"

#include "welvet/architecture.hpp"
#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/kmeans.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/mha.hpp"
#include "welvet/parallel.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/softmax.hpp"

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
