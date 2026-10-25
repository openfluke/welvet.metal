#include "welvet/gdn.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace welvet::gdn {
namespace {

void matvec_f32(const float* w, int rows, int cols, const float* x, float* y) {
  for (int r = 0; r < rows; ++r) {
    float sum = 0.0f;
    const float* row = w + r * cols;
    for (int c = 0; c < cols; ++c) {
      sum += row[c] * x[c];
    }
    y[r] = sum;
  }
}

void matvec_t_f32(const float* w, int rows, int cols, const float* gy, float* gx) {
  for (int r = 0; r < rows; ++r) {
    const float g = gy[r];
    const float* row = w + r * cols;
    for (int c = 0; c < cols; ++c) {
      gx[c] += row[c] * g;
    }
  }
}

float silu(float x) {
  return x / (1.0f + static_cast<float>(std::exp(static_cast<double>(-x))));
}

float silu_deriv(float x) {
  const float s = 1.0f / (1.0f + static_cast<float>(std::exp(static_cast<double>(-x))));
  return s + x * s * (1.0f - s);
}

double softplus(double x) {
  if (x > 20.0) return x;
  return std::log1p(std::exp(x));
}

void l2norm_inplace(float* x, int n, double eps) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const double v = static_cast<double>(x[i]);
    s += v * v;
  }
  const float inv = static_cast<float>(1.0 / std::sqrt(s + eps));
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

void rms_norm_gated(float* x, const float* z, const float* gamma, int n, float eps) {
  float mean = 0.0f;
  for (int i = 0; i < n; ++i) mean += x[i] * x[i];
  mean /= static_cast<float>(n);
  const float inv = 1.0f / static_cast<float>(std::sqrt(static_cast<double>(mean + eps)));
  for (int i = 0; i < n; ++i) {
    const float g = (gamma) ? gamma[i] : 1.0f;
    x[i] = x[i] * inv * g * silu(z[i]);
  }
}

// Match Go l2normBackward (float64 round-trip on sum).
void l2norm_backward(const float* x_raw, int n, float inv, const float* dy, float* dx) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const float y = x_raw[i] * inv;
    s += static_cast<double>(dy[i]) * static_cast<double>(y);
  }
  const float sf = static_cast<float>(s);
  for (int j = 0; j < n; ++j) {
    const float y = x_raw[j] * inv;
    dx[j] = inv * (dy[j] - y * sf);
  }
}

void accum_outer(float* dW, const float* gy, const float* x, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float gr = gy[r];
    if (gr == 0.0f) continue;
    float* base = dW + r * cols;
    for (int c = 0; c < cols; ++c) {
      base[c] += gr * x[c];
    }
  }
}

struct Scratch {
  std::vector<float> qkv, z, beta_raw, a_raw;
  std::vector<float> mixed, q_rep, k_rep;
  std::vector<float> beta, g, core;
  std::vector<float> kv_mem, delta;
  std::vector<float> conv_state, state;
};

void ensure_scratch(Scratch& s, const Config& c) {
  const int cd = conv_dim(c);
  const int vd = value_dim(c);
  const int nh = c.num_value_heads;
  const int hd_k = c.key_head_dim;
  const int hd_v = c.value_head_dim;
  const int k = c.conv_kernel < 1 ? 1 : c.conv_kernel;
  s.qkv.assign(static_cast<std::size_t>(cd), 0.0f);
  s.z.assign(static_cast<std::size_t>(vd), 0.0f);
  s.beta_raw.assign(static_cast<std::size_t>(nh), 0.0f);
  s.a_raw.assign(static_cast<std::size_t>(nh), 0.0f);
  s.mixed.assign(static_cast<std::size_t>(cd), 0.0f);
  s.q_rep.assign(static_cast<std::size_t>(nh * hd_k), 0.0f);
  s.k_rep.assign(static_cast<std::size_t>(nh * hd_k), 0.0f);
  s.beta.assign(static_cast<std::size_t>(nh), 0.0f);
  s.g.assign(static_cast<std::size_t>(nh), 0.0f);
  s.core.assign(static_cast<std::size_t>(nh * hd_v), 0.0f);
  s.kv_mem.assign(static_cast<std::size_t>(hd_v), 0.0f);
  s.delta.assign(static_cast<std::size_t>(hd_v), 0.0f);
  s.conv_state.assign(static_cast<std::size_t>(cd * (k - 1)), 0.0f);
  s.state.assign(static_cast<std::size_t>(nh * hd_k * hd_v), 0.0f);
}

struct GdnStep {
  std::vector<float> x;                 // H
  std::vector<float> qkv;               // convDim
  std::vector<float> conv_hist;         // convDim*(k-1)
  std::vector<float> acc;               // convDim
  std::vector<float> mixed;             // convDim
  std::vector<float> q_inv, k_inv;      // nk
  std::vector<float> q_rep, k_rep;      // nh*hdK
  std::vector<float> beta_raw, a_raw;   // nh
  std::vector<float> beta, g;           // nh
  std::vector<float> state_old;         // nh*hdK*hdV
  std::vector<float> state_after_decay; // nh*hdK*hdV
  std::vector<float> kv_mem;            // valDim
  std::vector<float> delta;             // valDim
  std::vector<float> core_pre;          // valDim
  std::vector<float> z;                 // valDim
  std::vector<float> core_normed;       // valDim
};

struct GdnGrads {
  std::vector<float> d_in_qkv, d_in_z, d_in_b, d_in_a, d_out;
  std::vector<float> d_conv_weight, d_a_log, d_dt_bias, d_norm_gamma;
};

GdnGrads new_grads(const Config& cfg) {
  const int cd = conv_dim(cfg);
  const int vd = value_dim(cfg);
  const int nh = cfg.num_value_heads;
  const int h = cfg.hidden_size;
  const int k = cfg.conv_kernel < 1 ? 1 : cfg.conv_kernel;
  GdnGrads g;
  g.d_in_qkv.assign(static_cast<std::size_t>(cd * h), 0.0f);
  g.d_in_z.assign(static_cast<std::size_t>(vd * h), 0.0f);
  g.d_in_b.assign(static_cast<std::size_t>(nh * h), 0.0f);
  g.d_in_a.assign(static_cast<std::size_t>(nh * h), 0.0f);
  g.d_out.assign(static_cast<std::size_t>(h * vd), 0.0f);
  g.d_conv_weight.assign(static_cast<std::size_t>(cd * k), 0.0f);
  g.d_a_log.assign(static_cast<std::size_t>(nh), 0.0f);
  g.d_dt_bias.assign(static_cast<std::size_t>(nh), 0.0f);
  g.d_norm_gamma.assign(static_cast<std::size_t>(cfg.value_head_dim), 0.0f);
  return g;
}

void flatten_grads(const GdnGrads& g, float* out) {
  int off = 0;
  auto put = [&](const std::vector<float>& s) {
    if (!s.empty()) {
      std::memcpy(out + off, s.data(), s.size() * sizeof(float));
      off += static_cast<int>(s.size());
    }
  };
  put(g.d_in_qkv);
  put(g.d_in_z);
  put(g.d_in_b);
  put(g.d_in_a);
  put(g.d_out);
  put(g.d_conv_weight);
  put(g.d_a_log);
  put(g.d_dt_bias);
  put(g.d_norm_gamma);
}

Status forward_decode(Layer& layer, Scratch& s, const float* x, float* y) {
  const Config& c = layer.cfg;
  const int h = c.hidden_size;
  const int key_d = key_dim(c);
  const int val_d = value_dim(c);
  const int conv_d = conv_dim(c);
  const int k = c.conv_kernel < 1 ? 1 : c.conv_kernel;
  const int nh = c.num_value_heads;
  const int nk = c.num_key_heads;
  const int hd_k = c.key_head_dim;
  const int hd_v = c.value_head_dim;
  const int rep = nh / nk;

  matvec_f32(layer.in_qkv.data(), conv_d, h, x, s.qkv.data());
  matvec_f32(layer.in_z.data(), val_d, h, x, s.z.data());
  matvec_f32(layer.in_b.data(), nh, h, x, s.beta_raw.data());
  matvec_f32(layer.in_a.data(), nh, h, x, s.a_raw.data());

  for (int ch = 0; ch < conv_d; ++ch) {
    float acc = 0.0f;
    for (int t = 0; t < k - 1; ++t) {
      acc += layer.conv_weight[static_cast<std::size_t>(ch * k + t)] *
             s.conv_state[static_cast<std::size_t>(ch * (k - 1) + t)];
    }
    acc += layer.conv_weight[static_cast<std::size_t>(ch * k + (k - 1))] * s.qkv[static_cast<std::size_t>(ch)];
    s.mixed[static_cast<std::size_t>(ch)] = silu(acc);
  }
  for (int ch = 0; ch < conv_d; ++ch) {
    const int base = ch * (k - 1);
    if (k > 2) {
      std::memmove(s.conv_state.data() + base, s.conv_state.data() + base + 1,
                   static_cast<std::size_t>(k - 2) * sizeof(float));
    }
    if (k > 1) {
      s.conv_state[static_cast<std::size_t>(base + k - 2)] = s.qkv[static_cast<std::size_t>(ch)];
    }
  }

  float* q = s.mixed.data();
  float* key = s.mixed.data() + key_d;
  float* v = s.mixed.data() + key_d * 2;

  for (int hi = 0; hi < nk; ++hi) {
    l2norm_inplace(q + hi * hd_k, hd_k, 1e-6);
    l2norm_inplace(key + hi * hd_k, hd_k, 1e-6);
    for (int r = 0; r < rep; ++r) {
      const int dst = (hi * rep + r) * hd_k;
      std::memcpy(s.q_rep.data() + dst, q + hi * hd_k, static_cast<std::size_t>(hd_k) * sizeof(float));
      std::memcpy(s.k_rep.data() + dst, key + hi * hd_k, static_cast<std::size_t>(hd_k) * sizeof(float));
    }
  }
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd_k)));
  for (float& vq : s.q_rep) vq *= scale;

  for (int i = 0; i < nh; ++i) {
    s.beta[static_cast<std::size_t>(i)] =
        1.0f / (1.0f + static_cast<float>(std::exp(static_cast<double>(-s.beta_raw[static_cast<std::size_t>(i)]))));
    const double al = static_cast<double>(layer.a_log[static_cast<std::size_t>(i)]);
    const double dt = static_cast<double>(s.a_raw[static_cast<std::size_t>(i)] +
                                          layer.dt_bias[static_cast<std::size_t>(i)]);
    s.g[static_cast<std::size_t>(i)] = static_cast<float>(-std::exp(al) * softplus(dt));
  }

  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    float* st = s.state.data() + h_idx * hd_k * hd_v;
    const float gt = static_cast<float>(std::exp(static_cast<double>(s.g[static_cast<std::size_t>(h_idx)])));
    const float bt = s.beta[static_cast<std::size_t>(h_idx)];
    const float* qt = s.q_rep.data() + h_idx * hd_k;
    const float* kt = s.k_rep.data() + h_idx * hd_k;
    const float* vt = v + h_idx * hd_v;

    for (int i = 0; i < hd_k * hd_v; ++i) st[i] *= gt;
    for (int d = 0; d < hd_v; ++d) {
      float sum = 0.0f;
      for (int j = 0; j < hd_k; ++j) sum += st[j * hd_v + d] * kt[j];
      s.kv_mem[static_cast<std::size_t>(d)] = sum;
    }
    for (int d = 0; d < hd_v; ++d) {
      s.delta[static_cast<std::size_t>(d)] = (vt[d] - s.kv_mem[static_cast<std::size_t>(d)]) * bt;
    }
    for (int j = 0; j < hd_k; ++j) {
      for (int d = 0; d < hd_v; ++d) {
        st[j * hd_v + d] += kt[j] * s.delta[static_cast<std::size_t>(d)];
      }
    }
    float* out_h = s.core.data() + h_idx * hd_v;
    for (int d = 0; d < hd_v; ++d) {
      float sum = 0.0f;
      for (int j = 0; j < hd_k; ++j) sum += st[j * hd_v + d] * qt[j];
      out_h[d] = sum;
    }
  }

  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    float* o = s.core.data() + h_idx * hd_v;
    const float* zz = s.z.data() + h_idx * hd_v;
    rms_norm_gated(o, zz, layer.norm_gamma.data(), hd_v, static_cast<float>(c.eps));
  }

  matvec_f32(layer.out.data(), h, val_d, s.core.data(), y);
  return Status::Ok();
}

// Mirrors ForwardDecode while recording tape for truncated-BPTT (Go forwardDecodeTape).
Status forward_decode_tape(Layer& layer, Scratch& s, const float* x, float* y, GdnStep* step) {
  if (!step) return Status::Err(StatusCode::InvalidArgument, "gdn: nil step");
  const Config& c = layer.cfg;
  const int h = c.hidden_size;
  const int key_d = key_dim(c);
  const int val_d = value_dim(c);
  const int conv_d = conv_dim(c);
  const int k = c.conv_kernel < 1 ? 1 : c.conv_kernel;
  const int nh = c.num_value_heads;
  const int nk = c.num_key_heads;
  const int hd_k = c.key_head_dim;
  const int hd_v = c.value_head_dim;
  const int rep = nh / nk;

  GdnStep tp;
  tp.x.assign(x, x + h);
  tp.qkv.assign(static_cast<std::size_t>(conv_d), 0.0f);
  tp.z.assign(static_cast<std::size_t>(val_d), 0.0f);
  tp.beta_raw.assign(static_cast<std::size_t>(nh), 0.0f);
  tp.a_raw.assign(static_cast<std::size_t>(nh), 0.0f);

  matvec_f32(layer.in_qkv.data(), conv_d, h, x, tp.qkv.data());
  matvec_f32(layer.in_z.data(), val_d, h, x, tp.z.data());
  matvec_f32(layer.in_b.data(), nh, h, x, tp.beta_raw.data());
  matvec_f32(layer.in_a.data(), nh, h, x, tp.a_raw.data());

  tp.conv_hist = s.conv_state;
  tp.acc.assign(static_cast<std::size_t>(conv_d), 0.0f);
  tp.mixed.assign(static_cast<std::size_t>(conv_d), 0.0f);
  for (int ch = 0; ch < conv_d; ++ch) {
    float a = 0.0f;
    for (int t = 0; t < k - 1; ++t) {
      a += layer.conv_weight[static_cast<std::size_t>(ch * k + t)] *
           s.conv_state[static_cast<std::size_t>(ch * (k - 1) + t)];
    }
    a += layer.conv_weight[static_cast<std::size_t>(ch * k + (k - 1))] * tp.qkv[static_cast<std::size_t>(ch)];
    tp.acc[static_cast<std::size_t>(ch)] = a;
    tp.mixed[static_cast<std::size_t>(ch)] = silu(a);
  }
  for (int ch = 0; ch < conv_d; ++ch) {
    const int base = ch * (k - 1);
    if (k > 2) {
      std::memmove(s.conv_state.data() + base, s.conv_state.data() + base + 1,
                   static_cast<std::size_t>(k - 2) * sizeof(float));
    }
    if (k > 1) {
      s.conv_state[static_cast<std::size_t>(base + k - 2)] = tp.qkv[static_cast<std::size_t>(ch)];
    }
  }

  const float* q = tp.mixed.data();
  const float* key = tp.mixed.data() + key_d;
  const float* v = tp.mixed.data() + key_d * 2;

  tp.q_rep.assign(static_cast<std::size_t>(nh * hd_k), 0.0f);
  tp.k_rep.assign(static_cast<std::size_t>(nh * hd_k), 0.0f);
  tp.q_inv.assign(static_cast<std::size_t>(nk), 0.0f);
  tp.k_inv.assign(static_cast<std::size_t>(nk), 0.0f);
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd_k)));
  for (int hi = 0; hi < nk; ++hi) {
    const float* qs = q + hi * hd_k;
    const float* ks = key + hi * hd_k;
    double sq = 0.0, sk = 0.0;
    for (int j = 0; j < hd_k; ++j) {
      const double vq = static_cast<double>(qs[j]);
      sq += vq * vq;
    }
    for (int j = 0; j < hd_k; ++j) {
      const double vk = static_cast<double>(ks[j]);
      sk += vk * vk;
    }
    const float invq = static_cast<float>(1.0 / std::sqrt(sq + 1e-6));
    const float invk = static_cast<float>(1.0 / std::sqrt(sk + 1e-6));
    tp.q_inv[static_cast<std::size_t>(hi)] = invq;
    tp.k_inv[static_cast<std::size_t>(hi)] = invk;
    for (int r = 0; r < rep; ++r) {
      const int dst = (hi * rep + r) * hd_k;
      for (int j = 0; j < hd_k; ++j) {
        tp.q_rep[static_cast<std::size_t>(dst + j)] = qs[j] * invq * scale;
        tp.k_rep[static_cast<std::size_t>(dst + j)] = ks[j] * invk;
      }
    }
  }

  tp.beta.assign(static_cast<std::size_t>(nh), 0.0f);
  tp.g.assign(static_cast<std::size_t>(nh), 0.0f);
  for (int i = 0; i < nh; ++i) {
    tp.beta[static_cast<std::size_t>(i)] =
        1.0f / (1.0f + static_cast<float>(std::exp(static_cast<double>(-tp.beta_raw[static_cast<std::size_t>(i)]))));
    const double al = static_cast<double>(layer.a_log[static_cast<std::size_t>(i)]);
    const double dt = static_cast<double>(tp.a_raw[static_cast<std::size_t>(i)] +
                                          layer.dt_bias[static_cast<std::size_t>(i)]);
    tp.g[static_cast<std::size_t>(i)] = static_cast<float>(-std::exp(al) * softplus(dt));
  }

  tp.state_old = s.state;
  std::vector<float> state = s.state;
  tp.state_after_decay.assign(state.size(), 0.0f);
  tp.kv_mem.assign(static_cast<std::size_t>(val_d), 0.0f);
  tp.delta.assign(static_cast<std::size_t>(val_d), 0.0f);
  tp.core_pre.assign(static_cast<std::size_t>(val_d), 0.0f);

  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    float* st = state.data() + h_idx * hd_k * hd_v;
    const float gt = static_cast<float>(std::exp(static_cast<double>(tp.g[static_cast<std::size_t>(h_idx)])));
    const float bt = tp.beta[static_cast<std::size_t>(h_idx)];
    const float* qt = tp.q_rep.data() + h_idx * hd_k;
    const float* kt = tp.k_rep.data() + h_idx * hd_k;
    const float* vt = v + h_idx * hd_v;

    for (int i = 0; i < hd_k * hd_v; ++i) st[i] *= gt;
    std::memcpy(tp.state_after_decay.data() + h_idx * hd_k * hd_v, st,
                static_cast<std::size_t>(hd_k * hd_v) * sizeof(float));

    float* kvm = tp.kv_mem.data() + h_idx * hd_v;
    for (int d = 0; d < hd_v; ++d) {
      float sum = 0.0f;
      for (int j = 0; j < hd_k; ++j) sum += st[j * hd_v + d] * kt[j];
      kvm[d] = sum;
    }
    float* dl = tp.delta.data() + h_idx * hd_v;
    for (int d = 0; d < hd_v; ++d) {
      dl[d] = (vt[d] - kvm[d]) * bt;
    }
    for (int j = 0; j < hd_k; ++j) {
      for (int d = 0; d < hd_v; ++d) {
        st[j * hd_v + d] += kt[j] * dl[d];
      }
    }
    float* out_h = tp.core_pre.data() + h_idx * hd_v;
    for (int d = 0; d < hd_v; ++d) {
      float sum = 0.0f;
      for (int j = 0; j < hd_k; ++j) sum += st[j * hd_v + d] * qt[j];
      out_h[d] = sum;
    }
  }
  s.state = state;

  tp.core_normed = tp.core_pre;
  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    float* o = tp.core_normed.data() + h_idx * hd_v;
    const float* zz = tp.z.data() + h_idx * hd_v;
    rms_norm_gated(o, zz, layer.norm_gamma.data(), hd_v, static_cast<float>(c.eps));
  }

  matvec_f32(layer.out.data(), h, val_d, tp.core_normed.data(), y);
  *step = std::move(tp);
  return Status::Ok();
}

Status backward_step(Layer& layer, const GdnStep& tp, const float* dy, GdnGrads* g,
                     float* dx) {
  if (!g || !dx) return Status::Err(StatusCode::InvalidArgument, "gdn: nil bwd buffers");
  const Config& c = layer.cfg;
  const int h = c.hidden_size;
  const int key_d = key_dim(c);
  const int val_d = value_dim(c);
  const int conv_d = conv_dim(c);
  const int nh = c.num_value_heads;
  const int nk = c.num_key_heads;
  const int rep = nh / nk;
  const int hd_k = c.key_head_dim;
  const int hd_v = c.value_head_dim;
  const int k = c.conv_kernel < 1 ? 1 : c.conv_kernel;

  std::vector<float> d_core_normed(static_cast<std::size_t>(val_d), 0.0f);
  matvec_t_f32(layer.out.data(), h, val_d, dy, d_core_normed.data());
  accum_outer(g->d_out.data(), dy, tp.core_normed.data(), h, val_d);

  std::vector<float> d_core_pre(static_cast<std::size_t>(val_d), 0.0f);
  std::vector<float> d_z(static_cast<std::size_t>(val_d), 0.0f);
  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    const float* xr = tp.core_pre.data() + h_idx * hd_v;
    const float* zz = tp.z.data() + h_idx * hd_v;
    const float* dyh = d_core_normed.data() + h_idx * hd_v;
    const int n = hd_v;
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += xr[i] * xr[i];
    mean /= static_cast<float>(n);
    const float inv = 1.0f / static_cast<float>(std::sqrt(static_cast<double>(mean + static_cast<float>(c.eps))));
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
      const float gamma = (i < static_cast<int>(layer.norm_gamma.size())) ? layer.norm_gamma[static_cast<std::size_t>(i)] : 1.0f;
      const float sz = silu(zz[i]);
      const float ci = gamma * sz;
      s += static_cast<double>(dyh[i]) * static_cast<double>(ci) * static_cast<double>(xr[i]);
    }
    const float sf = static_cast<float>(s);
    for (int i = 0; i < n; ++i) {
      const float gamma = (i < static_cast<int>(layer.norm_gamma.size())) ? layer.norm_gamma[static_cast<std::size_t>(i)] : 1.0f;
      const float sz = silu(zz[i]);
      const float ci = gamma * sz;
      d_core_pre[static_cast<std::size_t>(h_idx * hd_v + i)] =
          inv * ci * dyh[i] - inv * inv * inv / static_cast<float>(n) * xr[i] * sf;
      d_z[static_cast<std::size_t>(h_idx * hd_v + i)] =
          dyh[i] * xr[i] * inv * gamma * silu_deriv(zz[i]);
      if (i < static_cast<int>(g->d_norm_gamma.size())) {
        g->d_norm_gamma[static_cast<std::size_t>(i)] += dyh[i] * xr[i] * inv * sz;
      }
    }
  }

  std::vector<float> d_mixed(static_cast<std::size_t>(conv_d), 0.0f);
  std::vector<float> d_beta_raw(static_cast<std::size_t>(nh), 0.0f);
  std::vector<float> d_a_raw(static_cast<std::size_t>(nh), 0.0f);
  std::vector<float> d_q_rep(static_cast<std::size_t>(nh * hd_k), 0.0f);
  std::vector<float> d_k_rep(static_cast<std::size_t>(nh * hd_k), 0.0f);

  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    const float* dout_h = d_core_pre.data() + h_idx * hd_v;
    const float* st_after = tp.state_after_decay.data() + h_idx * hd_k * hd_v;
    const float* st_old = tp.state_old.data() + h_idx * hd_k * hd_v;
    const float* dl = tp.delta.data() + h_idx * hd_v;
    const float* qt = tp.q_rep.data() + h_idx * hd_k;
    const float* kt = tp.k_rep.data() + h_idx * hd_k;
    const float* v = tp.mixed.data() + key_d * 2 + h_idx * hd_v;

    std::vector<float> d_state_final(static_cast<std::size_t>(hd_k * hd_v), 0.0f);
    std::vector<float> dqt(static_cast<std::size_t>(hd_k), 0.0f);
    for (int j = 0; j < hd_k; ++j) {
      float sum = 0.0f;
      for (int d = 0; d < hd_v; ++d) {
        const float st_final = st_after[j * hd_v + d] + kt[j] * dl[d];
        d_state_final[static_cast<std::size_t>(j * hd_v + d)] = dout_h[d] * qt[j];
        sum += st_final * dout_h[d];
      }
      dqt[static_cast<std::size_t>(j)] = sum;
    }
    std::vector<float> dkt(static_cast<std::size_t>(hd_k), 0.0f);
    std::vector<float> ddelta(static_cast<std::size_t>(hd_v), 0.0f);
    std::vector<float> d_state_after(static_cast<std::size_t>(hd_k * hd_v), 0.0f);
    for (int j = 0; j < hd_k; ++j) {
      float dktj = 0.0f;
      for (int d = 0; d < hd_v; ++d) {
        const float dsf = d_state_final[static_cast<std::size_t>(j * hd_v + d)];
        d_state_after[static_cast<std::size_t>(j * hd_v + d)] += dsf;
        dktj += dl[d] * dsf;
        ddelta[static_cast<std::size_t>(d)] += kt[j] * dsf;
      }
      dkt[static_cast<std::size_t>(j)] += dktj;
    }
    const float bt = tp.beta[static_cast<std::size_t>(h_idx)];
    const float* kvm = tp.kv_mem.data() + h_idx * hd_v;
    float dbt = 0.0f;
    std::vector<float> d_kv_mem(static_cast<std::size_t>(hd_v), 0.0f);
    for (int d = 0; d < hd_v; ++d) {
      const float ddl = ddelta[static_cast<std::size_t>(d)];
      dbt += ddl * (v[d] - kvm[d]);
      d_kv_mem[static_cast<std::size_t>(d)] = -ddl * bt;
      d_mixed[static_cast<std::size_t>(key_d * 2 + h_idx * hd_v + d)] += ddl * bt;
    }
    for (int j = 0; j < hd_k; ++j) {
      float dktj = 0.0f;
      for (int d = 0; d < hd_v; ++d) {
        d_state_after[static_cast<std::size_t>(j * hd_v + d)] +=
            d_kv_mem[static_cast<std::size_t>(d)] * kt[j];
        dktj += d_kv_mem[static_cast<std::size_t>(d)] * st_after[j * hd_v + d];
      }
      dkt[static_cast<std::size_t>(j)] += dktj;
    }

    const float gt = static_cast<float>(std::exp(static_cast<double>(tp.g[static_cast<std::size_t>(h_idx)])));
    float dgt = 0.0f;
    for (int j = 0; j < hd_k; ++j) {
      for (int d = 0; d < hd_v; ++d) {
        dgt += d_state_after[static_cast<std::size_t>(j * hd_v + d)] * st_old[j * hd_v + d];
      }
    }
    const float d_gate = dgt * gt;
    const double al = static_cast<double>(layer.a_log[static_cast<std::size_t>(h_idx)]);
    const double dt_raw = static_cast<double>(tp.a_raw[static_cast<std::size_t>(h_idx)] +
                                              layer.dt_bias[static_cast<std::size_t>(h_idx)]);
    const float sig = static_cast<float>(1.0 / (1.0 + std::exp(-dt_raw)));
    const float d_dt_raw = d_gate * static_cast<float>(-std::exp(al)) * sig;
    d_a_raw[static_cast<std::size_t>(h_idx)] += d_dt_raw;
    g->d_dt_bias[static_cast<std::size_t>(h_idx)] += d_dt_raw;
    g->d_a_log[static_cast<std::size_t>(h_idx)] += d_gate * tp.g[static_cast<std::size_t>(h_idx)];

    d_beta_raw[static_cast<std::size_t>(h_idx)] += dbt * bt * (1.0f - bt);

    for (int j = 0; j < hd_k; ++j) {
      d_q_rep[static_cast<std::size_t>(h_idx * hd_k + j)] = dqt[static_cast<std::size_t>(j)];
      d_k_rep[static_cast<std::size_t>(h_idx * hd_k + j)] = dkt[static_cast<std::size_t>(j)];
    }
  }

  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd_k)));
  std::vector<float> dq(static_cast<std::size_t>(key_d), 0.0f);
  std::vector<float> dk(static_cast<std::size_t>(key_d), 0.0f);
  for (int hi = 0; hi < nk; ++hi) {
    std::vector<float> dq_norm(static_cast<std::size_t>(hd_k), 0.0f);
    std::vector<float> dk_norm(static_cast<std::size_t>(hd_k), 0.0f);
    for (int r = 0; r < rep; ++r) {
      const int h_idx = hi * rep + r;
      for (int j = 0; j < hd_k; ++j) {
        dq_norm[static_cast<std::size_t>(j)] +=
            d_q_rep[static_cast<std::size_t>(h_idx * hd_k + j)] * scale;
        dk_norm[static_cast<std::size_t>(j)] += d_k_rep[static_cast<std::size_t>(h_idx * hd_k + j)];
      }
    }
    const float* q_raw = tp.mixed.data() + hi * hd_k;
    const float* k_raw = tp.mixed.data() + key_d + hi * hd_k;
    std::vector<float> dq_raw(static_cast<std::size_t>(hd_k), 0.0f);
    std::vector<float> dk_raw(static_cast<std::size_t>(hd_k), 0.0f);
    l2norm_backward(q_raw, hd_k, tp.q_inv[static_cast<std::size_t>(hi)], dq_norm.data(),
                    dq_raw.data());
    l2norm_backward(k_raw, hd_k, tp.k_inv[static_cast<std::size_t>(hi)], dk_norm.data(),
                    dk_raw.data());
    std::memcpy(dq.data() + hi * hd_k, dq_raw.data(), static_cast<std::size_t>(hd_k) * sizeof(float));
    std::memcpy(dk.data() + hi * hd_k, dk_raw.data(), static_cast<std::size_t>(hd_k) * sizeof(float));
  }
  std::memcpy(d_mixed.data(), dq.data(), static_cast<std::size_t>(key_d) * sizeof(float));
  for (int i = 0; i < key_d; ++i) {
    d_mixed[static_cast<std::size_t>(key_d + i)] += dk[static_cast<std::size_t>(i)];
  }

  std::vector<float> d_qkv(static_cast<std::size_t>(conv_d), 0.0f);
  for (int ch = 0; ch < conv_d; ++ch) {
    const float d_acc = d_mixed[static_cast<std::size_t>(ch)] * silu_deriv(tp.acc[static_cast<std::size_t>(ch)]);
    for (int t = 0; t < k - 1; ++t) {
      g->d_conv_weight[static_cast<std::size_t>(ch * k + t)] +=
          d_acc * tp.conv_hist[static_cast<std::size_t>(ch * (k - 1) + t)];
    }
    g->d_conv_weight[static_cast<std::size_t>(ch * k + (k - 1))] +=
        d_acc * tp.qkv[static_cast<std::size_t>(ch)];
    d_qkv[static_cast<std::size_t>(ch)] +=
        d_acc * layer.conv_weight[static_cast<std::size_t>(ch * k + (k - 1))];
  }

  std::memset(dx, 0, static_cast<std::size_t>(h) * sizeof(float));
  matvec_t_f32(layer.in_qkv.data(), conv_d, h, d_qkv.data(), dx);
  accum_outer(g->d_in_qkv.data(), d_qkv.data(), tp.x.data(), conv_d, h);

  matvec_t_f32(layer.in_z.data(), val_d, h, d_z.data(), dx);
  accum_outer(g->d_in_z.data(), d_z.data(), tp.x.data(), val_d, h);

  matvec_t_f32(layer.in_b.data(), nh, h, d_beta_raw.data(), dx);
  accum_outer(g->d_in_b.data(), d_beta_raw.data(), tp.x.data(), nh, h);

  matvec_t_f32(layer.in_a.data(), nh, h, d_a_raw.data(), dx);
  accum_outer(g->d_in_a.data(), d_a_raw.data(), tp.x.data(), nh, h);

  return Status::Ok();
}

} // namespace

int key_dim(const Config& cfg) { return cfg.num_key_heads * cfg.key_head_dim; }

int value_dim(const Config& cfg) { return cfg.num_value_heads * cfg.value_head_dim; }

int conv_dim(const Config& cfg) { return key_dim(cfg) * 2 + value_dim(cfg); }

int weight_count(const Config& cfg) {
  Config c = cfg;
  if (c.conv_kernel < 1) c.conv_kernel = 4;
  const int cd = conv_dim(c);
  const int vd = value_dim(c);
  const int h = c.hidden_size;
  const int nh = c.num_value_heads;
  return cd * h + vd * h + nh * h + nh * h + h * vd + cd * c.conv_kernel + nh + nh + c.value_head_dim;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "gdn: nil config");
  if (cfg->hidden_size <= 0 || cfg->num_key_heads <= 0 || cfg->num_value_heads <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: need positive HiddenSize/heads");
  }
  if (cfg->key_head_dim <= 0 || cfg->value_head_dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: need positive head dims");
  }
  if (cfg->num_value_heads % cfg->num_key_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: NumValueHeads must divide by NumKeyHeads");
  }
  if (cfg->conv_kernel < 1) cfg->conv_kernel = 4;
  if (cfg->eps <= 0) cfg->eps = 1e-6;
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "gdn: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  const int h = cfg.hidden_size;
  const int cd = conv_dim(cfg);
  const int vd = value_dim(cfg);
  const int nh = cfg.num_value_heads;
  const int k = cfg.conv_kernel;
  const int hd_v = cfg.value_head_dim;

  const int n_qkv = cd * h;
  const int n_z = vd * h;
  const int n_b = nh * h;
  const int n_a = nh * h;
  const int n_out = h * vd;
  const int n_conv = cd * k;

  Layer layer;
  layer.cfg = cfg;
  layer.in_qkv.assign(static_cast<std::size_t>(n_qkv), 0.0f);
  layer.in_z.assign(static_cast<std::size_t>(n_z), 0.0f);
  layer.in_b.assign(static_cast<std::size_t>(n_b), 0.0f);
  layer.in_a.assign(static_cast<std::size_t>(n_a), 0.0f);
  layer.out.assign(static_cast<std::size_t>(n_out), 0.0f);
  layer.conv_weight.assign(static_cast<std::size_t>(n_conv), 0.0f);
  layer.a_log.assign(static_cast<std::size_t>(nh), -1.0f);
  layer.dt_bias.assign(static_cast<std::size_t>(nh), 0.0f);
  layer.norm_gamma.assign(static_cast<std::size_t>(hd_v), 1.0f);

  if (packed_or_null) {
    const float* p = packed_or_null;
    std::memcpy(layer.in_qkv.data(), p, static_cast<std::size_t>(n_qkv) * sizeof(float));
    p += n_qkv;
    std::memcpy(layer.in_z.data(), p, static_cast<std::size_t>(n_z) * sizeof(float));
    p += n_z;
    std::memcpy(layer.in_b.data(), p, static_cast<std::size_t>(n_b) * sizeof(float));
    p += n_b;
    std::memcpy(layer.in_a.data(), p, static_cast<std::size_t>(n_a) * sizeof(float));
    p += n_a;
    std::memcpy(layer.out.data(), p, static_cast<std::size_t>(n_out) * sizeof(float));
    p += n_out;
    std::memcpy(layer.conv_weight.data(), p, static_cast<std::size_t>(n_conv) * sizeof(float));
    p += n_conv;
    std::memcpy(layer.a_log.data(), p, static_cast<std::size_t>(nh) * sizeof(float));
    p += nh;
    std::memcpy(layer.dt_bias.data(), p, static_cast<std::size_t>(nh) * sizeof(float));
    p += nh;
    std::memcpy(layer.norm_gamma.data(), p, static_cast<std::size_t>(hd_v) * sizeof(float));
  }

  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  layer.cfg = cfg;

  if (input.shape.size() != 3 || static_cast<int>(input.shape[2]) != cfg.hidden_size) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: need [B,T,H]");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int t = static_cast<int>(input.shape[1]);
  const int h = cfg.hidden_size;
  if (batch <= 0 || t <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: bad batch/seq");
  }
  const int want = batch * t * h;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: data short");
  }

  TensorF32 out;
  out.shape = {batch, t, h};
  out.data.assign(static_cast<std::size_t>(want), 0.0f);

  Scratch scratch;
  for (int bi = 0; bi < batch; ++bi) {
    ensure_scratch(scratch, cfg); // Reset state per batch
    for (int ti = 0; ti < t; ++ti) {
      const int base = (bi * t + ti) * h;
      if (auto st = forward_decode(layer, scratch, input.data.data() + base, out.data.data() + base);
          !st) {
        return st;
      }
    }
  }

  *pre = out;
  *post = out;
  return Status::Ok();
}


int grad_w_size(const Layer& layer) { return weight_count(layer.cfg); }

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& /*pre*/, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: Backward nil");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  layer.cfg = cfg;

  if (input.shape.size() != 3 || static_cast<int>(input.shape[2]) != cfg.hidden_size) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: need [B,T,H]");
  }
  const int b = static_cast<int>(input.shape[0]);
  const int t = static_cast<int>(input.shape[1]);
  const int h = cfg.hidden_size;
  if (grad_out.shape.size() != 3 || static_cast<int>(grad_out.shape[0]) != b ||
      static_cast<int>(grad_out.shape[1]) != t || static_cast<int>(grad_out.shape[2]) != h) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: gradOut shape != input");
  }

  GdnGrads grads = new_grads(cfg);
  grad_in->shape = {b, t, h};
  grad_in->data.assign(static_cast<std::size_t>(b * t * h), 0.0f);

  Scratch scratch;
  std::vector<GdnStep> steps;
  steps.resize(static_cast<std::size_t>(t));
  std::vector<float> y_scratch(static_cast<std::size_t>(h), 0.0f);

  for (int bi = 0; bi < b; ++bi) {
    ensure_scratch(scratch, cfg);
    for (int ti = 0; ti < t; ++ti) {
      const int base = (bi * t + ti) * h;
      if (auto st = forward_decode_tape(layer, scratch, input.data.data() + base, y_scratch.data(),
                                        &steps[static_cast<std::size_t>(ti)]);
          !st) {
        return st;
      }
    }
    for (int ti = t - 1; ti >= 0; --ti) {
      const int base = (bi * t + ti) * h;
      if (auto st = backward_step(layer, steps[static_cast<std::size_t>(ti)],
                                  grad_out.data.data() + base, &grads, grad_in->data.data() + base);
          !st) {
        return st;
      }
    }
  }

  const int need = grad_w_size(layer);
  grad_w->shape = {need};
  grad_w->data.assign(static_cast<std::size_t>(need), 0.0f);
  flatten_grads(grads, grad_w->data.data());
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: ApplyGradSGD nil");
  }
  if (lr == 0.0 || dW.data.empty()) return Status::Ok();
  const int need = grad_w_size(*layer);
  if (static_cast<int>(dW.data.size()) < need) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: dW short");
  }
  const Config& c = layer->cfg;
  const int cd = conv_dim(c);
  const int vd = value_dim(c);
  const int nh = c.num_value_heads;
  const int h = c.hidden_size;
  const int k = c.conv_kernel < 1 ? 1 : c.conv_kernel;
  int off = 0;
  auto take = [&](int n) -> const float* {
    const float* p = dW.data.data() + off;
    off += n;
    return p;
  };
  auto sgd = [&](std::vector<float>& w, int n) {
    const float* d = take(n);
    for (int i = 0; i < n; ++i) {
      w[static_cast<std::size_t>(i)] -= static_cast<float>(lr) * d[i];
    }
  };
  sgd(layer->in_qkv, cd * h);
  sgd(layer->in_z, vd * h);
  sgd(layer->in_b, nh * h);
  sgd(layer->in_a, nh * h);
  sgd(layer->out, h * vd);
  sgd(layer->conv_weight, cd * k);
  sgd(layer->a_log, nh);
  sgd(layer->dt_bias, nh);
  sgd(layer->norm_gamma, c.value_head_dim);
  return Status::Ok();
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) return Status::Err(StatusCode::InvalidArgument, "gdn: nil export dst");
  const Config& c = layer.cfg;
  const int cd = conv_dim(c);
  const int vd = value_dim(c);
  const int nh = c.num_value_heads;
  const int h = c.hidden_size;
  const int k = c.conv_kernel < 1 ? 1 : c.conv_kernel;
  int off = 0;
  auto put = [&](const std::vector<float>& w) {
    if (!w.empty()) {
      std::memcpy(dst + off, w.data(), w.size() * sizeof(float));
      off += static_cast<int>(w.size());
    }
  };
  put(layer.in_qkv);
  put(layer.in_z);
  put(layer.in_b);
  put(layer.in_a);
  put(layer.out);
  put(layer.conv_weight);
  put(layer.a_log);
  put(layer.dt_bias);
  put(layer.norm_gamma);
  (void)cd;
  (void)vd;
  (void)nh;
  (void)h;
  (void)k;
  return Status::Ok();
}

} // namespace welvet::gdn
