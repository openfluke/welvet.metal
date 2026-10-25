namespace gdn_gpu {
struct Config {
  int hidden_size, num_key_heads, num_value_heads, key_head_dim, value_head_dim, conv_kernel;
  float eps;
};
struct Layer {
  Config cfg;
  device float *in_qkv, *in_z, *in_b, *in_a, *out, *conv_weight, *a_log, *dt_bias, *norm_gamma;
};
struct Arena {
  device float *data;
  uint used;
};
inline device float *alloc(thread Arena &arena, uint n) {
  device float *p = arena.data + arena.used;
  arena.used += n;
  for (uint i = 0; i < n; ++i)
    p[i] = 0;
  return p;
}
inline device float *clone(thread Arena &arena, device const float *src, uint n) {
  device float *p = alloc(arena, n);
  for (uint i = 0; i < n; ++i)
    p[i] = src[i];
  return p;
}
inline void gpu_copy(device float *dst, device const float *src, uint bytes) {
  for (uint i = 0; i < bytes / 4; ++i)
    dst[i] = src[i];
}
inline void gpu_move(device float *dst, device const float *src, uint bytes) {
  for (uint i = 0; i < bytes / 4; ++i)
    dst[i] = src[i];
}
inline void gpu_zero(device float *dst, int value, uint bytes) {
  for (uint i = 0; i < bytes / 4; ++i)
    dst[i] = 0;
}
inline int key_dim(thread const Config &c) { return c.num_key_heads * c.key_head_dim; }
inline int value_dim(thread const Config &c) { return c.num_value_heads * c.value_head_dim; }
inline int conv_dim(thread const Config &c) { return 2 * key_dim(c) + value_dim(c); }
void matvec_f32(device const float *w, int rows, int cols, device const float *x, device float *y) {
  for (int r = 0; r < rows; ++r) {
    float sum = 0.0f;
    device const float *row = w + r * cols;
    for (int c = 0; c < cols; ++c) {
      sum += row[c] * x[c];
    }
    y[r] = sum;
  }
}

void matvec_t_f32(device const float *w, int rows, int cols, device const float *gy,
                  device float *gx) {
  for (int r = 0; r < rows; ++r) {
    const float g = gy[r];
    device const float *row = w + r * cols;
    for (int c = 0; c < cols; ++c) {
      gx[c] += row[c] * g;
    }
  }
}

float silu(float x) { return x / (1.0f + static_cast<float>(exp(static_cast<float>(-x)))); }

float silu_deriv(float x) {
  const float s = 1.0f / (1.0f + static_cast<float>(exp(static_cast<float>(-x))));
  return s + x * s * (1.0f - s);
}

float softplus(float x) {
  if (x > 20.0)
    return x;
  return log(1.0f + exp(x));
}

void rms_norm_gated(device float *x, device const float *z, device const float *gamma, int n,
                    float eps) {
  float mean = 0.0f;
  for (int i = 0; i < n; ++i)
    mean += x[i] * x[i];
  mean /= static_cast<float>(n);
  const float inv = 1.0f / static_cast<float>(sqrt(static_cast<float>(mean + eps)));
  for (int i = 0; i < n; ++i) {
    const float g = (gamma) ? gamma[i] : 1.0f;
    x[i] = x[i] * inv * g * silu(z[i]);
  }
}

// Match Go l2normBackward (float64 round-trip on sum).
void l2norm_backward(device const float *x_raw, int n, float inv, device const float *dy,
                     device float *dx) {
  float s = 0.0;
  for (int i = 0; i < n; ++i) {
    const float y = x_raw[i] * inv;
    s += static_cast<float>(dy[i]) * static_cast<float>(y);
  }
  const float sf = static_cast<float>(s);
  for (int j = 0; j < n; ++j) {
    const float y = x_raw[j] * inv;
    dx[j] = inv * (dy[j] - y * sf);
  }
}

void accum_outer(device float *dW, device const float *gy, device const float *x, int rows,
                 int cols) {
  for (int r = 0; r < rows; ++r) {
    const float gr = gy[r];
    if (gr == 0.0f)
      continue;
    device float *base = dW + r * cols;
    for (int c = 0; c < cols; ++c) {
      base[c] += gr * x[c];
    }
  }
}

struct Scratch {
  device float *qkv;
  device float *z;
  device float *beta_raw;
  device float *a_raw;
  device float *mixed;
  device float *q_rep;
  device float *k_rep;
  device float *beta;
  device float *g;
  device float *core;
  device float *kv_mem;
  device float *delta;
  device float *conv_state;
  device float *state;
};

void ensure_scratch(thread Arena &arena, thread Scratch &s, thread const Config &c) {
  const int cd = conv_dim(c);
  const int vd = value_dim(c);
  const int nh = c.num_value_heads;
  const int hd_k = c.key_head_dim;
  const int hd_v = c.value_head_dim;
  const int k = c.conv_kernel < 1 ? 1 : c.conv_kernel;
  s.qkv = alloc(arena, cd);
  s.z = alloc(arena, vd);
  s.beta_raw = alloc(arena, nh);
  s.a_raw = alloc(arena, nh);
  s.mixed = alloc(arena, cd);
  s.q_rep = alloc(arena, nh * hd_k);
  s.k_rep = alloc(arena, nh * hd_k);
  s.beta = alloc(arena, nh);
  s.g = alloc(arena, nh);
  s.core = alloc(arena, nh * hd_v);
  s.kv_mem = alloc(arena, hd_v);
  s.delta = alloc(arena, hd_v);
  s.conv_state = alloc(arena, cd * (k - 1));
  s.state = alloc(arena, nh * hd_k * hd_v);
}

struct GdnStep {
  device float *x;         // H
  device float *qkv;       // convDim
  device float *conv_hist; // convDim*(k-1)
  device float *acc;       // convDim
  device float *mixed;     // convDim
  device float *q_inv;
  device float *k_inv; // nk
  device float *q_rep;
  device float *k_rep; // nh*hdK
  device float *beta_raw;
  device float *a_raw; // nh
  device float *beta;
  device float *g;                 // nh
  device float *state_old;         // nh*hdK*hdV
  device float *state_after_decay; // nh*hdK*hdV
  device float *kv_mem;            // valDim
  device float *delta;             // valDim
  device float *core_pre;          // valDim
  device float *z;                 // valDim
  device float *core_normed;       // valDim
};

struct GdnGrads {
  device float *d_in_qkv;
  device float *d_in_z;
  device float *d_in_b;
  device float *d_in_a;
  device float *d_out;
  device float *d_conv_weight;
  device float *d_a_log;
  device float *d_dt_bias;
  device float *d_norm_gamma;
};

void forward_decode_tape(thread Arena &arena, thread Layer &layer, thread Scratch &s,
                         device const float *x, device float *y, thread GdnStep *step) {
  thread const Config &c = layer.cfg;
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
  tp.x = clone(arena, x, h);
  tp.qkv = alloc(arena, conv_d);
  tp.z = alloc(arena, val_d);
  tp.beta_raw = alloc(arena, nh);
  tp.a_raw = alloc(arena, nh);

  matvec_f32(layer.in_qkv, conv_d, h, x, tp.qkv);
  matvec_f32(layer.in_z, val_d, h, x, tp.z);
  matvec_f32(layer.in_b, nh, h, x, tp.beta_raw);
  matvec_f32(layer.in_a, nh, h, x, tp.a_raw);

  tp.conv_hist = clone(arena, s.conv_state, conv_d * (k - 1));
  tp.acc = alloc(arena, conv_d);
  tp.mixed = alloc(arena, conv_d);
  for (int ch = 0; ch < conv_d; ++ch) {
    float a = 0.0f;
    for (int t = 0; t < k - 1; ++t) {
      a += layer.conv_weight[static_cast<uint>(ch * k + t)] *
           s.conv_state[static_cast<uint>(ch * (k - 1) + t)];
    }
    a += layer.conv_weight[static_cast<uint>(ch * k + (k - 1))] * tp.qkv[static_cast<uint>(ch)];
    tp.acc[static_cast<uint>(ch)] = a;
    tp.mixed[static_cast<uint>(ch)] = silu(a);
  }
  for (int ch = 0; ch < conv_d; ++ch) {
    const int base = ch * (k - 1);
    if (k > 2) {
      gpu_move(s.conv_state + base, s.conv_state + base + 1,
               static_cast<uint>(k - 2) * sizeof(float));
    }
    if (k > 1) {
      s.conv_state[static_cast<uint>(base + k - 2)] = tp.qkv[static_cast<uint>(ch)];
    }
  }

  device const float *q = tp.mixed;
  device const float *key = tp.mixed + key_d;
  device const float *v = tp.mixed + key_d * 2;

  tp.q_rep = alloc(arena, nh * hd_k);
  tp.k_rep = alloc(arena, nh * hd_k);
  tp.q_inv = alloc(arena, nk);
  tp.k_inv = alloc(arena, nk);
  const float scale = static_cast<float>(1.0 / sqrt(static_cast<float>(hd_k)));
  for (int hi = 0; hi < nk; ++hi) {
    device const float *qs = q + hi * hd_k;
    device const float *ks = key + hi * hd_k;
    float sq = 0.0, sk = 0.0;
    for (int j = 0; j < hd_k; ++j) {
      const float vq = static_cast<float>(qs[j]);
      sq += vq * vq;
    }
    for (int j = 0; j < hd_k; ++j) {
      const float vk = static_cast<float>(ks[j]);
      sk += vk * vk;
    }
    const float invq = static_cast<float>(1.0 / sqrt(sq + 1e-6));
    const float invk = static_cast<float>(1.0 / sqrt(sk + 1e-6));
    tp.q_inv[static_cast<uint>(hi)] = invq;
    tp.k_inv[static_cast<uint>(hi)] = invk;
    for (int r = 0; r < rep; ++r) {
      const int dst = (hi * rep + r) * hd_k;
      for (int j = 0; j < hd_k; ++j) {
        tp.q_rep[static_cast<uint>(dst + j)] = qs[j] * invq * scale;
        tp.k_rep[static_cast<uint>(dst + j)] = ks[j] * invk;
      }
    }
  }

  tp.beta = alloc(arena, nh);
  tp.g = alloc(arena, nh);
  for (int i = 0; i < nh; ++i) {
    tp.beta[static_cast<uint>(i)] =
        1.0f /
        (1.0f + static_cast<float>(exp(static_cast<float>(-tp.beta_raw[static_cast<uint>(i)]))));
    const float al = static_cast<float>(layer.a_log[static_cast<uint>(i)]);
    const float dt =
        static_cast<float>(tp.a_raw[static_cast<uint>(i)] + layer.dt_bias[static_cast<uint>(i)]);
    tp.g[static_cast<uint>(i)] = static_cast<float>(-exp(al) * softplus(dt));
  }

  tp.state_old = s.state;
  device float *state = clone(arena, s.state, nh * hd_k * hd_v);
  tp.state_after_decay = alloc(arena, nh * hd_k * hd_v);
  tp.kv_mem = alloc(arena, val_d);
  tp.delta = alloc(arena, val_d);
  tp.core_pre = alloc(arena, val_d);

  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    device float *st = state + h_idx * hd_k * hd_v;
    const float gt = static_cast<float>(exp(static_cast<float>(tp.g[static_cast<uint>(h_idx)])));
    const float bt = tp.beta[static_cast<uint>(h_idx)];
    device const float *qt = tp.q_rep + h_idx * hd_k;
    device const float *kt = tp.k_rep + h_idx * hd_k;
    device const float *vt = v + h_idx * hd_v;

    for (int i = 0; i < hd_k * hd_v; ++i)
      st[i] *= gt;
    gpu_copy(tp.state_after_decay + h_idx * hd_k * hd_v, st,
             static_cast<uint>(hd_k * hd_v) * sizeof(float));

    device float *kvm = tp.kv_mem + h_idx * hd_v;
    for (int d = 0; d < hd_v; ++d) {
      float sum = 0.0f;
      for (int j = 0; j < hd_k; ++j)
        sum += st[j * hd_v + d] * kt[j];
      kvm[d] = sum;
    }
    device float *dl = tp.delta + h_idx * hd_v;
    for (int d = 0; d < hd_v; ++d) {
      dl[d] = (vt[d] - kvm[d]) * bt;
    }
    for (int j = 0; j < hd_k; ++j) {
      for (int d = 0; d < hd_v; ++d) {
        st[j * hd_v + d] += kt[j] * dl[d];
      }
    }
    device float *out_h = tp.core_pre + h_idx * hd_v;
    for (int d = 0; d < hd_v; ++d) {
      float sum = 0.0f;
      for (int j = 0; j < hd_k; ++j)
        sum += st[j * hd_v + d] * qt[j];
      out_h[d] = sum;
    }
  }
  s.state = state;

  tp.core_normed = clone(arena, tp.core_pre, val_d);
  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    device float *o = tp.core_normed + h_idx * hd_v;
    device const float *zz = tp.z + h_idx * hd_v;
    rms_norm_gated(o, zz, layer.norm_gamma, hd_v, static_cast<float>(c.eps));
  }

  matvec_f32(layer.out, h, val_d, tp.core_normed, y);
  *step = tp;
  return;
}

void backward_step(thread Arena &arena, thread Layer &layer, thread const GdnStep &tp,
                   device const float *dy, thread GdnGrads *g, device float *dx) {
  thread const Config &c = layer.cfg;
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

  device float *d_core_normed = alloc(arena, val_d);
  matvec_t_f32(layer.out, h, val_d, dy, d_core_normed);
  accum_outer(g->d_out, dy, tp.core_normed, h, val_d);

  device float *d_core_pre = alloc(arena, val_d);
  device float *d_z = alloc(arena, val_d);
  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    device const float *xr = tp.core_pre + h_idx * hd_v;
    device const float *zz = tp.z + h_idx * hd_v;
    device const float *dyh = d_core_normed + h_idx * hd_v;
    const int n = hd_v;
    float mean = 0.0f;
    for (int i = 0; i < n; ++i)
      mean += xr[i] * xr[i];
    mean /= static_cast<float>(n);
    const float inv =
        1.0f / static_cast<float>(sqrt(static_cast<float>(mean + static_cast<float>(c.eps))));
    float s = 0.0;
    for (int i = 0; i < n; ++i) {
      const float gamma =
          (i < static_cast<int>(hd_v)) ? layer.norm_gamma[static_cast<uint>(i)] : 1.0f;
      const float sz = silu(zz[i]);
      const float ci = gamma * sz;
      s += static_cast<float>(dyh[i]) * static_cast<float>(ci) * static_cast<float>(xr[i]);
    }
    const float sf = static_cast<float>(s);
    for (int i = 0; i < n; ++i) {
      const float gamma =
          (i < static_cast<int>(hd_v)) ? layer.norm_gamma[static_cast<uint>(i)] : 1.0f;
      const float sz = silu(zz[i]);
      const float ci = gamma * sz;
      d_core_pre[static_cast<uint>(h_idx * hd_v + i)] =
          inv * ci * dyh[i] - inv * inv * inv / static_cast<float>(n) * xr[i] * sf;
      d_z[static_cast<uint>(h_idx * hd_v + i)] = dyh[i] * xr[i] * inv * gamma * silu_deriv(zz[i]);
      if (i < static_cast<int>(hd_v)) {
        g->d_norm_gamma[static_cast<uint>(i)] += dyh[i] * xr[i] * inv * sz;
      }
    }
  }

  device float *d_mixed = alloc(arena, conv_d);
  device float *d_beta_raw = alloc(arena, nh);
  device float *d_a_raw = alloc(arena, nh);
  device float *d_q_rep = alloc(arena, nh * hd_k);
  device float *d_k_rep = alloc(arena, nh * hd_k);

  for (int h_idx = 0; h_idx < nh; ++h_idx) {
    device const float *dout_h = d_core_pre + h_idx * hd_v;
    device const float *st_after = tp.state_after_decay + h_idx * hd_k * hd_v;
    device const float *st_old = tp.state_old + h_idx * hd_k * hd_v;
    device const float *dl = tp.delta + h_idx * hd_v;
    device const float *qt = tp.q_rep + h_idx * hd_k;
    device const float *kt = tp.k_rep + h_idx * hd_k;
    device const float *v = tp.mixed + key_d * 2 + h_idx * hd_v;

    device float *d_state_final = alloc(arena, hd_k * hd_v);
    device float *dqt = alloc(arena, hd_k);
    for (int j = 0; j < hd_k; ++j) {
      float sum = 0.0f;
      for (int d = 0; d < hd_v; ++d) {
        const float st_final = st_after[j * hd_v + d] + kt[j] * dl[d];
        d_state_final[static_cast<uint>(j * hd_v + d)] = dout_h[d] * qt[j];
        sum += st_final * dout_h[d];
      }
      dqt[static_cast<uint>(j)] = sum;
    }
    device float *dkt = alloc(arena, hd_k);
    device float *ddelta = alloc(arena, hd_v);
    device float *d_state_after = alloc(arena, hd_k * hd_v);
    for (int j = 0; j < hd_k; ++j) {
      float dktj = 0.0f;
      for (int d = 0; d < hd_v; ++d) {
        const float dsf = d_state_final[static_cast<uint>(j * hd_v + d)];
        d_state_after[static_cast<uint>(j * hd_v + d)] += dsf;
        dktj += dl[d] * dsf;
        ddelta[static_cast<uint>(d)] += kt[j] * dsf;
      }
      dkt[static_cast<uint>(j)] += dktj;
    }
    const float bt = tp.beta[static_cast<uint>(h_idx)];
    device const float *kvm = tp.kv_mem + h_idx * hd_v;
    float dbt = 0.0f;
    device float *d_kv_mem = alloc(arena, hd_v);
    for (int d = 0; d < hd_v; ++d) {
      const float ddl = ddelta[static_cast<uint>(d)];
      dbt += ddl * (v[d] - kvm[d]);
      d_kv_mem[static_cast<uint>(d)] = -ddl * bt;
      d_mixed[static_cast<uint>(key_d * 2 + h_idx * hd_v + d)] += ddl * bt;
    }
    for (int j = 0; j < hd_k; ++j) {
      float dktj = 0.0f;
      for (int d = 0; d < hd_v; ++d) {
        d_state_after[static_cast<uint>(j * hd_v + d)] += d_kv_mem[static_cast<uint>(d)] * kt[j];
        dktj += d_kv_mem[static_cast<uint>(d)] * st_after[j * hd_v + d];
      }
      dkt[static_cast<uint>(j)] += dktj;
    }

    const float gt = static_cast<float>(exp(static_cast<float>(tp.g[static_cast<uint>(h_idx)])));
    float dgt = 0.0f;
    for (int j = 0; j < hd_k; ++j) {
      for (int d = 0; d < hd_v; ++d) {
        dgt += d_state_after[static_cast<uint>(j * hd_v + d)] * st_old[j * hd_v + d];
      }
    }
    const float d_gate = dgt * gt;
    const float al = static_cast<float>(layer.a_log[static_cast<uint>(h_idx)]);
    const float dt_raw = static_cast<float>(tp.a_raw[static_cast<uint>(h_idx)] +
                                            layer.dt_bias[static_cast<uint>(h_idx)]);
    const float sig = static_cast<float>(1.0 / (1.0 + exp(-dt_raw)));
    const float d_dt_raw = d_gate * static_cast<float>(-exp(al)) * sig;
    d_a_raw[static_cast<uint>(h_idx)] += d_dt_raw;
    g->d_dt_bias[static_cast<uint>(h_idx)] += d_dt_raw;
    g->d_a_log[static_cast<uint>(h_idx)] += d_gate * tp.g[static_cast<uint>(h_idx)];

    d_beta_raw[static_cast<uint>(h_idx)] += dbt * bt * (1.0f - bt);

    for (int j = 0; j < hd_k; ++j) {
      d_q_rep[static_cast<uint>(h_idx * hd_k + j)] = dqt[static_cast<uint>(j)];
      d_k_rep[static_cast<uint>(h_idx * hd_k + j)] = dkt[static_cast<uint>(j)];
    }
  }

  const float scale = static_cast<float>(1.0 / sqrt(static_cast<float>(hd_k)));
  device float *dq = alloc(arena, key_d);
  device float *dk = alloc(arena, key_d);
  for (int hi = 0; hi < nk; ++hi) {
    device float *dq_norm = alloc(arena, hd_k);
    device float *dk_norm = alloc(arena, hd_k);
    for (int r = 0; r < rep; ++r) {
      const int h_idx = hi * rep + r;
      for (int j = 0; j < hd_k; ++j) {
        dq_norm[static_cast<uint>(j)] += d_q_rep[static_cast<uint>(h_idx * hd_k + j)] * scale;
        dk_norm[static_cast<uint>(j)] += d_k_rep[static_cast<uint>(h_idx * hd_k + j)];
      }
    }
    device const float *q_raw = tp.mixed + hi * hd_k;
    device const float *k_raw = tp.mixed + key_d + hi * hd_k;
    device float *dq_raw = alloc(arena, hd_k);
    device float *dk_raw = alloc(arena, hd_k);
    l2norm_backward(q_raw, hd_k, tp.q_inv[static_cast<uint>(hi)], dq_norm, dq_raw);
    l2norm_backward(k_raw, hd_k, tp.k_inv[static_cast<uint>(hi)], dk_norm, dk_raw);
    gpu_copy(dq + hi * hd_k, dq_raw, static_cast<uint>(hd_k) * sizeof(float));
    gpu_copy(dk + hi * hd_k, dk_raw, static_cast<uint>(hd_k) * sizeof(float));
  }
  gpu_copy(d_mixed, dq, static_cast<uint>(key_d) * sizeof(float));
  for (int i = 0; i < key_d; ++i) {
    d_mixed[static_cast<uint>(key_d + i)] += dk[static_cast<uint>(i)];
  }

  device float *d_qkv = alloc(arena, conv_d);
  for (int ch = 0; ch < conv_d; ++ch) {
    const float d_acc = d_mixed[static_cast<uint>(ch)] * silu_deriv(tp.acc[static_cast<uint>(ch)]);
    for (int t = 0; t < k - 1; ++t) {
      g->d_conv_weight[static_cast<uint>(ch * k + t)] +=
          d_acc * tp.conv_hist[static_cast<uint>(ch * (k - 1) + t)];
    }
    g->d_conv_weight[static_cast<uint>(ch * k + (k - 1))] += d_acc * tp.qkv[static_cast<uint>(ch)];
    d_qkv[static_cast<uint>(ch)] += d_acc * layer.conv_weight[static_cast<uint>(ch * k + (k - 1))];
  }

  gpu_zero(dx, 0, static_cast<uint>(h) * sizeof(float));
  matvec_t_f32(layer.in_qkv, conv_d, h, d_qkv, dx);
  accum_outer(g->d_in_qkv, d_qkv, tp.x, conv_d, h);

  matvec_t_f32(layer.in_z, val_d, h, d_z, dx);
  accum_outer(g->d_in_z, d_z, tp.x, val_d, h);

  matvec_t_f32(layer.in_b, nh, h, d_beta_raw, dx);
  accum_outer(g->d_in_b, d_beta_raw, tp.x, nh, h);

  matvec_t_f32(layer.in_a, nh, h, d_a_raw, dx);
  accum_outer(g->d_in_a, d_a_raw, tp.x, nh, h);

  return;
}

} // namespace gdn_gpu
kernel void op_gdn(OP_ARGS) {
  if (tid)
    return;
  using namespace gdn_gpu;
  Config cfg = {c[0], c[1], c[2], c[3], c[4], c[5], r[0]};
  Layer layer;
  layer.cfg = cfg;
  int H = cfg.hidden_size, CD = conv_dim(cfg), VD = value_dim(cfg), NH = cfg.num_value_heads,
      K = cfg.conv_kernel, HV = cfg.value_head_dim;
  uint offset = 0;
  layer.in_qkv = b + offset;
  offset += CD * H;
  layer.in_z = b + offset;
  offset += VD * H;
  layer.in_b = b + offset;
  offset += NH * H;
  layer.in_a = b + offset;
  offset += NH * H;
  layer.out = b + offset;
  offset += H * VD;
  layer.conv_weight = b + offset;
  offset += CD * K;
  layer.a_log = b + offset;
  offset += NH;
  layer.dt_bias = b + offset;
  offset += NH;
  layer.norm_gamma = b + offset;
  GdnGrads grads;
  uint goff = 0;
  grads.d_in_qkv = g + goff;
  goff += CD * H;
  grads.d_in_z = g + goff;
  goff += VD * H;
  grads.d_in_b = g + goff;
  goff += NH * H;
  grads.d_in_a = g + goff;
  goff += NH * H;
  grads.d_out = g + goff;
  goff += H * VD;
  grads.d_conv_weight = g + goff;
  goff += CD * K;
  grads.d_a_log = g + goff;
  goff += NH;
  grads.d_dt_bias = g + goff;
  goff += NH;
  grads.d_norm_gamma = g + goff;
  Arena arena = {h, 0};
  // Truncated BPTT: recompute the prefix for each reverse token, preserving the
  // original state/conv stop-gradient contract without a host tape.
  for (int batch = 0; batch < c[6]; ++batch) {
    if (!c[8]) {
      arena.used = 0;
      Scratch sc;
      ensure_scratch(arena, sc, cfg);
      for (int t = 0; t < c[7]; ++t) {
        GdnStep step;
        forward_decode_tape(arena, layer, sc, a + (batch * c[7] + t) * H,
                            e + (batch * c[7] + t) * H, &step);
      }
    } else
      for (int target = c[7] - 1; target >= 0; --target) {
        arena.used = 0;
        Scratch sc;
        ensure_scratch(arena, sc, cfg);
        GdnStep step;
        device float *ys = alloc(arena, H);
        for (int t = 0; t <= target; ++t)
          forward_decode_tape(arena, layer, sc, a + (batch * c[7] + t) * H, ys, &step);
        backward_step(arena, layer, step, d + (batch * c[7] + target) * H, &grads,
                      f + (batch * c[7] + target) * H);
      }
  }
}

kernel void op_gdn_sgd(OP_ARGS) {
  if (tid < uint(c[0]))
    a[tid] = a[tid] - r[0] * b[tid];
}
