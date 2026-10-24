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

float silu(float x) {
  return x / (1.0f + static_cast<float>(std::exp(static_cast<double>(-x))));
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

} // namespace welvet::gdn
