#include "welvet/mha.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::mha {
namespace {

struct Layout {
  int batch = 1;
  int seq_len = 0;
  int d_model = 0;
  int elem_stride = 0;
};

Status parse_layout(int d_model, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "mha: empty input");
  }
  lay->d_model = d_model;
  if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq_len = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != d_model) {
      return Status::Err(StatusCode::InvalidArgument, "mha: last dim != d_model");
    }
  } else if (input.shape.size() == 2) {
    lay->batch = 1;
    if (static_cast<int>(input.shape[1]) != d_model) {
      return Status::Err(StatusCode::InvalidArgument, "mha: width != d_model");
    }
    lay->seq_len = static_cast<int>(input.shape[0]);
  } else {
    return Status::Err(StatusCode::InvalidArgument, "mha: need [batch,seq,d] or [seq,d]");
  }
  if (lay->batch <= 0 || lay->seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: bad batch/seq");
  }
  lay->elem_stride = lay->seq_len * lay->d_model;
  const int want = lay->batch * lay->elem_stride;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "mha: short data");
  }
  return Status::Ok();
}

bool allow(const Config& cfg, int q_pos, int k_pos) {
  switch (cfg.mask) {
  case MaskKind::Bidirectional:
    return true;
  case MaskKind::Causal:
    return k_pos <= q_pos;
  case MaskKind::SlidingWindow: {
    const int w = cfg.window > 0 ? cfg.window : cfg.max_seq_len;
    if (cfg.window_causal || cfg.causal) {
      return k_pos <= q_pos && k_pos >= q_pos - w + 1;
    }
    int d = q_pos - k_pos;
    if (d < 0) d = -d;
    return d < w;
  }
  case MaskKind::PrefixLM: {
    const int pref = cfg.prefix_len;
    if (q_pos < pref && k_pos < pref) return true;
    return k_pos <= q_pos;
  }
  default:
    return k_pos <= q_pos;
  }
}

double alibi_slope(const Config& cfg, int head) {
  int n = cfg.num_heads;
  if (n <= 0) n = 1;
  return std::pow(2.0, -static_cast<double>(head + 1) * 8.0 / static_cast<double>(n));
}

double alibi_bias(const Config& cfg, int head, int q_pos, int k_pos) {
  if (!cfg.uses_alibi()) return 0;
  return -alibi_slope(cfg, head) * std::abs(static_cast<double>(q_pos - k_pos));
}

void apply_rope(double* vec, int pos, int num_heads, int head_dim, double theta) {
  const int half = head_dim / 2;
  if (half <= 0 || theta <= 0) return;
  for (int h = 0; h < num_heads; ++h) {
    const int base = h * head_dim;
    for (int d = 0; d < half; ++d) {
      const double angle =
          static_cast<double>(pos) /
          std::pow(theta, static_cast<double>(2 * d) / static_cast<double>(head_dim));
      const double c = std::cos(angle);
      const double s = std::sin(angle);
      const double v0 = vec[base + d];
      const double v1 = vec[base + d + half];
      vec[base + d] = v0 * c - v1 * s;
      vec[base + d + half] = v0 * s + v1 * c;
    }
  }
}

void attn_forward(const Config& cfg, double* attn_out, const double* Q, const double* cache_k,
                  const double* cache_v, int seq_len, int seq_base, int kv_len, int msl, int q_dim,
                  int kv_dim) {
  const int num_heads = cfg.num_heads;
  const int num_kv = cfg.num_kv_heads;
  const int head_dim = cfg.head_dim;
  const int heads_per_kv = num_heads / num_kv;
  const double scale = cfg.scale();
  const bool use_sigmoid = cfg.softmax == SoftmaxKind::Sigmoid;

  std::vector<double> scores(static_cast<std::size_t>(kv_len));
  std::vector<int> k_positions(static_cast<std::size_t>(kv_len));

  for (int s = 0; s < seq_len; ++s) {
    const int q_pos = seq_base + s;
    for (int h = 0; h < num_heads; ++h) {
      const int kv_head = h / heads_per_kv;
      const int q_off = s * q_dim + h * head_dim;
      const int a_off = s * q_dim + h * head_dim;
      int n = 0;
      double max_score = -1e9;
      for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
        if (!allow(cfg, q_pos, k_pos)) continue;
        const int k_idx = k_pos % msl;
        double dot = 0;
        const int k_base = k_idx * kv_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          dot += Q[q_off + d] * cache_k[k_base + d];
        }
        const double score = dot * scale + alibi_bias(cfg, h, q_pos, k_pos);
        scores[static_cast<std::size_t>(n)] = score;
        k_positions[static_cast<std::size_t>(n)] = k_pos;
        if (score > max_score) max_score = score;
        ++n;
      }
      if (n == 0) {
        for (int d = 0; d < head_dim; ++d) attn_out[a_off + d] = 0;
        continue;
      }
      if (use_sigmoid) {
        for (int i = 0; i < n; ++i) {
          scores[static_cast<std::size_t>(i)] =
              1.0 / (1.0 + std::exp(-scores[static_cast<std::size_t>(i)]));
        }
      } else {
        double exp_sum = 0;
        for (int i = 0; i < n; ++i) {
          scores[static_cast<std::size_t>(i)] =
              std::exp(scores[static_cast<std::size_t>(i)] - max_score);
          exp_sum += scores[static_cast<std::size_t>(i)];
        }
        const double inv = 1.0 / exp_sum;
        for (int i = 0; i < n; ++i) scores[static_cast<std::size_t>(i)] *= inv;
      }
      for (int d = 0; d < head_dim; ++d) {
        double sum = 0;
        for (int i = 0; i < n; ++i) {
          const int v_idx = k_positions[static_cast<std::size_t>(i)] % msl;
          sum += scores[static_cast<std::size_t>(i)] *
                 cache_v[v_idx * kv_dim + kv_head * head_dim + d];
        }
        attn_out[a_off + d] = sum;
      }
    }
  }
}

} // namespace

double Config::scale() const {
  if (scale_override > 0) return scale_override;
  return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "mha: nil config");
  if (cfg->d_model <= 0 || cfg->num_heads <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: need DModel>0 and NumHeads>0");
  }
  if (cfg->head_dim <= 0) {
    if (cfg->d_model % cfg->num_heads != 0) {
      return Status::Err(StatusCode::InvalidArgument, "mha: HeadDim unset and DModel%NumHeads!=0");
    }
    cfg->head_dim = cfg->d_model / cfg->num_heads;
  }
  if (cfg->num_kv_heads <= 0) cfg->num_kv_heads = cfg->num_heads;
  if (cfg->num_heads % cfg->num_kv_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: NumHeads not divisible by NumKVHeads");
  }
  if (cfg->max_seq_len <= 0) cfg->max_seq_len = 512;
  if (cfg->mask == MaskKind::Unspecified) {
    cfg->mask = MaskKind::Causal;
    cfg->causal = true;
  }
  if (cfg->mask == MaskKind::Causal) cfg->causal = true;
  if (cfg->rope_theta <= 0) cfg->rope_theta = 10000;
  if (cfg->softmax != SoftmaxKind::Standard && cfg->softmax != SoftmaxKind::Sigmoid) {
    return Status::Err(StatusCode::InvalidArgument, "mha: unknown Softmax");
  }
  return Status::Ok();
}

Config decoder_causal(int d_model, int num_heads, int num_kv_heads) {
  Config c;
  c.d_model = d_model;
  c.num_heads = num_heads;
  c.num_kv_heads = num_kv_heads;
  c.mask = MaskKind::Causal;
  c.pos = PosKind::RoPE;
  c.mode = AttnMode::Self;
  c.causal = true;
  return c;
}

Status new_configured(const Config& cfg_in, DType dt, const float* q_init, const float* k_init,
                      const float* v_init, const float* o_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "mha: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  const int d = cfg.d_model;
  const int q_dim = cfg.q_dim();
  const int kv_dim = cfg.kv_dim();

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(d, q_dim, Activation::Linear, dt, q_init, &layer.Q); !st) {
    return Status::Err(st.code, std::string("mha Q: ") + st.message);
  }
  if (auto st = dense::new_layer(d, kv_dim, Activation::Linear, dt, k_init, &layer.K); !st) {
    return Status::Err(st.code, std::string("mha K: ") + st.message);
  }
  if (auto st = dense::new_layer(d, kv_dim, Activation::Linear, dt, v_init, &layer.V); !st) {
    return Status::Err(st.code, std::string("mha V: ") + st.message);
  }
  if (auto st = dense::new_layer(q_dim, d, Activation::Linear, dt, o_init, &layer.O); !st) {
    return Status::Err(st.code, std::string("mha O: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "mha: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  if (cfg.mode != AttnMode::Self) {
    return Status::Gap("mha forward: ModeCross not in baseline");
  }

  Layout lay;
  if (auto st = parse_layout(cfg.d_model, input, &lay); !st) return st;

  const int q_dim = cfg.q_dim();
  const int kv_dim = cfg.kv_dim();
  const int msl = cfg.max_seq_len;
  const int d = cfg.d_model;
  const int bs = lay.batch * lay.seq_len;

  if (lay.seq_len > msl) {
    return Status::Err(StatusCode::InvalidArgument, "mha: seq > MaxSeqLen");
  }

  TensorF32 flat_in;
  flat_in.shape = {bs, d};
  flat_in.data.assign(input.data.begin(), input.data.begin() + static_cast<std::size_t>(bs * d));

  TensorF32 q_pre, q_post, k_pre, k_post, v_pre, v_post;
  if (auto st = dense::forward_f32(layer.Q, flat_in, &q_pre, &q_post); !st) {
    return Status::Err(st.code, std::string("mha Q: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.K, flat_in, &k_pre, &k_post); !st) {
    return Status::Err(st.code, std::string("mha K: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.V, flat_in, &v_pre, &v_post); !st) {
    return Status::Err(st.code, std::string("mha V: ") + st.message);
  }

  std::vector<double> cache_k(static_cast<std::size_t>(msl * kv_dim), 0.0);
  std::vector<double> cache_v(static_cast<std::size_t>(msl * kv_dim), 0.0);
  std::vector<double> Q(static_cast<std::size_t>(lay.seq_len * q_dim));
  std::vector<double> attn_out(static_cast<std::size_t>(lay.seq_len * q_dim));
  TensorF32 attn_flat;
  attn_flat.shape = {bs, q_dim};
  attn_flat.data.resize(static_cast<std::size_t>(bs * q_dim));

  for (int b = 0; b < lay.batch; ++b) {
    const int seq_base = b * lay.seq_len;
    for (int s = 0; s < lay.seq_len; ++s) {
      const int pos = seq_base + s;
      const int tok = b * lay.seq_len + s;
      for (int i = 0; i < q_dim; ++i) {
        Q[static_cast<std::size_t>(s * q_dim + i)] =
            static_cast<double>(q_post.data[static_cast<std::size_t>(tok * q_dim + i)]);
      }
      double* k_row = cache_k.data() + (pos % msl) * kv_dim;
      double* v_row = cache_v.data() + (pos % msl) * kv_dim;
      for (int i = 0; i < kv_dim; ++i) {
        k_row[i] = static_cast<double>(k_post.data[static_cast<std::size_t>(tok * kv_dim + i)]);
        v_row[i] = static_cast<double>(v_post.data[static_cast<std::size_t>(tok * kv_dim + i)]);
      }
      if (cfg.uses_rope()) {
        apply_rope(Q.data() + s * q_dim, pos, cfg.num_heads, cfg.head_dim, cfg.rope_theta);
        apply_rope(k_row, pos, cfg.num_kv_heads, cfg.head_dim, cfg.rope_theta);
      }
    }
    const int kv_len = seq_base + lay.seq_len;
    attn_forward(cfg, attn_out.data(), Q.data(), cache_k.data(), cache_v.data(), lay.seq_len,
                 seq_base, kv_len, msl, q_dim, kv_dim);
    for (int s = 0; s < lay.seq_len; ++s) {
      const int tok = b * lay.seq_len + s;
      for (int i = 0; i < q_dim; ++i) {
        attn_flat.data[static_cast<std::size_t>(tok * q_dim + i)] =
            static_cast<float>(attn_out[static_cast<std::size_t>(s * q_dim + i)]);
      }
    }
  }

  TensorF32 o_pre, o_post;
  if (auto st = dense::forward_f32(layer.O, attn_flat, &o_pre, &o_post); !st) {
    return Status::Err(st.code, std::string("mha O: ") + st.message);
  }

  pre->shape = {lay.batch, lay.seq_len, q_dim};
  pre->data = attn_flat.data;
  post->shape = {lay.batch, lay.seq_len, d};
  post->data = o_post.data;
  return Status::Ok();
}

} // namespace welvet::mha
