#include "welvet/mamba.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::mamba {
namespace {

double softplus(double x) {
  if (x > 20.0) return x;
  return std::log1p(std::exp(x));
}

} // namespace

int inner_dim(const Config& cfg) {
  int e = cfg.expand;
  if (e <= 0) e = 2;
  return e * cfg.d_model;
}

int weight_count(const Config& cfg) {
  Config c = cfg;
  if (c.expand <= 0) c.expand = 2;
  const int inner = c.expand * c.d_model;
  return inner * (3 * c.d_model + 2);
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "mamba: nil config");
  if (cfg->d_model <= 0 || cfg->d_state <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: need positive DModel/DState");
  }
  if (cfg->expand <= 0) cfg->expand = 2;
  if (cfg->seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: need positive SeqLen");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "mamba: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  const int d = cfg.d_model;
  const int inner = inner_dim(cfg);
  const int in_n = 2 * inner * d;
  const int out_n = d * inner;

  Layer layer;
  layer.cfg = cfg;

  const float* in_init = nullptr;
  const float* out_init = nullptr;
  if (packed_or_null) {
    in_init = packed_or_null;
    out_init = packed_or_null + in_n;
  }
  if (auto st = dense::new_layer(d, 2 * inner, Activation::Linear, dt, in_init, &layer.in_proj); !st) {
    return Status::Err(st.code, std::string("mamba in: ") + st.message);
  }
  if (auto st = dense::new_layer(inner, d, Activation::Linear, dt, out_init, &layer.out_proj); !st) {
    return Status::Err(st.code, std::string("mamba out: ") + st.message);
  }

  layer.a_log.resize(static_cast<std::size_t>(inner));
  layer.d_skip.resize(static_cast<std::size_t>(inner));
  if (packed_or_null) {
    const float* a = packed_or_null + in_n + out_n;
    const float* ds = a + inner;
    for (int i = 0; i < inner; ++i) {
      layer.a_log[static_cast<std::size_t>(i)] = a[i];
      layer.d_skip[static_cast<std::size_t>(i)] = ds[i];
    }
  } else {
    for (int i = 0; i < inner; ++i) {
      layer.a_log[static_cast<std::size_t>(i)] = -static_cast<float>(i + 1) * 0.1f;
      layer.d_skip[static_cast<std::size_t>(i)] = 1.0f;
    }
  }

  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  if (input.shape.size() != 3) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: need [batch,seq,dmodel]");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int t = static_cast<int>(input.shape[1]);
  const int d = static_cast<int>(input.shape[2]);
  if (t != cfg.seq_len || d != cfg.d_model) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: shape mismatch vs config");
  }
  if (batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: bad batch");
  }
  const int inner = inner_dim(cfg);
  const int n = batch * t;
  const int want = n * d;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: data short");
  }
  if (static_cast<int>(layer.a_log.size()) != inner ||
      static_cast<int>(layer.d_skip.size()) != inner) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: ALog/DSkip size");
  }

  TensorF32 flat;
  flat.shape = {n, d};
  flat.data.assign(input.data.begin(), input.data.begin() + static_cast<std::ptrdiff_t>(want));


  TensorF32 z_pre, z;
  if (auto st = dense::forward_f32(layer.in_proj, flat, &z_pre, &z); !st) {
    return Status::Err(st.code, std::string("mamba in: ") + st.message);
  }

  TensorF32 scanned;
  scanned.shape = {n, inner};
  scanned.data.resize(static_cast<std::size_t>(n * inner));

  for (int b = 0; b < batch; ++b) {
    std::vector<double> h(static_cast<std::size_t>(inner), 0.0);
    for (int ti = 0; ti < t; ++ti) {
      const int row = (b * t + ti) * (2 * inner);
      const int out_row = (b * t + ti) * inner;
      for (int i = 0; i < inner; ++i) {
        const double x = static_cast<double>(z.data[static_cast<std::size_t>(row + i)]);
        const double dt =
            softplus(static_cast<double>(z.data[static_cast<std::size_t>(row + inner + i)]));
        const double a =
            std::exp(-dt * std::exp(static_cast<double>(layer.a_log[static_cast<std::size_t>(i)])));
        h[static_cast<std::size_t>(i)] = a * h[static_cast<std::size_t>(i)] + dt * x;
        const double y =
            h[static_cast<std::size_t>(i)] +
            static_cast<double>(layer.d_skip[static_cast<std::size_t>(i)]) * x;
        scanned.data[static_cast<std::size_t>(out_row + i)] = static_cast<float>(y);
      }
    }
  }

  TensorF32 pre_flat, post_flat;
  if (auto st = dense::forward_f32(layer.out_proj, scanned, &pre_flat, &post_flat); !st) {
    return Status::Err(st.code, std::string("mamba out: ") + st.message);
  }

  pre->shape = {batch, t, d};
  post->shape = {batch, t, d};
  const int out_n = n * d;
  pre->data.assign(pre_flat.data.begin(),
                   pre_flat.data.begin() + static_cast<std::ptrdiff_t>(out_n));
  post->data.assign(post_flat.data.begin(),
                    post_flat.data.begin() + static_cast<std::ptrdiff_t>(out_n));
  return Status::Ok();
}

} // namespace welvet::mamba
