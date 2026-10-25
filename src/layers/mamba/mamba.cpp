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

int grad_w_size(const Layer& layer) {
  const int in_n = layer.in_proj.weights.rows * layer.in_proj.weights.cols;
  const int out_n = layer.out_proj.weights.rows * layer.out_proj.weights.cols;
  return in_n + out_n + static_cast<int>(layer.a_log.size()) +
         static_cast<int>(layer.d_skip.size());
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& /*pre*/, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: nil grad out");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  if (input.shape.size() != 3 || grad_out.shape.size() != 3) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: need [batch,seq,dmodel]");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int t = static_cast<int>(input.shape[1]);
  const int d = static_cast<int>(input.shape[2]);
  if (t != cfg.seq_len || d != cfg.d_model) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: shape mismatch vs config");
  }
  if (static_cast<int>(grad_out.shape[0]) != batch || static_cast<int>(grad_out.shape[1]) != t ||
      static_cast<int>(grad_out.shape[2]) != d) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: gradOut shape");
  }
  const int inner = inner_dim(cfg);
  const int n = batch * t;
  const int want = n * d;
  if (static_cast<int>(input.data.size()) < want || static_cast<int>(grad_out.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: data short");
  }
  if (static_cast<int>(layer.a_log.size()) != inner ||
      static_cast<int>(layer.d_skip.size()) != inner) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: ALog/DSkip size");
  }

  TensorF32 flat;
  flat.shape = {n, d};
  flat.data.assign(input.data.begin(), input.data.begin() + static_cast<std::ptrdiff_t>(want));

  TensorF32 in_pre, z;
  if (auto st = dense::forward_f32(layer.in_proj, flat, &in_pre, &z); !st) {
    return Status::Err(st.code, std::string("mamba in: ") + st.message);
  }

  struct Step {
    double h = 0.0;
    double x = 0.0;
    double dt = 0.0;
    double a = 0.0;
  };
  TensorF32 scanned;
  scanned.shape = {n, inner};
  scanned.data.resize(static_cast<std::size_t>(n * inner));
  std::vector<std::vector<Step>> tape(static_cast<std::size_t>(batch));
  for (int b = 0; b < batch; ++b) {
    tape[static_cast<std::size_t>(b)].resize(static_cast<std::size_t>(t * inner));
    std::vector<double> h(static_cast<std::size_t>(inner), 0.0);
    for (int ti = 0; ti < t; ++ti) {
      const int row = (b * t + ti) * (2 * inner);
      const int out_row = (b * t + ti) * inner;
      for (int i = 0; i < inner; ++i) {
        const double x = static_cast<double>(z.data[static_cast<std::size_t>(row + i)]);
        const double z_dt =
            static_cast<double>(z.data[static_cast<std::size_t>(row + inner + i)]);
        const double dt = softplus(z_dt);
        const double a =
            std::exp(-dt * std::exp(static_cast<double>(layer.a_log[static_cast<std::size_t>(i)])));
        const double h_prev = h[static_cast<std::size_t>(i)];
        h[static_cast<std::size_t>(i)] = a * h_prev + dt * x;
        const double y =
            h[static_cast<std::size_t>(i)] +
            static_cast<double>(layer.d_skip[static_cast<std::size_t>(i)]) * x;
        scanned.data[static_cast<std::size_t>(out_row + i)] = static_cast<float>(y);
        tape[static_cast<std::size_t>(b)][static_cast<std::size_t>(ti * inner + i)] =
            Step{h_prev, x, dt, a};
      }
    }
  }

  TensorF32 out_pre, out_post_unused;
  if (auto st = dense::forward_f32(layer.out_proj, scanned, &out_pre, &out_post_unused); !st) {
    return Status::Err(st.code, std::string("mamba out: ") + st.message);
  }
  (void)out_post_unused;

  TensorF32 gy_out;
  gy_out.shape = {n, d};
  gy_out.data.assign(grad_out.data.begin(),
                     grad_out.data.begin() + static_cast<std::ptrdiff_t>(want));

  TensorF32 g_scan, dW_out;
  if (auto st = dense::backward_f32(layer.out_proj, gy_out, scanned, out_pre, &g_scan, &dW_out);
      !st) {
    return Status::Err(st.code, std::string("mamba out bwd: ") + st.message);
  }

  TensorF32 g_z;
  g_z.shape = {n, 2 * inner};
  g_z.data.assign(static_cast<std::size_t>(n * 2 * inner), 0.0f);
  std::vector<double> dA(static_cast<std::size_t>(inner), 0.0);
  std::vector<double> dD(static_cast<std::size_t>(inner), 0.0);

  for (int b = 0; b < batch; ++b) {
    std::vector<double> gh(static_cast<std::size_t>(inner), 0.0);
    for (int ti = t - 1; ti >= 0; --ti) {
      const int out_row = (b * t + ti) * inner;
      const int row = (b * t + ti) * (2 * inner);
      for (int i = 0; i < inner; ++i) {
        const Step& stp =
            tape[static_cast<std::size_t>(b)][static_cast<std::size_t>(ti * inner + i)];
        const double gy =
            static_cast<double>(g_scan.data[static_cast<std::size_t>(out_row + i)]);
        dD[static_cast<std::size_t>(i)] += gy * stp.x;
        const double gx_y =
            gy * static_cast<double>(layer.d_skip[static_cast<std::size_t>(i)]);
        const double gh_new = gy + gh[static_cast<std::size_t>(i)];
        const double ga = gh_new * stp.h;
        const double gdt = gh_new * stp.x;
        const double gx = gh_new * stp.dt + gx_y;
        gh[static_cast<std::size_t>(i)] = gh_new * stp.a;
        const double exp_a =
            std::exp(static_cast<double>(layer.a_log[static_cast<std::size_t>(i)]));
        dA[static_cast<std::size_t>(i)] += ga * stp.a * (-stp.dt * exp_a);
        const double z_dt =
            static_cast<double>(z.data[static_cast<std::size_t>(row + inner + i)]);
        const double sig = 1.0 / (1.0 + std::exp(-z_dt));
        g_z.data[static_cast<std::size_t>(row + i)] = static_cast<float>(gx);
        g_z.data[static_cast<std::size_t>(row + inner + i)] =
            static_cast<float>((gdt + ga * stp.a * (-exp_a)) * sig);
      }
    }
  }

  TensorF32 gx_flat, dW_in;
  if (auto st = dense::backward_f32(layer.in_proj, g_z, flat, in_pre, &gx_flat, &dW_in); !st) {
    return Status::Err(st.code, std::string("mamba in bwd: ") + st.message);
  }

  grad_in->shape = {batch, t, d};
  grad_in->data.assign(gx_flat.data.begin(),
                       gx_flat.data.begin() + static_cast<std::ptrdiff_t>(want));

  const int n_in = layer.in_proj.weights.rows * layer.in_proj.weights.cols;
  const int n_out = layer.out_proj.weights.rows * layer.out_proj.weights.cols;
  if (static_cast<int>(dW_in.data.size()) < n_in || static_cast<int>(dW_out.data.size()) < n_out) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: short proj dW");
  }
  const int need = grad_w_size(layer);
  grad_w->shape = {1, need};
  grad_w->data.resize(static_cast<std::size_t>(need));
  int off = 0;
  for (int i = 0; i < n_in; ++i) {
    grad_w->data[static_cast<std::size_t>(off++)] = dW_in.data[static_cast<std::size_t>(i)];
  }
  for (int i = 0; i < n_out; ++i) {
    grad_w->data[static_cast<std::size_t>(off++)] = dW_out.data[static_cast<std::size_t>(i)];
  }
  for (int i = 0; i < inner; ++i) {
    grad_w->data[static_cast<std::size_t>(off + i)] = static_cast<float>(dA[static_cast<std::size_t>(i)]);
    grad_w->data[static_cast<std::size_t>(off + inner + i)] =
        static_cast<float>(dD[static_cast<std::size_t>(i)]);
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: ApplyGradSGD nil");
  }
  const int need = grad_w_size(*layer);
  if (static_cast<int>(dW.data.size()) < need) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: dW short");
  }
  int off = 0;
  const int n_in = layer->in_proj.weights.rows * layer->in_proj.weights.cols;
  TensorF32 s_in;
  s_in.shape = {layer->in_proj.weights.rows, layer->in_proj.weights.cols};
  s_in.data.assign(dW.data.begin() + off, dW.data.begin() + off + n_in);
  if (auto st = dense::apply_grad_sgd(&layer->in_proj, s_in, lr); !st) return st;
  off += n_in;
  const int n_out = layer->out_proj.weights.rows * layer->out_proj.weights.cols;
  TensorF32 s_out;
  s_out.shape = {layer->out_proj.weights.rows, layer->out_proj.weights.cols};
  s_out.data.assign(dW.data.begin() + off, dW.data.begin() + off + n_out);
  if (auto st = dense::apply_grad_sgd(&layer->out_proj, s_out, lr); !st) return st;
  off += n_out;
  const int inner = static_cast<int>(layer->a_log.size());
  for (int i = 0; i < inner; ++i) {
    layer->a_log[static_cast<std::size_t>(i)] -= static_cast<float>(
        lr * static_cast<double>(dW.data[static_cast<std::size_t>(off + i)]));
    layer->d_skip[static_cast<std::size_t>(i)] -= static_cast<float>(
        lr * static_cast<double>(dW.data[static_cast<std::size_t>(off + inner + i)]));
  }
  return Status::Ok();
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "mamba: nil dst");
  }
  int off = 0;
  const int n_in = layer.in_proj.weights.rows * layer.in_proj.weights.cols;
  if (auto st = layer.in_proj.weights.export_f32(dst + off); !st) return st;
  off += n_in;
  const int n_out = layer.out_proj.weights.rows * layer.out_proj.weights.cols;
  if (auto st = layer.out_proj.weights.export_f32(dst + off); !st) return st;
  off += n_out;
  for (float v : layer.a_log) {
    dst[off++] = v;
  }
  for (float v : layer.d_skip) {
    dst[off++] = v;
  }
  return Status::Ok();
}

} // namespace welvet::mamba
