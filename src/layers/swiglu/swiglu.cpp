#include "welvet/swiglu.hpp"

#include "welvet/activate.hpp"

#include <string>
#include <vector>

namespace welvet::swiglu {
namespace {

struct Layout {
  int batch = 1;
  int seq = 1;
  int in = 0;
};

Status parse_layout(int in_dim, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: empty input");
  }
  lay->in = in_dim;
  if (input.shape.size() == 2) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = 1;
    if (static_cast<int>(input.shape[1]) != in_dim) {
      return Status::Err(StatusCode::InvalidArgument, "swiglu: width != InputDim");
    }
  } else if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != in_dim) {
      return Status::Err(StatusCode::InvalidArgument, "swiglu: last dim != InputDim");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: need [batch,in] or [batch,seq,in]");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: bad batch/seq");
  }
  const int want = lay->batch * lay->seq * lay->in;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: short data");
  }
  return Status::Ok();
}

} // namespace

Status validate(const Config& cfg) {
  if (cfg.input_dim <= 0 || cfg.intermediate_dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: need InputDim>0 and IntermediateDim>0");
  }
  return Status::Ok();
}

Config default_ffn(int d_model) {
  int inter = (d_model * 8) / 3;
  inter = (inter + 7) / 8 * 8;
  if (inter < d_model) inter = d_model * 2;
  return Config{d_model, inter};
}

Status new_configured(const Config& cfg, DType dt, const float* gate_init, const float* up_init,
                      const float* down_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "swiglu: nil out");
  if (auto st = validate(cfg); !st) return st;

  const int in = cfg.input_dim;
  const int inter = cfg.intermediate_dim;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(in, inter, Activation::Linear, dt, gate_init, &layer.Gate); !st) {
    return Status::Err(st.code, std::string("swiglu Gate: ") + st.message);
  }
  if (auto st = dense::new_layer(in, inter, Activation::Linear, dt, up_init, &layer.Up); !st) {
    return Status::Err(st.code, std::string("swiglu Up: ") + st.message);
  }
  if (auto st = dense::new_layer(inter, in, Activation::Linear, dt, down_init, &layer.Down); !st) {
    return Status::Err(st.code, std::string("swiglu Down: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: nil pre/post");
  }
  if (auto st = validate(layer.cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(layer.cfg.input_dim, input, &lay); !st) return st;

  const int in = layer.cfg.input_dim;
  const int inter = layer.cfg.intermediate_dim;
  const int bs = lay.batch * lay.seq;

  TensorF32 flat;
  flat.shape = {bs, in};
  flat.data.assign(input.data.begin(), input.data.begin() + static_cast<std::size_t>(bs * in));

  TensorF32 gate_pre, gate_post, up_pre, up_post;
  if (auto st = dense::forward_f32(layer.Gate, flat, &gate_pre, &gate_post); !st) {
    return Status::Err(st.code, std::string("swiglu Gate: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.Up, flat, &up_pre, &up_post); !st) {
    return Status::Err(st.code, std::string("swiglu Up: ") + st.message);
  }

  // Match Go: SiLU(gatePre) → f32, then f64(g)*f64(upPost) → f32.
  TensorF32 h;
  h.shape = {bs, inter};
  h.data.resize(static_cast<std::size_t>(bs * inter));
  const int n = bs * inter;
  for (int i = 0; i < n; ++i) {
    const double silu = activate_f64(static_cast<double>(gate_pre.data[static_cast<std::size_t>(i)]),
                                     Activation::Silu);
    const float g_f = static_cast<float>(silu);
    h.data[static_cast<std::size_t>(i)] = static_cast<float>(
        static_cast<double>(g_f) * static_cast<double>(up_post.data[static_cast<std::size_t>(i)]));
  }

  TensorF32 down_pre, down_post;
  if (auto st = dense::forward_f32(layer.Down, h, &down_pre, &down_post); !st) {
    return Status::Err(st.code, std::string("swiglu Down: ") + st.message);
  }

  if (lay.seq == 1) {
    pre->shape = {lay.batch, inter};
    post->shape = {lay.batch, in};
  } else {
    pre->shape = {lay.batch, lay.seq, inter};
    post->shape = {lay.batch, lay.seq, in};
  }
  pre->data = std::move(h.data);
  post->data = std::move(down_post.data);
  return Status::Ok();
}

} // namespace welvet::swiglu
