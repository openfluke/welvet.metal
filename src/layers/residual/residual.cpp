#include "welvet/residual.hpp"

#include <string>

namespace welvet::residual {
namespace {

struct Layout {
  int batch = 0;
  int seq = 0;
  int in = 0;
};

Status parse_layout(int in_dim, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "residual: empty input");
  }
  lay->in = in_dim;
  if (input.shape.size() == 2) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = 1;
    if (static_cast<int>(input.shape[1]) != in_dim) {
      return Status::Err(StatusCode::InvalidArgument, "residual: width != Dim");
    }
  } else if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != in_dim) {
      return Status::Err(StatusCode::InvalidArgument, "residual: last dim != Dim");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument, "residual: need [batch,in] or [batch,seq,in]");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "residual: invalid batch/seq");
  }
  const int want = lay->batch * lay->seq * lay->in;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "residual: data short");
  }
  return Status::Ok();
}

TensorF32 flatten(const TensorF32& input, const Layout& lay) {
  TensorF32 out;
  const int bs = lay.batch * lay.seq;
  out.shape = {bs, lay.in};
  out.data.assign(input.data.begin(),
                  input.data.begin() + static_cast<std::ptrdiff_t>(bs * lay.in));
  return out;
}

TensorF32 unflatten(const TensorF32& flat, const Layout& lay, int last) {
  TensorF32 out;
  if (lay.seq == 1) {
    out.shape = {lay.batch, last};
  } else {
    out.shape = {lay.batch, lay.seq, last};
  }
  const int n = lay.batch * lay.seq * last;
  out.data.assign(flat.data.begin(), flat.data.begin() + static_cast<std::ptrdiff_t>(n));
  return out;
}

} // namespace

int weight_count(const Config& cfg) {
  int d = cfg.depth;
  if (d <= 0) d = 1;
  return d * cfg.dim * cfg.dim;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "residual: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "residual: need positive Dim");
  }
  if (cfg->seq_len <= 0) cfg->seq_len = 1;
  if (cfg->depth <= 0) cfg->depth = 1;
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, Activation child_act,
                      const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "residual: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  const int dim = cfg.dim;
  const int depth = cfg.depth;
  const int child_n = dim * dim;
  Layer layer;
  layer.cfg = cfg;
  layer.children.resize(static_cast<std::size_t>(depth));
  for (int i = 0; i < depth; ++i) {
    const float* init = nullptr;
    if (packed_or_null) {
      init = packed_or_null + i * child_n;
    }
    if (auto st = dense::new_layer(dim, dim, child_act, dt, init,
                                   &layer.children[static_cast<std::size_t>(i)]);
        !st) {
      return Status::Err(st.code, std::string("residual F child ") + std::to_string(i) + ": " +
                                      st.message);
    }
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "residual: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  if (layer.children.empty()) {
    *pre = input;
    *post = input;
    return Status::Ok();
  }

  Layout lay;
  if (auto st = parse_layout(cfg.dim, input, &lay); !st) return st;

  TensorF32 x_flat = flatten(input, lay);
  TensorF32 current = x_flat;
  TensorF32 last_pre;
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    auto& ch = layer.children[i];
    TensorF32 p, o;
    if (auto st = dense::forward_f32(ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("residual F fwd child ") + std::to_string(i) + ": " +
                                      st.message);
    }
    last_pre = std::move(p);
    current = std::move(o);
  }

  // y = F(x) + x via f64 round-trip (matches Go AsFloat64 + FromFloat64).
  TensorF32 y;
  y.shape = current.shape;
  y.data.resize(current.data.size());
  for (std::size_t i = 0; i < current.data.size(); ++i) {
    y.data[i] = static_cast<float>(static_cast<double>(current.data[i]) +
                                   static_cast<double>(x_flat.data[i]));
  }

  *pre = unflatten(last_pre, lay, cfg.dim);
  *post = unflatten(y, lay, cfg.dim);
  return Status::Ok();
}

} // namespace welvet::residual
