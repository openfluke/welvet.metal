#include "welvet/sequential.hpp"

#include <string>
#include <vector>

namespace welvet::sequential {
namespace {

struct Layout {
  int batch = 0;
  int seq = 0;
  int in = 0;
};

Status parse_layout(int in_dim, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: empty input");
  }
  lay->in = in_dim;
  if (input.shape.size() == 2) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = 1;
    if (static_cast<int>(input.shape[1]) != in_dim) {
      return Status::Err(StatusCode::InvalidArgument, "sequential: width != Dim");
    }
  } else if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != in_dim) {
      return Status::Err(StatusCode::InvalidArgument, "sequential: last dim != Dim");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument, "sequential: need [batch,in] or [batch,seq,in]");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: invalid batch/seq");
  }
  const int want = lay->batch * lay->seq * lay->in;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: data short");
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
  if (d <= 0) d = 2;
  return d * cfg.dim * cfg.dim;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "sequential: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: need positive Dim");
  }
  if (cfg->seq_len <= 0) cfg->seq_len = 1;
  if (cfg->depth <= 0) cfg->depth = 2;
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, Activation child_act,
                      const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "sequential: nil out");
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
      return Status::Err(st.code, std::string("sequential child ") + std::to_string(i) + ": " +
                                      st.message);
    }
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: nil pre/post");
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

  TensorF32 current = flatten(input, lay);
  TensorF32 last_pre;
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    auto& ch = layer.children[i];
    TensorF32 p, o;
    if (auto st = dense::forward_f32(ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("sequential fwd child ") + std::to_string(i) + ": " +
                                      st.message);
    }
    last_pre = std::move(p);
    current = std::move(o);
  }
  *pre = unflatten(last_pre, lay, cfg.dim);
  *post = unflatten(current, lay, cfg.dim);
  return Status::Ok();
}


int grad_w_size(const Layer& layer) {
  int n = 0;
  for (const auto& ch : layer.children) {
    n += ch.weights.rows * ch.weights.cols;
  }
  return n;
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& /*pre*/, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: nil grad out");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  if (layer.children.empty()) {
    *grad_in = grad_out;
    grad_w->shape = {1, 0};
    grad_w->data.clear();
    return Status::Ok();
  }

  Layout lay;
  if (auto st = parse_layout(cfg.dim, input, &lay); !st) return st;
  const int want_out = lay.batch * lay.seq * cfg.dim;
  if (static_cast<int>(grad_out.data.size()) < want_out) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: gradOut short");
  }

  const int n = static_cast<int>(layer.children.size());
  std::vector<TensorF32> ins(static_cast<std::size_t>(n));
  std::vector<TensorF32> pres(static_cast<std::size_t>(n));
  TensorF32 current = flatten(input, lay);
  for (int i = 0; i < n; ++i) {
    ins[static_cast<std::size_t>(i)] = current;
    auto& ch = layer.children[static_cast<std::size_t>(i)];
    TensorF32 p, o;
    if (auto st = dense::forward_f32(ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("sequential recompute child ") + std::to_string(i) +
                                      ": " + st.message);
    }
    pres[static_cast<std::size_t>(i)] = std::move(p);
    current = std::move(o);
  }

  TensorF32 gy = flatten(grad_out, lay);
  std::vector<TensorF32> d_ws(static_cast<std::size_t>(n));
  for (int i = n - 1; i >= 0; --i) {
    auto& ch = layer.children[static_cast<std::size_t>(i)];
    TensorF32 gx, dw;
    if (auto st = dense::backward_f32(ch, gy, ins[static_cast<std::size_t>(i)],
                                      pres[static_cast<std::size_t>(i)], &gx, &dw);
        !st) {
      return Status::Err(st.code, std::string("sequential bwd child ") + std::to_string(i) + ": " +
                                      st.message);
    }
    d_ws[static_cast<std::size_t>(i)] = std::move(dw);
    gy = std::move(gx);
  }

  *grad_in = unflatten(gy, lay, cfg.dim);
  const int need = grad_w_size(layer);
  grad_w->shape = {1, need};
  grad_w->data.resize(static_cast<std::size_t>(need));
  int off = 0;
  for (int i = 0; i < n; ++i) {
    const int cn = layer.children[static_cast<std::size_t>(i)].weights.rows *
                   layer.children[static_cast<std::size_t>(i)].weights.cols;
    if (static_cast<int>(d_ws[static_cast<std::size_t>(i)].data.size()) < cn) {
      return Status::Err(StatusCode::InvalidArgument, "sequential: short child dW");
    }
    for (int j = 0; j < cn; ++j) {
      grad_w->data[static_cast<std::size_t>(off + j)] =
          d_ws[static_cast<std::size_t>(i)].data[static_cast<std::size_t>(j)];
    }
    off += cn;
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: ApplyGradSGD nil");
  }
  const int need = grad_w_size(*layer);
  if (static_cast<int>(dW.data.size()) < need) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: dW short");
  }
  int off = 0;
  for (auto& ch : layer->children) {
    const int cn = ch.weights.rows * ch.weights.cols;
    TensorF32 slice;
    slice.shape = {ch.weights.rows, ch.weights.cols};
    slice.data.assign(dW.data.begin() + off, dW.data.begin() + off + cn);
    if (auto st = dense::apply_grad_sgd(&ch, slice, lr); !st) return st;
    off += cn;
  }
  return Status::Ok();
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "sequential: nil dst");
  }
  int off = 0;
  for (const auto& ch : layer.children) {
    const int cn = ch.weights.rows * ch.weights.cols;
    if (auto st = ch.weights.export_f32(dst + off); !st) return st;
    off += cn;
  }
  return Status::Ok();
}

} // namespace welvet::sequential
