#include "welvet/layernorm.hpp"
#include "welvet/metal/layer_utils.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::layernorm {
namespace {

struct Layout {
  int batch = 1;
  int seq = 1;
  int dim = 0;
};

Status parse_layout(int dim, const TensorF32 &input, Layout *lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: empty input");
  }
  lay->dim = dim;
  if (input.shape.size() == 2) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = 1;
    if (static_cast<int>(input.shape[1]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "layernorm: width != Dim");
    }
  } else if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "layernorm: last dim != Dim");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument,
                       "layernorm: need [batch,dim] or [batch,seq,dim]");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: bad batch/seq");
  }
  const int want = lay->batch * lay->seq * dim;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: short data");
  }
  return Status::Ok();
}

} // namespace

Status validate(Config *cfg) {
  if (!cfg)
    return Status::Err(StatusCode::InvalidArgument, "layernorm: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: need Dim>0");
  }
  if (cfg->eps <= 0.0) {
    cfg->eps = 1e-5;
  }
  return Status::Ok();
}

Status new_configured(const Config &cfg_in, DType dt, const float *gamma_init,
                      const float *beta_init, Layer *out) {
  if (!out)
    return Status::Err(StatusCode::InvalidArgument, "layernorm: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st)
    return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = WeightStore::make(1, cfg.dim, dt, gamma_init, &layer.gamma); !st) {
    return Status::Err(st.code, std::string("layernorm gamma: ") + st.message);
  }
  if (auto st = WeightStore::make(1, cfg.dim, dt, beta_init, &layer.beta); !st) {
    return Status::Err(st.code, std::string("layernorm beta: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer &layer, const TensorF32 &input, TensorF32 *pre, TensorF32 *post) {
  if (!pre || !post)
    return Status::Err(StatusCode::InvalidArgument, "layernorm: nil output");
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  Layout lay;
  if (auto st = parse_layout(cfg.dim, input, &lay); !st)
    return st;
  const int dim = cfg.dim, nt = lay.batch * lay.seq;
  std::vector<float> gamma, beta;
  if (auto st = metal::decode_xy(layer.gamma, &gamma); !st)
    return st;
  if (auto st = metal::decode_xy(layer.beta, &beta); !st)
    return st;
  *pre = input;
  *post = input;
  pre->data.resize(nt * dim);
  post->data.resize(nt * dim);
  float hi, lo;
  metal::split_double(cfg.eps, &hi, &lo);
  return metal::run("op_norm_forward",
                    {metal::read(input.data), metal::read(gamma), metal::read(beta),
                     metal::write(pre->data), metal::write(post->data)},
                    {dim, nt, 1}, {hi, lo}, nt);
}

Status backward_f32(const Layer &layer, const TensorF32 &grad_out, const TensorF32 &input,
                    const TensorF32 &pre, TensorF32 *grad_in, TensorF32 *grad_w) {
  if (!grad_in || !grad_w)
    return Status::Err(StatusCode::InvalidArgument, "layernorm: nil gradients");
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  Layout lay;
  if (auto st = parse_layout(cfg.dim, input, &lay); !st)
    return st;
  const int dim = cfg.dim, nt = lay.batch * lay.seq;
  if (pre.data.size() < nt * dim || grad_out.data.size() < nt * dim)
    return Status::Err(StatusCode::InvalidArgument, "layernorm: short gradient/tape");
  std::vector<float> gamma, empty;
  if (auto st = metal::decode_xy(layer.gamma, &gamma); !st)
    return st;
  *grad_in = input;
  *grad_w = TensorF32::zeros(1, 2 * dim);
  float hi, lo;
  metal::split_double(cfg.eps, &hi, &lo);
  if (auto st = metal::run("op_norm_backward",
                           {metal::read(input.data), metal::read(gamma), metal::read(empty),
                            metal::read(pre.data), metal::read(empty), metal::read(grad_out.data),
                            metal::write(grad_in->data), metal::write(grad_w->data)},
                           {dim, nt, 1}, {hi, lo}, nt);
      !st)
    return st;
  return metal::run("op_norm_param_grad",
                    {metal::read(empty), metal::read(empty), metal::read(empty),
                     metal::read(pre.data), metal::read(empty), metal::read(grad_out.data),
                     metal::read(empty), metal::write(grad_w->data)},
                    {dim, nt, 1}, {}, dim);
}

Status apply_grad_sgd(Layer *layer, const TensorF32 &dW, double lr) {
  if (!layer)
    return Status::Err(StatusCode::InvalidArgument, "layernorm: nil layer");
  if (auto st = metal::update(layer->gamma, dW, lr); !st)
    return st;
  return metal::update(layer->beta, dW, lr, layer->cfg.dim);
}

} // namespace welvet::layernorm
