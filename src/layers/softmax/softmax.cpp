#include "welvet/metal/layer_utils.hpp"
#include "welvet/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace welvet::softmax {
namespace {

struct Layout {
  std::vector<int64_t> shape;
  int n = 0;
  int rows = 0;
  int cols = 0;
  double temp = 1.0;
};

Status parse_layout(const Config &cfg, const TensorF32 &input, Layout *lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: empty input");
  }
  lay->shape = input.shape;
  lay->n = static_cast<int>(input.data.size());
  lay->temp = cfg.temperature == 0.0 ? 1.0 : cfg.temperature;

  switch (cfg.kind) {
  case Kind::Grid: {
    lay->cols = cfg.cols > 0 ? cfg.cols : cfg.dim;
    lay->rows = cfg.rows;
    if (lay->rows <= 0) {
      if (lay->cols <= 0 || lay->n % lay->cols != 0) {
        return Status::Err(StatusCode::InvalidArgument, "softmax: grid n not divisible by cols");
      }
      lay->rows = lay->n / lay->cols;
    }
    if (lay->rows * lay->cols > lay->n) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: grid rows×cols > n");
    }
    break;
  }
  case Kind::Hierarchical: {
    if (!cfg.hierarchy_levels.empty()) {
      lay->cols = cfg.hierarchy_levels.back();
    } else if (cfg.cols > 0) {
      lay->cols = cfg.cols;
    } else {
      lay->cols = cfg.dim;
    }
    lay->rows = cfg.rows;
    if (lay->rows <= 0) {
      if (lay->cols <= 0 || lay->n % lay->cols != 0) {
        return Status::Err(StatusCode::InvalidArgument,
                           "softmax: hierarchical n not divisible by cols");
      }
      lay->rows = lay->n / lay->cols;
    }
    if (lay->rows * lay->cols > lay->n) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: hierarchical rows×cols > n");
    }
    break;
  }
  default: {
    if (input.shape.empty()) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: scalar input");
    }
    lay->cols = static_cast<int>(input.shape.back());
    if (cfg.dim > 0 && lay->cols != cfg.dim) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: last dim != Dim");
    }
    if (lay->cols <= 0 || lay->n % lay->cols != 0) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: n not divisible by cols");
    }
    lay->rows = lay->n / lay->cols;
    break;
  }
  }
  return Status::Ok();
}

} // namespace

Status validate(Config *cfg) {
  if (!cfg)
    return Status::Err(StatusCode::InvalidArgument, "softmax: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: need positive Dim");
  }
  if (cfg->seq_len <= 0)
    cfg->seq_len = 1;
  if (cfg->temperature == 0.0)
    cfg->temperature = 1.0;
  if (cfg->temperature < 0.0) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: Temperature < 0");
  }
  switch (cfg->kind) {
  case Kind::Grid:
    if (cfg->cols <= 0)
      cfg->cols = cfg->dim;
    if (cfg->rows <= 0) {
      return Status::Err(StatusCode::InvalidArgument, "softmax: KindGrid needs positive Rows");
    }
    break;
  case Kind::Hierarchical:
    if (cfg->hierarchy_levels.empty()) {
      return Status::Err(StatusCode::InvalidArgument,
                         "softmax: KindHierarchical needs HierarchyLevels");
    }
    if (cfg->cols <= 0) {
      cfg->cols = cfg->hierarchy_levels.back();
    }
    break;
  default:
    break;
  }
  return Status::Ok();
}

Status new_configured(const Config &cfg_in, DType dt, Layer *out) {
  if (!out)
    return Status::Err(StatusCode::InvalidArgument, "softmax: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st)
    return st;
  Layer layer;
  layer.cfg = std::move(cfg);
  layer.dtype = dt;
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(const Layer &layer, const TensorF32 &input, TensorF32 *pre, TensorF32 *post) {
  if (!pre || !post)
    return Status::Err(StatusCode::InvalidArgument, "softmax: nil outputs");
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st)
    return st;
  std::vector<int32_t> mask;
  for (bool v : cfg.mask)
    mask.push_back(v ? 1 : 0);
  std::vector<float> logits(lay.n), sorted(lay.n);
  *pre = input;
  pre->data.assign(lay.n, 0);
  float hi, lo;
  metal::split_double(lay.temp, &hi, &lo);
  if (auto st =
          metal::run("op_softmax_forward",
                     {metal::read(input.data), metal::read(mask), metal::write(logits),
                      metal::write(sorted), metal::write(pre->data)},
                     {lay.rows, lay.cols, int(cfg.kind), int(mask.size()), int(cfg.gumbel_seed)},
                     {hi, lo, float(cfg.entmax_alpha)}, lay.rows);
      !st)
    return st;
  *post = *pre;
  return Status::Ok();
}

Status backward_f32(const Layer &layer, const TensorF32 &grad_out, const TensorF32 & /*input*/,
                    const TensorF32 &pre, TensorF32 *grad_in, TensorF32 *grad_w) {
  if (!grad_in || !grad_w)
    return Status::Err(StatusCode::InvalidArgument, "softmax: nil gradient");
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  Layout lay;
  if (auto st = parse_layout(cfg, pre, &lay); !st)
    return st;
  if (grad_out.data.size() < pre.data.size())
    return Status::Err(StatusCode::InvalidArgument, "softmax: gradient shape");
  std::vector<int32_t> mask;
  for (bool v : cfg.mask)
    mask.push_back(v ? 1 : 0);
  *grad_in = pre;
  grad_w->shape.clear();
  grad_w->data.clear();
  float hi, lo;
  metal::split_double(lay.temp, &hi, &lo);
  return metal::run("op_softmax_backward",
                    {metal::read(pre.data), metal::read(grad_out.data), metal::read(mask),
                     metal::write(grad_in->data)},
                    {lay.rows, lay.cols, int(cfg.kind), int(mask.size())}, {hi, lo}, lay.rows);
}

Status apply_grad_sgd(Layer *layer, const TensorF32 & /*dW*/, double /*lr*/) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: ApplyGradSGD nil layer");
  }
  return Status::Ok();
}

} // namespace welvet::softmax
