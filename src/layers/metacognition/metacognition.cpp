#include "welvet/metacognition.hpp"

#include "welvet/weights.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::metacognition {
namespace {

struct Stats {
  double avg = 0, std = 0, max = 0, min = 0;
  int active = 0, total = 0;
};

struct Layout {
  int batch = 0;
  int seq = 0;
  int dim = 0;
};

} // namespace

int weight_count(const Config& cfg) { return cfg.dim * cfg.dim; }

std::vector<Rule> default_stability_rules() {
  return {
      {Condition::GainDrift, 0.5, Effect::Identity, 5, 0},
      {Condition::StdAbove, 50.0, Effect::Gate, 3, 0},
      {Condition::ActiveBelow, 0.01, Effect::Identity, 10, 0},
  };
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "metacognition: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: need positive Dim");
  }
  if (cfg->seq_len < 0) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: SeqLen < 0");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "metacognition: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st =
          dense::new_layer(cfg.dim, cfg.dim, Activation::Linear, dt, packed_or_null, &layer.observed);
      !st) {
    return Status::Err(st.code, std::string("metacognition observed: ") + st.message);
  }
  if (cfg.rules.empty()) {
    layer.rules = default_stability_rules();
  } else {
    layer.rules = cfg.rules;
  }
  layer.last_gate = 1.0;
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "metacognition.forward: Metal kernel not implemented; CPU execution is forbidden");
}

int grad_w_size(const Layer& layer) { return layer.cfg.dim * layer.cfg.dim; }

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "metacognition.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "metacognition.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "metacognition: nil dst");
  }
  return layer.observed.weights.export_f32(dst);
}

} // namespace welvet::metacognition
