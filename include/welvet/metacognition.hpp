#pragma once

#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>
#include <vector>

namespace welvet::metacognition {

enum class Condition : int32_t {
  None = 0,
  StdAbove = 1,
  StdBelow = 2,
  AvgAbove = 3,
  AvgBelow = 4,
  MaxAbove = 5,
  ActiveBelow = 6,
  GainDrift = 7,
};

enum class Effect : int32_t {
  None = 0,
  Gate = 1,
  Identity = 2,
  Passthrough = 3,
};

struct Rule {
  Condition condition = Condition::None;
  double threshold = 0.0;
  Effect effect = Effect::None;
  int cooldown = 0;
  int fire_count = 0;
};

struct Config {
  int dim = 0;
  int seq_len = 0;         // 0 → [batch, dim] layout hint
  std::vector<Rule> rules; // empty → DefaultStabilityRules
};

int weight_count(const Config &cfg);
std::vector<Rule> default_stability_rules();
Status validate(Config *cfg);

// Observed Dense Dim→Dim soft-double + Metal heuristic rules (WMC1).
struct Layer {
  Config cfg{};
  dense::Layer observed;
  std::vector<Rule> rules;
  double last_gate = 1.0;
};

// packed: Observed [Dim×Dim] (f32 → FormatNone dtype).
Status new_configured(const Config &cfg, DType dt, const float *packed_or_null, Layer *out);

// Input [batch,dim] or [batch,seq,dim]. pre/post same outer shape.
Status forward(Layer &layer, const TensorF32 &input, TensorF32 *pre, TensorF32 *post);

int grad_w_size(const Layer &layer);

Status backward_f32(Layer &layer, const TensorF32 &grad_out, const TensorF32 &input,
                    const TensorF32 &pre, TensorF32 *grad_in, TensorF32 *grad_w);

Status apply_grad_sgd(Layer *layer, const TensorF32 &dW, double lr);

Status export_packed_f32(const Layer &layer, float *dst);

} // namespace welvet::metacognition
