#pragma once

#include "welvet/activate.hpp"
#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>

namespace welvet::kmeans {

enum class OutputMode : int32_t {
  Probabilities = 0,
  Features = 1,
};

struct Config {
  int num_clusters = 0;
  int feature_dim = 0;
  double temperature = 1.0; // 0 → 1
  OutputMode output_mode = OutputMode::Probabilities;
  Activation activation = Activation::Linear;
};

int out_dim(const Config& cfg);
Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  dense::Layer centers; // Rows=K, Cols=FeatureDim
};

// init: flat [K×D] f32 (packed into FormatNone dtype).
Status new_configured(const Config& cfg, DType dt, const float* init_or_null, Layer* out);

// Soft assignment host f64 (Go/cpp exact_bit). Centers via decode_row_f64 FormatNone×34.
Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::kmeans
