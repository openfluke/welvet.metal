#pragma once

#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>
#include <vector>

namespace welvet::softmax {

// Mirrors welvet/layers/softmax Kind.
enum class Kind : int32_t {
  Standard = 0,
  Temperature = 1,
  Gumbel = 2,
  Masked = 3,
  Sparse = 4,
  Entmax = 5,
  Grid = 6,
  Hierarchical = 7,
};

struct Config {
  int dim = 0;
  int seq_len = 1;
  int rows = 0;
  int cols = 0;
  double temperature = 1.0;
  Kind kind = Kind::Standard;
  std::vector<bool> mask;            // KindMasked: false → masked out
  double entmax_alpha = 0.0;         // 0 → 1.5
  std::vector<int> hierarchy_levels; // KindHierarchical: last = cols
  uint32_t gumbel_seed = 0;          // 0 → fixed LCG; else seed
};

Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  DType dtype = DType::Float32; // harness only (weightless)
};

Status new_configured(const Config& cfg, DType dt, Layer* out);

// pre/post both probabilities y (same values). Softmax over last axis or grid rows×cols.
// Host f64 temperature + f32 stable Softmax/sparsemax/entmax (Go/cpp exact_bit).
Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::softmax
