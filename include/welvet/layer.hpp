#pragma once

#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"

#include <cstdint>

namespace welvet {

enum class LayerType : int32_t {
  Dense = 0,
  MultiHeadAttention = 1,
  SwiGLU = 2,
  RMSNorm = 3,
  CNN1 = 4,
  CNN2 = 5,
  CNN3 = 6,
  RNN = 7,
  LSTM = 8,
  LayerNorm = 9,
  ConvTransposed1D = 10,
  ConvTransposed2D = 11,
  ConvTransposed3D = 12,
  Embedding = 13,
  KMeans = 14,
  Softmax = 15,
  Parallel = 16,
  Sequential = 17,
  Residual = 18,
  Metacognition = 19,
  Mamba = 20,
  GDN = 21,
  Stack = 22,
};

struct LayerMeta {
  LayerType type = LayerType::Dense;
  DType dtype = DType::Float32;
  Activation activation = Activation::ReLU;
  int input_height = 0;
  int output_height = 0;
  int z = 0, y = 0, x = 0, l = 0;
  int tile_size = 32;
  bool multi_core = true;
  bool is_disabled = false;
};

} // namespace welvet
