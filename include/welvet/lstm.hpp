#pragma once

#include "welvet/dense.hpp"
#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::lstm {

struct Config {
  int input_size = 0;
  int hidden_size = 0;
  int seq_len = 0;
};

int gate_size(const Config& cfg);
int weight_count(const Config& cfg);
Status validate(const Config& cfg);

struct Gate {
  dense::Layer IH;
  dense::Layer HH;
};

struct Layer {
  Config cfg{};
  Gate I, F, G, O;
};

// packed: [i|f|g|o] each [IH|HH|bias] as f32.
Status new_configured(const Config& cfg, DType dt, const float* packed_or_null, Layer* out);

// Forward: pre [batch,seq,5H]=[i,f,g,o,c]; post [batch,seq,H]=h. h₀=c₀=0.
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::lstm
