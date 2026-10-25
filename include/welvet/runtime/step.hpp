#pragma once

#include "welvet/architecture.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>
#include <vector>

namespace welvet::runtime::step {

struct State {
  std::vector<TensorF32> layer_data;
  std::vector<TensorF32> next_buffer;
  std::vector<std::vector<TensorF32>> history_in;
  std::vector<std::vector<TensorF32>> history_pre;
  TensorF32 last_input;
  uint64_t step_count = 0;
};

void ensure_size(State* s, int n);
void set_input(State* s, const TensorF32& input);
void clear_history(State* s);

Status forward(architecture::Grid& g, State* s, bool capture);

Status backward(architecture::Grid& g, State* s, const TensorF32& grad_out, TensorF32* grad_in,
                std::vector<TensorF32>* layer_dws);

Status apply_sgd(architecture::Grid& g, const std::vector<TensorF32>& layer_dws, double lr);

enum class Credit : int32_t {
  FamilyBP = 0,
  FamilyTween = 1,
  FamilyTweenSplit = 2,
  FamilyTweenAlt = 3,
  FamilyTweenSplitFastProxy = 4,
  FamilyTweenSplitSparse = 5,
};

Status train_mse(architecture::Grid& g, const TensorF32& input, const TensorF32& target, int ticks,
                 double lr, double* loss, Credit credit = Credit::FamilyBP);

} // namespace welvet::runtime::step
