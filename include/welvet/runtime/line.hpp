#pragma once

#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>
#include <vector>

namespace welvet::runtime::line {

enum class FlightRole : uint8_t {
  Train = 0,
  Action = 1,
};

enum class Credit : uint8_t {
  FamilyBP = 0,
  FamilyTween = 1,
  FamilyTweenSplit = 2,
  FamilyTweenAlt = 3,
  FamilyTweenSplitHeadProxy = 4,
  FamilyTweenSplitLinear = 5,
  FamilyTweenSplitFastProxy = 6,
  FamilyTweenSplitLinearCache = 7,
  FamilyTweenSplitHeadProxyAsync = 8,
  FamilyTweenSplitSparse = 9,
};

struct Flight {
  FlightRole role = FlightRole::Train;
  TensorF32 target;
  int stage = 0;
  std::vector<TensorF32> ins;
  std::vector<TensorF32> pres;
  std::vector<TensorF32> posts;
};

struct State {
  std::vector<Flight> flights;
};

struct TickResult {
  double loss = 0.0;
  bool popped_train = false;
  TensorF32 action_out;
};

Status tick(stack::Layer& ops, State* st, const TensorF32& input, const TensorF32* target,
            FlightRole role, double lr, TickResult* out, Credit credit = Credit::FamilyBP);

Status train_mse(stack::Layer& ops, State* st, const TensorF32& input, const TensorF32& target,
                 double lr, double* loss, Credit credit = Credit::FamilyBP);

int in_flight(const State& st);

} // namespace welvet::runtime::line
