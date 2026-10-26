#pragma once
#include "welvet/train_mode.hpp"
#include <optional>
#include <vector>

namespace welvet::cameral {
enum class SyncWhen : int {
  Manual = 0,
  AfterSample = 1,
  AfterStep = 2,
  AfterPulse = 3
};
struct Endpoint {
  int stack_idx = 0, branch = 0, store = 0;
};
struct Pair {
  Endpoint a, b;
};
struct SyncConfig {
  bool enabled = false;
  double alpha = 1;
  SyncWhen when = SyncWhen::Manual;
  std::vector<double> branch_alpha;
  std::vector<std::vector<int>> groups;
  std::vector<Pair> cross;
};
// Scheduling/state metadata. Tensor arithmetic, loss and blending run in Metal.
struct Controls {
  std::vector<TrainMode> modes;
  std::vector<double> branch_lrs;
  double shadow_coef = 0, surprise_thresh = 0, last_loss = 0;
  std::optional<SyncConfig> sync;
};
} // namespace welvet::cameral
