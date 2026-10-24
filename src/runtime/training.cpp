#include "welvet/runtime/training.hpp"

namespace welvet::runtime {

Status train_stack_gap(TrainMode) {
  return Status::Gap("TrainStackMSE / Step / Mesh / Tween* modes on Metal");
}

} // namespace welvet::runtime
