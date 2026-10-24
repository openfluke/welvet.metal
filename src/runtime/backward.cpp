#include "welvet/runtime/backward.hpp"

namespace welvet::runtime {

Status volume_backward_gap() {
  return Status::Gap("volume backward on Metal (tape + dX/dW across cells)");
}

} // namespace welvet::runtime
