#include "welvet/tensor.hpp"

// Tensor helpers stay header-light for M0; this TU anchors the library.
namespace welvet {
namespace {
volatile int tensor_tu_anchor = 0;
}
void tensor_force_link() { tensor_tu_anchor = 1; }
} // namespace welvet
