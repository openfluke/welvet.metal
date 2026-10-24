#include "welvet/status.hpp"

namespace welvet {
namespace {
volatile int status_tu_anchor = 0;
}
void status_force_link() { status_tu_anchor = 1; }
} // namespace welvet
