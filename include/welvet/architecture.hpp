#pragma once

#include <cstdint>
#include <vector>

#include "welvet/status.hpp"

namespace welvet::architecture {

struct Grid {
  int32_t depth = 1;
  int32_t rows = 1;
  int32_t cols = 1;
  int32_t layers_per_cell = 1;
};

Status new_grid(int32_t depth, int32_t rows, int32_t cols, int32_t layers_per_cell, Grid* out);
int32_t cell_count(const Grid& g);

} // namespace welvet::architecture
