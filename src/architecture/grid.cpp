#include "welvet/architecture.hpp"

namespace welvet::architecture {

Status new_grid(int32_t depth, int32_t rows, int32_t cols, int32_t layers_per_cell, Grid* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null Grid");
  if (depth <= 0 || rows <= 0 || cols <= 0 || layers_per_cell <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "bad grid dims");
  }
  out->depth = depth;
  out->rows = rows;
  out->cols = cols;
  out->layers_per_cell = layers_per_cell;
  return Status::Ok();
}

int32_t cell_count(const Grid& g) {
  return g.depth * g.rows * g.cols * g.layers_per_cell;
}

} // namespace welvet::architecture
