#pragma once
#include "welvet/metal/matvec.hpp"
#include "welvet/metal/ops.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"
#include <cstring>
namespace welvet::metal {
inline Status decode_xy(const WeightStore &w, std::vector<float> *xy) {
  xy->resize(static_cast<std::size_t>(w.rows) * w.cols * 2);
  std::vector<double> row(w.cols);
  for (int i = 0; i < w.rows; ++i) {
    if (auto st = w.decode_row_f64(i, row.data()); !st)
      return st;
    for (int k = 0; k < w.cols; ++k)
      split_double(row[k], &(*xy)[2 * (i * w.cols + k)], &(*xy)[2 * (i * w.cols + k) + 1]);
  }
  return Status::Ok();
}
inline Status update(WeightStore &w, const TensorF32 &dw, double lr, int offset = 0) {
  const auto n = static_cast<std::size_t>(w.rows) * w.cols;
  if (offset < 0 || dw.data.size() < n + offset)
    return Status::Err(StatusCode::InvalidArgument, "Metal SGD: gradient shape");
  std::vector<double> values(n);
  for (std::size_t i = 0; i < n; ++i)
    values[i] = dw.data[offset + i];
  return w.apply_sgd(values.data(), lr);
}
inline Status add(const std::vector<float> &x, const std::vector<float> &y, std::vector<float> *z) {
  if (x.size() != y.size())
    return Status::Err(StatusCode::InvalidArgument, "Metal add: shape");
  z->resize(x.size());
  return run("op_add", {read(x), read(y), write(*z)}, {int(x.size())}, {}, x.size());
}
} // namespace welvet::metal
