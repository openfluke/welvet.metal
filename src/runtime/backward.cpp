#include "welvet/runtime/backward.hpp"

#include "welvet/dense.hpp"

#include <sstream>
#include <utility>

namespace welvet::runtime::backward {

Status cell(architecture::Cell& cell, const TensorF32& grad_out, const TensorF32& input,
            const TensorF32& pre, TensorF32* grad_in, TensorF32* dW) {
  if (!grad_in || !dW) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.backward: nil grad buffers");
  }
  switch (cell.op_kind) {
  case architecture::OpKind::Dense:
    return dense::backward_f32(cell.dense_op, grad_out, input, pre, grad_in, dW);
  default:
    return Status::Gap("runtime.backward: cell Op not yet on Metal (Dense NormalBP first)");
  }
}

Status volume_dense(architecture::Grid& g, const forward::Result& fwd, const TensorF32& grad_out,
                    Result* out) {
  if (!out) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.backward: nil result");
  }
  if (fwd.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.backward: empty tape");
  }
  if (grad_out.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.backward: empty gradOut");
  }

  Result result;
  result.grad_ws.reserve(fwd.steps.size());
  TensorF32 gy = grad_out;

  for (int i = static_cast<int>(fwd.steps.size()) - 1; i >= 0; --i) {
    const auto& st = fwd.steps[static_cast<std::size_t>(i)];
    architecture::Cell* c = architecture::at(g, st.coord.z, st.coord.y, st.coord.x, st.coord.l);
    if (!c) {
      std::ostringstream oss;
      oss << "runtime.backward: missing cell at z=" << st.coord.z << " y=" << st.coord.y
          << " x=" << st.coord.x << " l=" << st.coord.l;
      return Status::Err(StatusCode::InvalidArgument, oss.str());
    }

    TensorF32 gx, dw;
    if (auto st_b = cell(*c, gy, st.input, st.pre, &gx, &dw); !st_b) {
      return st_b;
    }
    if (!dw.data.empty()) {
      result.grad_ws.push_back(GradW{st.coord, std::move(dw)});
    }
    gy = std::move(gx);
  }

  result.grad_in = std::move(gy);
  *out = std::move(result);
  return Status::Ok();
}

} // namespace welvet::runtime::backward
