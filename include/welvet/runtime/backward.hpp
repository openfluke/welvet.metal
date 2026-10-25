#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <vector>

namespace welvet::runtime::backward {

struct GradW {
  architecture::Coord coord;
  TensorF32 dW;
};

struct Result {
  TensorF32 grad_in;
  std::vector<GradW> grad_ws;
};

// Per-cell backward dispatch. Dense NormalBP first; other kinds Gap until ported.
Status cell(architecture::Cell& cell, const TensorF32& grad_out, const TensorF32& input,
            const TensorF32& pre, TensorF32* grad_in, TensorF32* dW);

// Reverse hop over a forward tape. Remote cells keep local gradIn
// (not re-routed to the remote source — Go v0 parity).
Status volume_dense(architecture::Grid& g, const forward::Result& fwd, const TensorF32& grad_out,
                    Result* out);

} // namespace welvet::runtime::backward
