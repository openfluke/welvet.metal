#include "welvet/runtime/backward.hpp"

#include "welvet/cnn1.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/dense.hpp"
#include "welvet/embedding.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/mha.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/rnn.hpp"
#include "welvet/lstm.hpp"
#include "welvet/sequential.hpp"
#include "welvet/residual.hpp"
#include "welvet/gdn.hpp"
#include "welvet/mamba.hpp"
#include "welvet/metacognition.hpp"
#include "welvet/stack.hpp"
#include "welvet/softmax.hpp"
#include "welvet/swiglu.hpp"

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
  case architecture::OpKind::RMSNorm:
    return rmsnorm::backward_f32(cell.rmsnorm_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::LayerNorm:
    return layernorm::backward_f32(cell.layernorm_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::SwiGLU:
    return swiglu::backward_f32(cell.swiglu_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::Softmax:
    return softmax::backward_f32(cell.softmax_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::Embedding:
    return embedding::backward_f32(cell.embedding_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::MHA:
    return mha::backward_f32(cell.mha_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::CNN1:
    return cnn1::backward_f32(cell.cnn1_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::CNN2:
    return cnn2::backward_f32(cell.cnn2_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::CNN3:
    return cnn3::backward_f32(cell.cnn3_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::ConvT1:
    return convt1::backward_f32(cell.convt1_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::ConvT2:
    return convt2::backward_f32(cell.convt2_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::ConvT3:
    return convt3::backward_f32(cell.convt3_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::RNN:
    return rnn::backward_f32(cell.rnn_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::LSTM:
    return lstm::backward_f32(cell.lstm_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::Sequential:
    return sequential::backward_f32(cell.sequential_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::Residual:
    return residual::backward_f32(cell.residual_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::Mamba:
    return mamba::backward_f32(cell.mamba_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::Stack:
    return stack::backward_f32(cell.stack_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::Metacognition:
    return metacognition::backward_f32(cell.metacognition_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::GDN:
    return gdn::backward_f32(cell.gdn_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::KMeans:
    return kmeans::backward_f32(cell.kmeans_op, grad_out, input, pre, grad_in, dW);
  case architecture::OpKind::Parallel:
    return parallel::backward_f32(cell.parallel_op, grad_out, input, pre, grad_in, dW);
  default:
    return Status::Gap(
        "runtime.backward: cell Op not yet on Metal (Dense/…/Stack/GDN NormalBP first)");
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
