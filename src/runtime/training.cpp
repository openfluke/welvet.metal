#include "welvet/metal/training.hpp"
#include "welvet/runtime/training.hpp"

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

#include <map>
#include <utility>

namespace welvet::runtime::training {
namespace {

struct CoordLess {
  bool operator()(const architecture::Coord& a, const architecture::Coord& b) const {
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    if (a.x != b.x) return a.x < b.x;
    return a.l < b.l;
  }
};

} // namespace

Status mse(const TensorF32& pred, const TensorF32& target, double* loss) {
  if (!loss || pred.shape != target.shape || pred.data.size() != target.data.size() || pred.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "training: MSE shape");
  }
  return metal::mse_f32(pred.data.data(), target.data.data(), pred.data.size(), loss, nullptr);
}

Status mse_grad(const TensorF32& pred, const TensorF32& target, TensorF32* grad) {
  if (!grad || pred.shape != target.shape || pred.data.size() != target.data.size() || pred.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "training: MSEGrad shape");
  }
  TensorF32 result = pred;
  if (auto st = metal::mse_f32(pred.data.data(), target.data.data(), pred.data.size(), nullptr,
                             result.data.data()); !st) return st;
  *grad = std::move(result);
  return Status::Ok();
}

Status sgd_dense(architecture::Grid& g, const forward::Result& fwd, const backward::Result& bwd,
                 double lr) {
  if (lr == 0.0) {
    return Status::Err(StatusCode::InvalidArgument, "training: LR is 0");
  }
  std::map<architecture::Coord, const TensorF32*, CoordLess> dw;
  for (const auto& gw : bwd.grad_ws) {
    dw[gw.coord] = &gw.dW;
  }
  for (const auto& st : fwd.steps) {
    auto it = dw.find(st.coord);
    if (it == dw.end() || !it->second) {
      continue;
    }
    architecture::Cell* cell = architecture::at(g, st.coord.z, st.coord.y, st.coord.x, st.coord.l);
    if (!cell) {
      return Status::Err(StatusCode::InvalidArgument, "training: SGD missing cell");
    }
    Status st_a = Status::Ok();
    switch (cell->op_kind) {
    case architecture::OpKind::Dense:
      st_a = dense::apply_grad_sgd(&cell->dense_op, *it->second, lr);
      break;
    case architecture::OpKind::RMSNorm:
      st_a = rmsnorm::apply_grad_sgd(&cell->rmsnorm_op, *it->second, lr);
      break;
    case architecture::OpKind::LayerNorm:
      st_a = layernorm::apply_grad_sgd(&cell->layernorm_op, *it->second, lr);
      break;
    case architecture::OpKind::SwiGLU:
      st_a = swiglu::apply_grad_sgd(&cell->swiglu_op, *it->second, lr);
      break;
    case architecture::OpKind::Softmax:
      st_a = softmax::apply_grad_sgd(&cell->softmax_op, *it->second, lr);
      break;
    case architecture::OpKind::Embedding:
      st_a = embedding::apply_grad_sgd(&cell->embedding_op, *it->second, lr);
      break;
    case architecture::OpKind::MHA:
      st_a = mha::apply_grad_sgd(&cell->mha_op, *it->second, lr);
      break;
    case architecture::OpKind::CNN1:
      st_a = cnn1::apply_grad_sgd(&cell->cnn1_op, *it->second, lr);
      break;
    case architecture::OpKind::CNN2:
      st_a = cnn2::apply_grad_sgd(&cell->cnn2_op, *it->second, lr);
      break;
    case architecture::OpKind::CNN3:
      st_a = cnn3::apply_grad_sgd(&cell->cnn3_op, *it->second, lr);
      break;
    case architecture::OpKind::ConvT1:
      st_a = convt1::apply_grad_sgd(&cell->convt1_op, *it->second, lr);
      break;
    case architecture::OpKind::ConvT2:
      st_a = convt2::apply_grad_sgd(&cell->convt2_op, *it->second, lr);
      break;
    case architecture::OpKind::ConvT3:
      st_a = convt3::apply_grad_sgd(&cell->convt3_op, *it->second, lr);
      break;
    case architecture::OpKind::RNN:
      st_a = rnn::apply_grad_sgd(&cell->rnn_op, *it->second, lr);
      break;
    case architecture::OpKind::LSTM:
      st_a = lstm::apply_grad_sgd(&cell->lstm_op, *it->second, lr);
      break;
    case architecture::OpKind::Sequential:
      st_a = sequential::apply_grad_sgd(&cell->sequential_op, *it->second, lr);
      break;
    case architecture::OpKind::Residual:
      st_a = residual::apply_grad_sgd(&cell->residual_op, *it->second, lr);
      break;
    case architecture::OpKind::Mamba:
      st_a = mamba::apply_grad_sgd(&cell->mamba_op, *it->second, lr);
      break;
    case architecture::OpKind::Stack:
      st_a = stack::apply_grad_sgd(&cell->stack_op, *it->second, lr);
      break;
    case architecture::OpKind::Metacognition:
      st_a = metacognition::apply_grad_sgd(&cell->metacognition_op, *it->second, lr);
      break;
    case architecture::OpKind::GDN:
      st_a = gdn::apply_grad_sgd(&cell->gdn_op, *it->second, lr);
      break;
    default:
      return Status::Gap(
          "training: SGD cell Op not yet on Metal (Dense/…/Stack/GDN NormalBP first)");
    }
    if (!st_a) {
      return st_a;
    }
  }
  return Status::Ok();
}

Status step_dense(architecture::Grid& g, const forward::Result& fwd, const TensorF32& target,
                  double lr, double* loss, backward::Result* bwd_out) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "training: nil loss");
  }
  if (auto st = mse(fwd.output, target, loss); !st) {
    return st;
  }
  TensorF32 gy;
  if (auto st = mse_grad(fwd.output, target, &gy); !st) {
    return st;
  }
  backward::Result bwd;
  if (auto st = backward::volume_dense(g, fwd, gy, &bwd); !st) {
    return st;
  }
  if (auto st = sgd_dense(g, fwd, bwd, lr); !st) {
    return st;
  }
  if (bwd_out) {
    *bwd_out = std::move(bwd);
  }
  return Status::Ok();
}

} // namespace welvet::runtime::training
