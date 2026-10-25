#include "welvet/runtime/forward.hpp"

#include <map>
#include <sstream>
#include <utility>

namespace welvet::runtime::forward {
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

Status cell(architecture::Cell& cell, const TensorF32& in, TensorF32* pre, TensorF32* post) {
  switch (cell.op_kind) {
  case architecture::OpKind::Dense:
    return dense::forward_f32(cell.dense_op, in, pre, post);
  case architecture::OpKind::RMSNorm:
    return rmsnorm::forward(cell.rmsnorm_op, in, pre, post);
  case architecture::OpKind::LayerNorm:
    return layernorm::forward(cell.layernorm_op, in, pre, post);
  case architecture::OpKind::SwiGLU:
    return swiglu::forward(cell.swiglu_op, in, pre, post);
  case architecture::OpKind::Softmax:
    return softmax::forward(cell.softmax_op, in, pre, post);
  case architecture::OpKind::Embedding:
    return embedding::forward(cell.embedding_op, in, pre, post);
  case architecture::OpKind::MHA:
    return mha::forward(cell.mha_op, in, pre, post);
  case architecture::OpKind::CNN1:
    return cnn1::forward(cell.cnn1_op, in, pre, post);
  case architecture::OpKind::CNN2:
    return cnn2::forward(cell.cnn2_op, in, pre, post);
  case architecture::OpKind::CNN3:
    return cnn3::forward(cell.cnn3_op, in, pre, post);
  case architecture::OpKind::ConvT1:
    return convt1::forward(cell.convt1_op, in, pre, post);
  case architecture::OpKind::ConvT2:
    return convt2::forward(cell.convt2_op, in, pre, post);
  case architecture::OpKind::ConvT3:
    return convt3::forward(cell.convt3_op, in, pre, post);
  case architecture::OpKind::RNN:
    return rnn::forward(cell.rnn_op, in, pre, post);
  case architecture::OpKind::LSTM:
    return lstm::forward(cell.lstm_op, in, pre, post);
  case architecture::OpKind::Sequential:
    return sequential::forward(cell.sequential_op, in, pre, post);
  case architecture::OpKind::Residual:
    return residual::forward(cell.residual_op, in, pre, post);
  case architecture::OpKind::Mamba:
    return mamba::forward(cell.mamba_op, in, pre, post);
  case architecture::OpKind::Stack:
    return stack::forward(cell.stack_op, in, pre, post);
  case architecture::OpKind::Metacognition:
    return metacognition::forward(cell.metacognition_op, in, pre, post);
  case architecture::OpKind::GDN:
    return gdn::forward(cell.gdn_op, in, pre, post);
  case architecture::OpKind::KMeans:
    return kmeans::forward(cell.kmeans_op, in, pre, post);
  case architecture::OpKind::Parallel:
    return parallel::forward(cell.parallel_op, in, pre, post);
  default:
    return Status::Gap("runtime.forward: unsupported cell Op");
  }
}

Status volume_dense(architecture::Grid& g, const TensorF32& input, Result* out) {
  if (!out) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.forward: nil result");
  }
  if (input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "runtime.forward: empty input");
  }

  TensorF32 current = input;
  std::map<architecture::Coord, TensorF32, CoordLess> posts;
  Result result;

  for (const auto& coord : architecture::hop_order(g)) {
    architecture::Cell* c = architecture::at(g, coord.z, coord.y, coord.x, coord.l);
    if (!c || c->meta.is_disabled) continue;
    if (c->op_kind == architecture::OpKind::None) {
      std::ostringstream oss;
      oss << "runtime.forward: no op at z=" << coord.z << " y=" << coord.y << " x=" << coord.x
          << " l=" << coord.l;
      return Status::Err(StatusCode::InvalidArgument, oss.str());
    }

    const TensorF32* in = &current;
    TensorF32 remote_holder;
    if (c->is_remote_link) {
      architecture::Coord tc{c->target_z, c->target_y, c->target_x, c->target_l};
      auto it = posts.find(tc);
      if (it == posts.end()) {
        std::ostringstream oss;
        oss << "runtime.forward: remote hop has no recorded activation → z=" << tc.z
            << " y=" << tc.y << " x=" << tc.x << " l=" << tc.l;
        return Status::Err(StatusCode::InvalidArgument, oss.str());
      }
      remote_holder = it->second;
      in = &remote_holder;
    }

    TensorF32 pre, post;
    Status st = cell(*c, *in, &pre, &post);
    if (!st) return st;

    Step step;
    step.coord = coord;
    step.input = *in;
    step.pre = std::move(pre);
    step.post = post;
    posts[coord] = post;
    current = std::move(post);
    result.steps.push_back(std::move(step));
  }

  if (result.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument,
                       "runtime.forward: grid has no enabled cells with ops");
  }
  result.output = std::move(current);
  *out = std::move(result);
  return Status::Ok();
}

} // namespace welvet::runtime::forward
