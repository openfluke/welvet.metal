#pragma once
#include "welvet/interop/child_wire.hpp"
namespace welvet::interop::wire {
welvet::Status bind_mesh_child(welvet::architecture::Grid &grid,
                               const welvet::architecture::Coord &c,
                               welvet::stack::Child child) {
  using namespace welvet;
  switch (child.kind) {
  case stack::Kind::Dense:
    return architecture::bind_dense(grid, c.z, c.y, c.x, c.l,
                                    std::move(child.dense_op));
  case stack::Kind::RMSNorm:
    return architecture::bind_rmsnorm(grid, c.z, c.y, c.x, c.l,
                                      std::move(child.rmsnorm_op));
  case stack::Kind::LayerNorm:
    return architecture::bind_layernorm(grid, c.z, c.y, c.x, c.l,
                                        std::move(child.layernorm_op));
  case stack::Kind::SwiGLU:
    return architecture::bind_swiglu(grid, c.z, c.y, c.x, c.l,
                                     std::move(child.swiglu_op));
  case stack::Kind::Softmax:
    return architecture::bind_softmax(grid, c.z, c.y, c.x, c.l,
                                      std::move(child.softmax_op));
  case stack::Kind::Embedding:
    return architecture::bind_embedding(grid, c.z, c.y, c.x, c.l,
                                        std::move(child.embedding_op));
  case stack::Kind::MHA:
    return architecture::bind_mha(grid, c.z, c.y, c.x, c.l,
                                  std::move(child.mha_op));
  case stack::Kind::CNN1:
    return architecture::bind_cnn1(grid, c.z, c.y, c.x, c.l,
                                   std::move(child.cnn1_op));
  case stack::Kind::CNN2:
    return architecture::bind_cnn2(grid, c.z, c.y, c.x, c.l,
                                   std::move(child.cnn2_op));
  case stack::Kind::CNN3:
    return architecture::bind_cnn3(grid, c.z, c.y, c.x, c.l,
                                   std::move(child.cnn3_op));
  case stack::Kind::ConvT1:
    return architecture::bind_convt1(grid, c.z, c.y, c.x, c.l,
                                     std::move(child.convt1_op));
  case stack::Kind::ConvT2:
    return architecture::bind_convt2(grid, c.z, c.y, c.x, c.l,
                                     std::move(child.convt2_op));
  case stack::Kind::ConvT3:
    return architecture::bind_convt3(grid, c.z, c.y, c.x, c.l,
                                     std::move(child.convt3_op));
  case stack::Kind::RNN:
    return architecture::bind_rnn(grid, c.z, c.y, c.x, c.l,
                                  std::move(child.rnn_op));
  case stack::Kind::LSTM:
    return architecture::bind_lstm(grid, c.z, c.y, c.x, c.l,
                                   std::move(child.lstm_op));
  case stack::Kind::Sequential:
    return architecture::bind_sequential(grid, c.z, c.y, c.x, c.l,
                                         std::move(child.sequential_op));
  case stack::Kind::Residual:
    return architecture::bind_residual(grid, c.z, c.y, c.x, c.l,
                                       std::move(child.residual_op));
  case stack::Kind::Mamba:
    return architecture::bind_mamba(grid, c.z, c.y, c.x, c.l,
                                    std::move(child.mamba_op));
  case stack::Kind::Metacognition:
    return architecture::bind_metacognition(grid, c.z, c.y, c.x, c.l,
                                            std::move(child.metacognition_op));
  case stack::Kind::GDN:
    return architecture::bind_gdn(grid, c.z, c.y, c.x, c.l,
                                  std::move(child.gdn_op));
  case stack::Kind::KMeans:
    return architecture::bind_kmeans(grid, c.z, c.y, c.x, c.l,
                                     std::move(child.kmeans_op));
  case stack::Kind::NestedStack: {
    stack::Layer layer;
    layer.children = std::move(child.nested);
    layer.controls = std::move(child.controls);
    layer.accel = std::move(child.accel);
    return architecture::bind_stack(grid, c.z, c.y, c.x, c.l, std::move(layer));
  }
  case stack::Kind::Parallel: {
    parallel::Layer layer;
    const auto &p = child.parallel_nest;
    layer.cfg = {p.dim,
                 p.out_feat,
                 int(child.nested.size()),
                 parallel::Combine(p.combine),
                 p.seq_len,
                 p.sparse_k,
                 p.disagree_beta};
    layer.controls = std::move(child.controls);
    layer.accel = std::move(child.accel);
    layer.branches = std::move(child.nested);
    if (p.has_gate)
      layer.gate = std::move(child.dense_op);
    return architecture::bind_parallel(grid, c.z, c.y, c.x, c.l,
                                       std::move(layer));
  }
  default:
    return Status::Err(StatusCode::InvalidArgument, "mesh: invalid child");
  }
}
welvet::stack::Child take_mesh_child(welvet::architecture::Cell &cell) {
  using namespace welvet;
  stack::Child child;
  child.kind = stack::Kind(int(cell.op_kind));
  switch (child.kind) {
  case stack::Kind::Dense:
    child.dense_op = std::move(cell.dense_op);
    break;
  case stack::Kind::RMSNorm:
    child.rmsnorm_op = std::move(cell.rmsnorm_op);
    break;
  case stack::Kind::LayerNorm:
    child.layernorm_op = std::move(cell.layernorm_op);
    break;
  case stack::Kind::SwiGLU:
    child.swiglu_op = std::move(cell.swiglu_op);
    break;
  case stack::Kind::Softmax:
    child.softmax_op = std::move(cell.softmax_op);
    break;
  case stack::Kind::Embedding:
    child.embedding_op = std::move(cell.embedding_op);
    break;
  case stack::Kind::MHA:
    child.mha_op = std::move(cell.mha_op);
    break;
  case stack::Kind::CNN1:
    child.cnn1_op = std::move(cell.cnn1_op);
    break;
  case stack::Kind::CNN2:
    child.cnn2_op = std::move(cell.cnn2_op);
    break;
  case stack::Kind::CNN3:
    child.cnn3_op = std::move(cell.cnn3_op);
    break;
  case stack::Kind::ConvT1:
    child.convt1_op = std::move(cell.convt1_op);
    break;
  case stack::Kind::ConvT2:
    child.convt2_op = std::move(cell.convt2_op);
    break;
  case stack::Kind::ConvT3:
    child.convt3_op = std::move(cell.convt3_op);
    break;
  case stack::Kind::RNN:
    child.rnn_op = std::move(cell.rnn_op);
    break;
  case stack::Kind::LSTM:
    child.lstm_op = std::move(cell.lstm_op);
    break;
  case stack::Kind::Sequential:
    child.sequential_op = std::move(cell.sequential_op);
    break;
  case stack::Kind::Residual:
    child.residual_op = std::move(cell.residual_op);
    break;
  case stack::Kind::Mamba:
    child.mamba_op = std::move(cell.mamba_op);
    break;
  case stack::Kind::Metacognition:
    child.metacognition_op = std::move(cell.metacognition_op);
    break;
  case stack::Kind::GDN:
    child.gdn_op = std::move(cell.gdn_op);
    break;
  case stack::Kind::KMeans:
    child.kmeans_op = std::move(cell.kmeans_op);
    break;
  case stack::Kind::NestedStack:
    child.nested = std::move(cell.stack_op.children);
    child.controls = std::move(cell.stack_op.controls);
    child.accel = std::move(cell.stack_op.accel);
    break;
  case stack::Kind::Parallel: {
    const auto &p = cell.parallel_op.cfg;
    child.parallel_nest = {int(p.combine),
                           p.dim,
                           p.out_feat,
                           p.seq_len,
                           p.sparse_k,
                           p.disagree_beta,
                           cell.parallel_op.gate ? 1 : 0};
    child.controls = std::move(cell.parallel_op.controls);
    child.accel = std::move(cell.parallel_op.accel);
    child.nested = std::move(cell.parallel_op.branches);
    if (cell.parallel_op.gate)
      child.dense_op = std::move(*cell.parallel_op.gate);
    break;
  }
  default:
    break;
  }
  return child;
}
template <class R> bool read_cam_controls(R &r, welvet::cameral::Controls *c) {
  int32_t n = 0;
  if (!r.read(&n, 1) || n < 0 || n > 4096)
    return false;
  c->modes.resize(n);
  for (auto &mode : c->modes) {
    int32_t m;
    if (!r.read(&m, 1) || m < 0 || m > 33)
      return false;
    mode = welvet::TrainMode(m);
  }
  if (!r.read(&n, 1) || n < 0 || n > 4096)
    return false;
  c->branch_lrs.resize(n);
  if (!r.read(c->branch_lrs.data(), n))
    return false;
  if (!r.read(&c->shadow_coef, 1) || !r.read(&c->surprise_thresh, 1) ||
      !r.read(&c->last_loss, 1))
    return false;
  int32_t present;
  if (!r.read(&present, 1))
    return false;
  if (!present) {
    c->sync.reset();
    return true;
  }
  c->sync.emplace();
  auto &cfg = *c->sync;
  int32_t enabled, when;
  if (!r.read(&enabled, 1) || !r.read(&cfg.alpha, 1) || !r.read(&when, 1) ||
      when < 0 || when > 3)
    return false;
  cfg.enabled = enabled != 0;
  cfg.when = welvet::cameral::SyncWhen(when);
  if (!r.read(&n, 1) || n < 0 || n > 4096)
    return false;
  cfg.branch_alpha.resize(n);
  if (!r.read(cfg.branch_alpha.data(), n))
    return false;
  if (!r.read(&n, 1) || n < 0 || n > 4096)
    return false;
  cfg.groups.resize(n);
  for (auto &group : cfg.groups) {
    int32_t count;
    if (!r.read(&count, 1) || count < 0 || count > 4096)
      return false;
    group.resize(count);
    if (!r.read(group.data(), count))
      return false;
  }
  if (!r.read(&n, 1) || n < 0 || n > 4096)
    return false;
  cfg.cross.resize(n);
  for (auto &pair : cfg.cross)
    for (auto *e : {&pair.a, &pair.b})
      if (!r.read(&e->stack_idx, 1) || !r.read(&e->branch, 1) ||
          !r.read(&e->store, 1))
        return false;
  return true;
}
template <class R> bool read_cam_tree(R &r, welvet::stack::Child *c) {
  if (!read_cam_controls(r, &c->controls))
    return false;
  for (auto &ch : c->nested)
    if (!read_cam_tree(r, &ch))
      return false;
  return true;
}

} // namespace welvet::interop::wire
