#include "welvet/metal/training.hpp"
#include "welvet/runtime/step.hpp"
#include "welvet/systems/tween.hpp"

#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/runtime/training.hpp"

#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/dense.hpp"
#include "welvet/embedding.hpp"
#include "welvet/gdn.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mamba.hpp"
#include "welvet/metacognition.hpp"
#include "welvet/mha.hpp"
#include "welvet/residual.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/softmax.hpp"
#include "welvet/stack.hpp"
#include "welvet/swiglu.hpp"

#include <sstream>
#include <utility>

namespace welvet::runtime::step {
namespace {

TensorF32 clone_t(const TensorF32& t) {
  TensorF32 o;
  o.shape = t.shape;
  o.data = t.data;
  return o;
}

bool tensor_empty(const TensorF32& t) { return t.data.empty(); }

Status accumulate(TensorF32* dst, const TensorF32& src) {
  if (!dst) return Status::Err(StatusCode::InvalidArgument, "step: nil accumulation buffer");
  if (dst->data.empty()) {
    *dst = clone_t(src);
    return Status::Ok();
  }
  if (dst->shape != src.shape || dst->data.size() != src.data.size()) {
    return Status::Err(StatusCode::InvalidArgument, "step: accumulation shape mismatch");
  }
  return metal::accumulate_f32(dst->data.data(), src.data.data(), src.data.size());
}

Status accumulate_mesh_grad(architecture::Grid& g, std::vector<TensorF32>& buffers, int idx,
                          const TensorF32& grad) {
  if (idx < 0 || idx >= static_cast<int>(g.cells.size()) || grad.data.empty()) {
    return Status::Ok();
  }
  architecture::Cell& cell = g.cells[static_cast<std::size_t>(idx)];
  int source_idx = 0;
  if (cell.is_remote_link) {
    source_idx = architecture::index(g, cell.target_z, cell.target_y, cell.target_x, cell.target_l);
  } else if (idx > 0) {
    source_idx = idx - 1;
  } else {
    source_idx = 0;
  }
  if (source_idx < 0 || source_idx >= static_cast<int>(buffers.size())) {
    return Status::Ok();
  }
  return accumulate(&buffers[static_cast<std::size_t>(source_idx)], grad);
}

Status apply_one(architecture::Cell* cell, const TensorF32& dW, double lr) {
  if (!cell || dW.data.empty()) {
    return Status::Ok();
  }
  switch (cell->op_kind) {
  case architecture::OpKind::Dense:
    return dense::apply_grad_sgd(&cell->dense_op, dW, lr);
  case architecture::OpKind::RMSNorm:
    return rmsnorm::apply_grad_sgd(&cell->rmsnorm_op, dW, lr);
  case architecture::OpKind::LayerNorm:
    return layernorm::apply_grad_sgd(&cell->layernorm_op, dW, lr);
  case architecture::OpKind::SwiGLU:
    return swiglu::apply_grad_sgd(&cell->swiglu_op, dW, lr);
  case architecture::OpKind::Softmax:
    return softmax::apply_grad_sgd(&cell->softmax_op, dW, lr);
  case architecture::OpKind::Embedding:
    return embedding::apply_grad_sgd(&cell->embedding_op, dW, lr);
  case architecture::OpKind::MHA:
    return mha::apply_grad_sgd(&cell->mha_op, dW, lr);
  case architecture::OpKind::CNN1:
    return cnn1::apply_grad_sgd(&cell->cnn1_op, dW, lr);
  case architecture::OpKind::CNN2:
    return cnn2::apply_grad_sgd(&cell->cnn2_op, dW, lr);
  case architecture::OpKind::CNN3:
    return cnn3::apply_grad_sgd(&cell->cnn3_op, dW, lr);
  case architecture::OpKind::ConvT1:
    return convt1::apply_grad_sgd(&cell->convt1_op, dW, lr);
  case architecture::OpKind::ConvT2:
    return convt2::apply_grad_sgd(&cell->convt2_op, dW, lr);
  case architecture::OpKind::ConvT3:
    return convt3::apply_grad_sgd(&cell->convt3_op, dW, lr);
  case architecture::OpKind::RNN:
    return rnn::apply_grad_sgd(&cell->rnn_op, dW, lr);
  case architecture::OpKind::LSTM:
    return lstm::apply_grad_sgd(&cell->lstm_op, dW, lr);
  case architecture::OpKind::Sequential:
    return sequential::apply_grad_sgd(&cell->sequential_op, dW, lr);
  case architecture::OpKind::Residual:
    return residual::apply_grad_sgd(&cell->residual_op, dW, lr);
  case architecture::OpKind::Mamba:
    return mamba::apply_grad_sgd(&cell->mamba_op, dW, lr);
  case architecture::OpKind::Stack:
    return stack::apply_grad_sgd(&cell->stack_op, dW, lr);
  case architecture::OpKind::Metacognition:
    return metacognition::apply_grad_sgd(&cell->metacognition_op, dW, lr);
  case architecture::OpKind::GDN:
    return gdn::apply_grad_sgd(&cell->gdn_op, dW, lr);
  case architecture::OpKind::KMeans:
  case architecture::OpKind::Parallel:
    return Status::Gap("step: SGD unsupported cell Op (KMeans/Parallel)");
  default:
    return Status::Err(StatusCode::InvalidArgument, "step: SGD unsupported cell Op");
  }
}

} // namespace

void ensure_size(State* s, int n) {
  if (!s || n < 0) return;
  if (static_cast<int>(s->layer_data.size()) != n) {
    s->layer_data.assign(static_cast<std::size_t>(n), TensorF32{});
    s->next_buffer.assign(static_cast<std::size_t>(n), TensorF32{});
  }
}

void set_input(State* s, const TensorF32& input) {
  if (!s) return;
  s->last_input = clone_t(input);
  if (!s->layer_data.empty()) {
    s->layer_data[0] = clone_t(input);
  }
}

void clear_history(State* s) {
  if (!s) return;
  s->history_in.clear();
  s->history_pre.clear();
}

Status forward(architecture::Grid& g, State* s, bool capture) {
  if (!s) {
    return Status::Err(StatusCode::InvalidArgument, "step: nil state");
  }
  const int num = architecture::stack_layer_count(g);
  ensure_size(s, num);
  if (static_cast<int>(s->next_buffer.size()) != num) {
    s->next_buffer.assign(static_cast<std::size_t>(num), TensorF32{});
  }

  std::vector<TensorF32> current_in;
  std::vector<TensorF32> current_pre;
  if (capture) {
    current_in.assign(static_cast<std::size_t>(num), TensorF32{});
    current_pre.assign(static_cast<std::size_t>(num), TensorF32{});
  }

  for (int z = 0; z < g.depth; ++z) {
    for (int y = 0; y < g.rows; ++y) {
      for (int x = 0; x < g.cols; ++x) {
        for (int l = 0; l < g.layers_per_cell; ++l) {
          const int idx = architecture::index(g, z, y, x, l);
          architecture::Cell* cell = architecture::at(g, z, y, x, l);
          if (!cell) {
            continue;
          }
          if (cell->meta.is_disabled) {
            if (idx > 0) {
              s->next_buffer[static_cast<std::size_t>(idx)] =
                  s->layer_data[static_cast<std::size_t>(idx - 1)];
            } else {
              s->next_buffer[static_cast<std::size_t>(idx)] =
                  s->layer_data[static_cast<std::size_t>(idx)];
            }
            continue;
          }
          if (cell->op_kind == architecture::OpKind::None) {
            std::ostringstream oss;
            oss << "step: no op at (" << z << "," << y << "," << x << "," << l << ")";
            return Status::Err(StatusCode::InvalidArgument, oss.str());
          }

          const TensorF32* input = nullptr;
          if (cell->is_remote_link) {
            const int t_idx =
                architecture::index(g, cell->target_z, cell->target_y, cell->target_x, cell->target_l);
            if (t_idx >= 0 && t_idx < num) {
              input = &s->layer_data[static_cast<std::size_t>(t_idx)];
            }
          } else if (idx > 0) {
            input = &s->layer_data[static_cast<std::size_t>(idx - 1)];
          } else {
            input = &s->layer_data[0];
          }
          if (!input || tensor_empty(*input)) {
            continue;
          }

          TensorF32 pre, post;
          if (auto st = forward::cell(*cell, *input, &pre, &post); !st) {
            return Status::Err(st.code, std::string("step fwd: ") + st.message);
          }
          s->next_buffer[static_cast<std::size_t>(idx)] = post;
          if (capture) {
            current_in[static_cast<std::size_t>(idx)] = clone_t(*input);
            current_pre[static_cast<std::size_t>(idx)] = std::move(pre);
          }
        }
      }
    }
  }

  s->layer_data = s->next_buffer;
  if (capture) {
    s->history_in.push_back(std::move(current_in));
    s->history_pre.push_back(std::move(current_pre));
  }
  s->step_count++;
  return Status::Ok();
}

Status backward(architecture::Grid& g, State* s, const TensorF32& grad_out, TensorF32* grad_in,
                std::vector<TensorF32>* layer_dws) {
  if (!s) {
    return Status::Err(StatusCode::InvalidArgument, "step: nil state");
  }
  if (s->history_in.empty()) {
    if (grad_in) *grad_in = TensorF32{};
    if (layer_dws) layer_dws->clear();
    return Status::Ok();
  }
  const int num_layers = static_cast<int>(g.cells.size());
  const int num_steps = static_cast<int>(s->history_in.size());
  if (layer_dws) {
    layer_dws->assign(static_cast<std::size_t>(num_layers), TensorF32{});
  }

  std::vector<TensorF32> grad_buffers(static_cast<std::size_t>(num_layers));
  if (num_layers > 0) {
    grad_buffers[static_cast<std::size_t>(num_layers - 1)] = clone_t(grad_out);
  }

  for (int step_idx = num_steps - 1; step_idx >= 0; --step_idx) {
    const auto& step_in = s->history_in[static_cast<std::size_t>(step_idx)];
    const auto& step_pre = s->history_pre[static_cast<std::size_t>(step_idx)];
    std::vector<TensorF32> next_grad(static_cast<std::size_t>(num_layers));

    for (int idx = num_layers - 1; idx >= 0; --idx) {
      architecture::Cell& cell = g.cells[static_cast<std::size_t>(idx)];
      if (cell.meta.is_disabled) {
        if (!tensor_empty(grad_buffers[static_cast<std::size_t>(idx)])) {
          if (auto st = accumulate_mesh_grad(g, next_grad, idx, grad_buffers[static_cast<std::size_t>(idx)]); !st) return st;
        }
        continue;
      }
      const TensorF32& input = step_in[static_cast<std::size_t>(idx)];
      const TensorF32& pre = step_pre[static_cast<std::size_t>(idx)];
      const TensorF32& current_grad = grad_buffers[static_cast<std::size_t>(idx)];
      if (tensor_empty(input) || tensor_empty(pre) || tensor_empty(current_grad)) {
        continue;
      }
      TensorF32 g_in, g_w;
      if (auto st = backward::cell(cell, current_grad, input, pre, &g_in, &g_w); !st) {
        return Status::Err(st.code, std::string("step bwd: ") + st.message);
      }
      if (layer_dws && !g_w.data.empty()) {
        if (auto st = accumulate(&(*layer_dws)[static_cast<std::size_t>(idx)], g_w); !st) return st;
      }
      if (!g_in.data.empty()) {
        if (auto st = accumulate_mesh_grad(g, next_grad, idx, g_in); !st) return st;
      }
    }
    grad_buffers = std::move(next_grad);
  }

  if (grad_in) {
    *grad_in = num_layers > 0 ? std::move(grad_buffers[0]) : TensorF32{};
  }
  return Status::Ok();
}

Status apply_sgd(architecture::Grid& g, const std::vector<TensorF32>& layer_dws, double lr) {
  if (lr == 0.0) {
    return Status::Err(StatusCode::InvalidArgument, "step: LR is 0");
  }
  const int n = static_cast<int>(g.cells.size());
  for (int i = 0; i < n && i < static_cast<int>(layer_dws.size()); ++i) {
    if (auto st = apply_one(&g.cells[static_cast<std::size_t>(i)], layer_dws[static_cast<std::size_t>(i)], lr);
        !st) {
      return st;
    }
  }
  return Status::Ok();
}

Status train_mse(architecture::Grid& g, const TensorF32& input, const TensorF32& target, int ticks,
                 double lr, double* loss, Credit credit) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "step: nil loss");
  }
  if (ticks <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "step: ticks <= 0");
  }
  if (credit != Credit::FamilyBP && credit != Credit::FamilyTween) {
    return Status::Gap(std::string("runtime.step Credit ") +
                       std::to_string(static_cast<int>(credit)) + " (unsupported mesh credit on Metal)");
  }

  State st;
  ensure_size(&st, architecture::stack_layer_count(g));
  clear_history(&st);
  for (int t = 0; t < ticks; ++t) {
    set_input(&st, input);
    if (auto s = forward(g, &st, true); !s) {
      return s;
    }
  }
  if (st.layer_data.empty() || tensor_empty(st.layer_data.back())) {
    return Status::Err(StatusCode::InvalidArgument, "step: empty mesh output");
  }
  const TensorF32& pred = st.layer_data.back();
  if (auto s = training::mse(pred, target, loss); !s) {
    return s;
  }
  TensorF32 gy;
  if (auto s = training::mse_grad(pred, target, &gy); !s) {
    return s;
  }

  if (credit == Credit::FamilyTween) {
    const auto& inputs = st.history_in.back();
    const auto& pres = st.history_pre.back();
    TensorF32 gap = std::move(gy);
    for (int idx = int(g.cells.size())-1; idx >= 0; --idx) {
      auto& cell = g.cells[size_t(idx)];
      const auto& cin = inputs[size_t(idx)];
      if (cell.meta.is_disabled || cin.data.empty() || pres[size_t(idx)].data.empty() ||
          st.layer_data[size_t(idx)].data.empty()) continue;
      if (auto s = systems::tween::apply_cell_projected(cell,gap,cin,pres[size_t(idx)],
                         st.layer_data[size_t(idx)],lr*0.5); !s) return s;
      TensorF32 next;
      next.shape = cin.shape;
      next.data.assign(cin.data.size(),0.0f);
      if (gap.data.size() == next.data.size()) next.data = gap.data;
      gap = std::move(next);
    }
    return Status::Ok();
  }
  TensorF32 gin;
  std::vector<TensorF32> dws;
  if (auto s = backward(g, &st, gy, &gin, &dws); !s) {
    return s;
  }
  (void)gin;
  return apply_sgd(g, dws, lr);
}

} // namespace welvet::runtime::step
