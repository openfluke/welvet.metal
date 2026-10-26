#include "welvet/parallel.hpp"

#include "welvet/activate.hpp"
#include "welvet/metal/layer_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace welvet::parallel {
int out_dim(const Config& cfg) {
  if (cfg.out_feat <= 0) return 0;
  if (cfg.combine == Combine::Concat) return cfg.branches * cfg.out_feat;
  return cfg.out_feat;
}

int weight_count(const Config& cfg) {
  const int branch_w = cfg.branches * cfg.out_feat * cfg.dim;
  const int gate_w = cfg.combine == Combine::Filter ? cfg.branches * cfg.dim : 0;
  return branch_w + gate_w;
}

int sparse_k_eff(const Config& cfg) {
  int k = cfg.sparse_k;
  if (k <= 0) k = 2;
  if (k > cfg.branches) k = cfg.branches;
  if (k < 1) k = 1;
  return k;
}

double disagree_beta_eff(const Config& cfg) {
  if (cfg.disagree_beta == 0.0) return 1.0;
  return cfg.disagree_beta;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "parallel: nil config");
  if (cfg->dim <= 0 || cfg->branches <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: need positive Dim/Branches");
  }
  if (cfg->out_feat < 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: OutFeat < 0");
  }
  switch (cfg->combine) {
    case Combine::Concat:
    case Combine::Add:
    case Combine::Avg:
    case Combine::Filter:
    case Combine::Max:
    case Combine::SparseK:
    case Combine::Disagree:
      break;
    default:
      return Status::Err(StatusCode::InvalidArgument, "parallel: unknown Combine");
  }
  if (cfg->seq_len < 0 || cfg->sparse_k < 0 || !std::isfinite(cfg->disagree_beta)) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: SeqLen/SparseK < 0");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "parallel: nil out");
  Config cfg = cfg_in;
  if (cfg.out_feat <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: NewConfigured requires OutFeat > 0");
  }
  if (auto st = validate(&cfg); !st) return st;

  const int child_n = cfg.out_feat * cfg.dim;
  Layer layer;
  layer.cfg = cfg;
  layer.branches.resize(static_cast<std::size_t>(cfg.branches));
  for (int i = 0; i < cfg.branches; ++i) {
    layer.branches[i].kind = BranchKind::Dense;
    const float* init = nullptr;
    if (packed_or_null) init = packed_or_null + i * child_n;
    if (auto st = dense::new_layer(cfg.dim, cfg.out_feat, Activation::Linear, dt, init,
                                   &layer.branches[static_cast<std::size_t>(i)].dense_op);
        !st) {
      return Status::Err(st.code, std::string("parallel branch ") + std::to_string(i) + ": " +
                                      st.message);
    }
  }
  if (cfg.combine == Combine::Filter) {
    dense::Layer gate;
    const float* ginit = nullptr;
    if (packed_or_null) ginit = packed_or_null + cfg.branches * child_n;
    if (auto st = dense::new_layer(cfg.dim, cfg.branches, Activation::Linear, dt, ginit, &gate);
        !st) {
      return Status::Err(st.code, std::string("parallel gate: ") + st.message);
    }
    layer.gate = std::move(gate);
  }
  *out = std::move(layer);
  return Status::Ok();
}


Status new_from_branches(const Config& cfg, std::vector<Branch> branches,
                         std::optional<dense::Layer> gate, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "parallel: nil output");
  Config checked = cfg;
  checked.branches = int(branches.size());
  if (auto st = validate(&checked); !st) return st;
  if ((checked.combine == Combine::Filter && !gate) ||
      (gate && (checked.combine != Combine::Filter || gate->input_height != checked.dim ||
                gate->output_height != checked.branches))) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: Filter gate shape");
  }
  out->cfg = checked;
  out->branches = std::move(branches);
  out->gate = std::move(gate);
  return Status::Ok();
}

dense::Layer* dense_branch(Layer& layer, int i) {
  return i >= 0 && i < int(layer.branches.size()) && layer.branches[i].kind == BranchKind::Dense
             ? &layer.branches[i].dense_op : nullptr;
}

const dense::Layer* dense_branch(const Layer& layer, int i) {
  return i >= 0 && i < int(layer.branches.size()) && layer.branches[i].kind == BranchKind::Dense
             ? &layer.branches[i].dense_op : nullptr;
}

namespace {
Status execute(Layer& layer, const TensorF32& input, const TensorF32* grad_out,
               TensorF32* pre, TensorF32* post, TensorF32* grad_in, TensorF32* grad_w) {
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  if (input.shape.size() < 2 || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: invalid input");
  }
  const int nb = int(layer.branches.size());
  if (nb == 0 || nb != cfg.branches) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: branch count");
  }
  const bool standard = (input.shape.size() == 2 || input.shape.size() == 3) &&
                        input.shape.back() == cfg.dim;
  if (cfg.seq_len > 0 && (!standard || input.shape.size() != 3 ||
                         input.shape[1] != cfg.seq_len)) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: sequence shape");
  }
  int n = int(input.shape[0]);
  if (standard && input.shape.size() == 3) n *= input.shape[1];
  if (n <= 0) return Status::Err(StatusCode::InvalidArgument, "parallel: batch/rows");
  TensorF32 flat = input;
  if (standard) {
    flat.shape = {n, cfg.dim};
    if (flat.data.size() != size_t(n * cfg.dim)) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: input size");
    }
  }
  std::vector<TensorF32> branch_pre(nb), branch_out(nb);
  std::vector<int32_t> widths;
  std::vector<float> packed;
  int od = 0;
  for (int i = 0; i < nb; ++i) {
    auto& branch = layer.branches[i];
    if (!standard && branch.kind == BranchKind::Dense) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: Dense input layout");
    }
    const auto& x = branch.kind == BranchKind::Dense ? flat : input;
    if (auto st = stack::child_forward(branch, x, &branch_pre[i], &branch_out[i]); !st) return st;
    if (branch_out[i].data.empty() || branch_out[i].data.size() % n) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: output layout");
    }
    const int width = int(branch_out[i].data.size()) / n;
    if (i && cfg.combine != Combine::Concat && width != widths[0]) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: branch width mismatch");
    }
    widths.push_back(width);
    packed.insert(packed.end(), branch_out[i].data.begin(), branch_out[i].data.end());
    od += width;
  }
  if (cfg.combine != Combine::Concat) od = widths[0];
  if (cfg.out_feat > 0 && out_dim(cfg) != od) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: output width");
  }
  TensorF32 gate_pre, gate_out;
  if (cfg.combine == Combine::Filter) {
    if (!layer.gate || !standard || layer.gate->input_height != cfg.dim ||
        layer.gate->output_height != nb) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: gate layout");
    }
    if (auto st = dense::forward_f32(*layer.gate, flat, &gate_pre, &gate_out); !st) return st;
  }
  std::vector<float> combined(n * od), branch_grads(packed.size()), gate_grad(n * nb), scratch(4 * nb);
  const auto& gy = grad_out ? grad_out->data : combined;
  if (grad_out && gy.size() != combined.size()) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: gradient size");
  }
  const double beta = disagree_beta_eff(cfg);
  uint64_t beta_bits;
  std::memcpy(&beta_bits, &beta, sizeof(beta_bits)); // ABI packing, no host arithmetic.
  if (auto st = metal::run("op_parallel",
          {metal::read(packed), metal::read(widths), metal::read(gate_out.data), metal::read(gy),
           metal::write(combined), metal::write(branch_grads), metal::write(gate_grad),
           metal::write(scratch)},
          {n, nb, od, int(cfg.combine), sparse_k_eff(cfg), grad_out ? 1 : 0,
           int32_t(uint32_t(beta_bits)), int32_t(uint32_t(beta_bits >> 32))}); !st) return st;
  if (pre) *pre = input;
  if (post) {
    post->data = std::move(combined);
    // Go unflattenFeat keeps a singleton row axis for non-matrix inputs.
    post->shape = input.shape.size() == 2
                      ? std::vector<int64_t>{input.shape[0], od}
                      : std::vector<int64_t>{input.shape[0], n / int(input.shape[0]), od};
  }
  if (!grad_out) return Status::Ok();
  grad_in->shape = input.shape;
  grad_in->data.resize(input.data.size());
  grad_w->data.clear();
  std::vector<float> input_grads;
  // Packing and padding only; the reduction and final cast run on Metal.
  auto collect = [&](const TensorF32& gx) {
    const size_t start = input_grads.size();
    input_grads.resize(start + input.data.size(), 0);
    std::copy_n(gx.data.begin(), std::min(gx.data.size(), input.data.size()),
                input_grads.begin() + start);
  };
  size_t offset = 0;
  for (int i = 0; i < nb; ++i) {
    TensorF32 branch_gy, gx, gw;
    branch_gy.shape = branch_out[i].shape;
    branch_gy.data.assign(branch_grads.begin() + offset,
                          branch_grads.begin() + offset + branch_out[i].data.size());
    offset += branch_out[i].data.size();
    auto& branch = layer.branches[i];
    const auto& x = branch.kind == BranchKind::Dense ? flat : input;
    if (auto st = stack::child_backward(branch, branch_gy, x, branch_pre[i], &gx, &gw); !st) return st;
    collect(gx);
    grad_w->data.insert(grad_w->data.end(), gw.data.begin(), gw.data.end());
  }
  if (cfg.combine == Combine::Filter) {
    TensorF32 gate_gy, gx, gw;
    gate_gy.shape = gate_out.shape;
    gate_gy.data = std::move(gate_grad);
    if (auto st = dense::backward_f32(*layer.gate, gate_gy, flat, gate_pre, &gx, &gw); !st) return st;
    collect(gx);
    grad_w->data.insert(grad_w->data.end(), gw.data.begin(), gw.data.end());
  }
  const int groups = nb + (cfg.combine == Combine::Filter ? 1 : 0);
  if (auto st = metal::run("op_parallel_sum", {metal::read(input_grads), metal::write(grad_in->data)},
                          {int(input.data.size()), groups, n * cfg.dim}, {}, input.data.size()); !st)
    return st;
  grad_w->shape = {int64_t(grad_w->data.size())};
  return Status::Ok();
}
} // namespace

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) return Status::Err(StatusCode::InvalidArgument, "parallel: nil output");
  return execute(layer, input, nullptr, pre, post, nullptr, nullptr);
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32&, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: nil gradient");
  }
  return execute(layer, input, &grad_out, nullptr, nullptr, grad_in, grad_w);
}

int grad_w_size(const Layer& layer) {
  int size = 0;
  for (const auto& branch : layer.branches) size += stack::child_grad_w_size(branch);
  if (layer.gate) size += layer.gate->weights.rows * layer.gate->weights.cols;
  return size;
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst && grad_w_size(layer) > 0) return Status::Err(StatusCode::InvalidArgument, "parallel: nil export");
  int offset = 0;
  for (const auto& branch : layer.branches) {
    if (auto st = stack::child_export_f32(branch, dst + offset); !st) return st;
    offset += stack::child_grad_w_size(branch);
  }
  if (layer.gate) return layer.gate->weights.export_f32(dst + offset);
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& grad_w, double lr) {
  if (!layer) return Status::Err(StatusCode::InvalidArgument, "parallel: nil layer");
  if (grad_w.data.size() != size_t(grad_w_size(*layer))) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: gradient size");
  }
  int offset = 0;
  for (auto& branch : layer->branches) {
    const int n = stack::child_grad_w_size(branch);
    TensorF32 slice;
    slice.shape = {n};
    slice.data.assign(grad_w.data.begin() + offset, grad_w.data.begin() + offset + n);
    offset += n;
    if (auto st = stack::child_apply_sgd(&branch, slice, lr); !st) return st;
  }
  if (layer->gate) {
    TensorF32 slice;
    slice.shape = {int64_t(grad_w.data.size() - offset)};
    slice.data.assign(grad_w.data.begin() + offset, grad_w.data.end());
    return dense::apply_grad_sgd(&*layer->gate, slice, lr);
  }
  return Status::Ok();
}
} // namespace welvet::parallel
