#include "welvet/parallel.hpp"

#include "welvet/activate.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace welvet::parallel {
namespace {

struct Layout {
  int batch = 0;
  int rows = 0; // seq or 1
  int dim = 0;
  std::vector<int64_t> shape;
};

} // namespace

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
  if (cfg->seq_len < 0 || cfg->sparse_k < 0) {
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
    const float* init = nullptr;
    if (packed_or_null) init = packed_or_null + i * child_n;
    if (auto st = dense::new_layer(cfg.dim, cfg.out_feat, Activation::Linear, dt, init,
                                   &layer.branches[static_cast<std::size_t>(i)]);
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

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "parallel.forward: Metal kernel not implemented; CPU execution is forbidden");
}

} // namespace welvet::parallel
