#include "welvet/cam_sync.hpp"
#include "welvet/metal/ops.hpp"
#include <algorithm>
#include <cmath>
#include <string>

namespace welvet::cameral {
void collect_stores(stack::Child &c, std::vector<StoreRef> *out) {
  auto dense = [&](dense::Layer &d) { out->push_back({&d.weights, &d}); };
  switch (c.kind) {
  case stack::Kind::Dense:
    dense(c.dense_op);
    break;
  case stack::Kind::RMSNorm:
    out->push_back({&c.rmsnorm_op.gamma});
    break;
  case stack::Kind::LayerNorm:
    out->push_back({&c.layernorm_op.gamma});
    out->push_back({&c.layernorm_op.beta});
    break;
  case stack::Kind::SwiGLU:
    dense(c.swiglu_op.Gate);
    dense(c.swiglu_op.Up);
    dense(c.swiglu_op.Down);
    break;
  case stack::Kind::Embedding:
    out->push_back({&c.embedding_op.weights});
    break;
  case stack::Kind::MHA:
    dense(c.mha_op.Q);
    dense(c.mha_op.K);
    dense(c.mha_op.V);
    dense(c.mha_op.O);
    break;
  case stack::Kind::CNN1:
    dense(c.cnn1_op.proj);
    break;
  case stack::Kind::CNN2:
    dense(c.cnn2_op.proj);
    break;
  case stack::Kind::CNN3:
    dense(c.cnn3_op.proj);
    break;
  case stack::Kind::ConvT1:
    dense(c.convt1_op.proj);
    break;
  case stack::Kind::ConvT2:
    dense(c.convt2_op.proj);
    break;
  case stack::Kind::ConvT3:
    dense(c.convt3_op.proj);
    break;
  case stack::Kind::RNN:
    dense(c.rnn_op.IH);
    dense(c.rnn_op.HH);
    break;
  case stack::Kind::LSTM:
    for (auto *gate :
         {&c.lstm_op.I, &c.lstm_op.F, &c.lstm_op.G, &c.lstm_op.O}) {
      dense(gate->IH);
      dense(gate->HH);
    }
    break;
  case stack::Kind::Sequential:
    for (auto &d : c.sequential_op.children)
      dense(d);
    break;
  case stack::Kind::Residual:
    for (auto &d : c.residual_op.children)
      dense(d);
    break;
  case stack::Kind::KMeans:
    dense(c.kmeans_op.centers);
    break;
  case stack::Kind::Mamba:
    dense(c.mamba_op.in_proj);
    dense(c.mamba_op.out_proj);
    break;
  case stack::Kind::Metacognition:
    dense(c.metacognition_op.observed);
    break;
  case stack::Kind::NestedStack:
  case stack::Kind::Parallel:
    for (auto &ch : c.nested)
      collect_stores(ch, out);
    if (c.kind == stack::Kind::Parallel && c.parallel_nest.has_gate)
      dense(c.dense_op);
    break;
  default:
    break; // Go Softmax/GDN own no blendable weights.Store.
  }
}
Status blend_stores(const std::vector<StoreRef> &refs,
                    const std::vector<double> &alphas) {
  if (refs.size() < 2)
    return Status::Ok();
  if (refs.size() != alphas.size())
    return Status::Err(StatusCode::InvalidArgument, "BlendStores alpha count");
  WeightStore *ref = refs[0].store;
  if (!ref)
    return Status::Err(StatusCode::InvalidArgument, "BlendStores nil store");
  const int n = ref->rows * ref->cols;
  if (n <= 0)
    return Status::Ok();
  std::vector<float> packed;
  bool any_write = false;
  int bias_n = int(ref->bias.size());
  for (size_t i = 0; i < refs.size(); ++i) {
    auto *s = refs[i].store;
    if (!s || s->rows != ref->rows || s->cols != ref->cols ||
        !std::isfinite(alphas[i]))
      return Status::Err(StatusCode::InvalidArgument,
                         "BlendStores shape/alpha");
    any_write |= alphas[i] > 0;
    size_t off = packed.size();
    packed.resize(off + n);
    if (auto st = s->export_f32(packed.data() + off); !st)
      return st;
    if (int(s->bias.size()) != bias_n)
      bias_n = -1;
  }
  if (!any_write)
    return Status::Ok();
  std::vector<float> blended(packed.size());
  if (auto st = metal::run(
          "op_cam_blend",
          {metal::read(packed), metal::read(alphas), metal::write(blended)},
          {n, int(refs.size())}, {}, n);
      !st)
    return st;
  std::vector<double> biases, blended_bias;
  if (bias_n > 0) {
    for (auto r : refs)
      biases.insert(biases.end(), r.store->bias.begin(), r.store->bias.end());
    blended_bias.resize(biases.size());
    if (auto st = metal::run("op_cam_bias_blend",
                             {metal::read(biases), metal::read(alphas),
                              metal::write(blended_bias)},
                             {bias_n, int(refs.size())}, {}, bias_n);
        !st)
      return st;
  }
  for (size_t i = 0; i < refs.size(); ++i) {
    if (alphas[i] <= 0)
      continue;
    std::vector<float> flat(blended.begin() + i * n,
                            blended.begin() + (i + 1) * n);
    if (auto st = refs[i].store->set_from_f32_gpu(flat); !st)
      return st;
    if (bias_n > 0)
      refs[i].store->bias.assign(blended_bias.begin() + i * bias_n,
                                 blended_bias.begin() + (i + 1) * bias_n);
    if (refs[i].dense)
      refs[i].dense->invalidate_gpu();
  }
  return Status::Ok();
}
namespace {
double alpha(const SyncConfig &cfg) {
  return cfg.alpha <= 0 ? 1 : std::min(1.0, cfg.alpha);
}
Status sync_branches(std::vector<stack::Child> &branches, Controls &ctl) {
  if (!ctl.sync || !ctl.sync->enabled)
    return Status::Ok();
  auto &cfg = *ctl.sync;
  auto groups = cfg.groups;
  if (groups.empty()) {
    groups.resize(1);
    for (int i = 0; i < int(branches.size()); ++i)
      groups[0].push_back(i);
  }
  for (auto &group : groups) {
    std::vector<std::vector<StoreRef>> stores;
    std::vector<double> rates;
    size_t slots = 0;
    for (int i : group) {
      if (i < 0 || i >= int(branches.size()))
        continue;
      auto mode = resolve_train_mode(
          i < int(ctl.modes.size()) ? ctl.modes[i] : TrainMode::Inherit,
          TrainMode::NormalBP);
      if (train_mode_is_frozen(mode))
        continue;
      stores.emplace_back();
      collect_stores(branches[i], &stores.back());
      slots = std::max(slots, stores.back().size());
      rates.push_back(i < int(cfg.branch_alpha.size()) ? cfg.branch_alpha[i]
                                                       : alpha(cfg));
    }
    for (size_t slot = 0; slot < slots; ++slot) {
      std::vector<StoreRef> clique;
      std::vector<double> writes;
      for (size_t i = 0; i < stores.size(); ++i) {
        if (slot >= stores[i].size())
          continue;
        auto r = stores[i][slot];
        if (!clique.empty() && (r.store->rows != clique[0].store->rows ||
                                r.store->cols != clique[0].store->cols)) {
          clique.clear();
          break;
        }
        clique.push_back(r);
        writes.push_back(rates[i]);
      }
      if (auto st = blend_stores(clique, writes); !st)
        return st;
    }
  }
  return Status::Ok();
}
Status endpoint(std::vector<stack::Child> &children, Endpoint e,
                StoreRef *out) {
  if (e.stack_idx < 0 || e.stack_idx >= int(children.size()))
    return Status::Err(StatusCode::InvalidArgument, "CamSync stack index");
  auto &c = children[e.stack_idx];
  if (c.kind == stack::Kind::NestedStack)
    return endpoint(c.nested, {e.branch, e.store, 0}, out);
  stack::Child *source = &c;
  if (c.kind == stack::Kind::Parallel) {
    if (e.branch < 0 || e.branch >= int(c.nested.size()))
      return Status::Err(StatusCode::InvalidArgument, "CamSync branch index");
    source = &c.nested[e.branch];
  }
  std::vector<StoreRef> stores;
  collect_stores(*source, &stores);
  if (e.store < 0 || e.store >= int(stores.size()))
    return Status::Err(StatusCode::InvalidArgument, "CamSync store index");
  *out = stores[e.store];
  return Status::Ok();
}
Status sync_stack(std::vector<stack::Child> &children, Controls &ctl) {
  if (!ctl.sync || !ctl.sync->enabled)
    return Status::Ok();
  for (auto &c : children)
    if (c.kind == stack::Kind::Parallel) {
      if (!c.controls.sync) {
        c.controls.sync = *ctl.sync;
        c.controls.sync->cross.clear();
      }
      if (auto st = sync_branches(c.nested, c.controls); !st)
        return st;
    }
  for (auto &pair : ctl.sync->cross) {
    StoreRef a, b;
    if (auto st = endpoint(children, pair.a, &a); !st)
      return st;
    if (auto st = endpoint(children, pair.b, &b); !st)
      return st;
    if (auto st = blend_stores({a, b}, {alpha(*ctl.sync), alpha(*ctl.sync)});
        !st)
      return st;
  }
  return Status::Ok();
}
Status maybe_children(std::vector<stack::Child> &children, SyncWhen when) {
  for (auto &c : children) {
    if (c.kind == stack::Kind::Parallel && c.controls.sync &&
        c.controls.sync->enabled && c.controls.sync->when == when) {
      if (auto st = sync_branches(c.nested, c.controls); !st)
        return st;
    }
    if (c.kind == stack::Kind::NestedStack) {
      if (c.controls.sync && c.controls.sync->enabled &&
          c.controls.sync->when == when) {
        if (auto st = sync_stack(c.nested, c.controls); !st)
          return st;
      } else if (auto st = maybe_children(c.nested, when); !st)
        return st;
    }
  }
  return Status::Ok();
}
} // namespace
Status sync_now(parallel::Layer &l) {
  return sync_branches(l.branches, l.controls);
}
Status sync_now(stack::Layer &l) { return sync_stack(l.children, l.controls); }
Status maybe_sync(parallel::Layer &l, SyncWhen when) {
  if (l.controls.sync && l.controls.sync->enabled &&
      l.controls.sync->when == when)
    return sync_now(l);
  return Status::Ok();
}
Status maybe_sync(stack::Layer &l, SyncWhen when) {
  if (l.controls.sync && l.controls.sync->enabled &&
      l.controls.sync->when == when)
    return sync_now(l);
  return maybe_children(l.children, when);
}
void set_sync(stack::Layer &l, const SyncConfig &cfg) {
  l.controls.sync = cfg;
  for (auto &c : l.children)
    if (c.kind == stack::Kind::Parallel && !c.controls.sync) {
      c.controls.sync = cfg;
      c.controls.sync->cross.clear();
    }
}
} // namespace welvet::cameral
