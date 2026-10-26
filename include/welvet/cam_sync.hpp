#pragma once
#include "welvet/parallel.hpp"
#include "welvet/weights.hpp"
namespace welvet::cameral {
struct StoreRef {
  WeightStore *store = nullptr;
  dense::Layer *dense = nullptr;
};
void collect_stores(stack::Child &, std::vector<StoreRef> *);
Status blend_stores(const std::vector<StoreRef> &, const std::vector<double> &);
Status sync_now(parallel::Layer &);
Status sync_now(stack::Layer &);
Status maybe_sync(parallel::Layer &, SyncWhen);
Status maybe_sync(stack::Layer &, SyncWhen);
void set_sync(stack::Layer &, const SyncConfig &);
} // namespace welvet::cameral
