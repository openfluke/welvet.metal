#include "welvet/cam_sync.hpp"
#include "welvet/interop/model_wire.hpp"
#include "welvet/metal/device.hpp"
#include "welvet/runtime/cameral_train.hpp"
#include "welvet/runtime/step.hpp"
#include "welvet/systems/tween.hpp"
#include "welvet/systems/tween_alt.hpp"
#include "welvet/systems/tween_split.hpp"
#include "welvet_c.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

using namespace welvet;
namespace {
struct Reader {
  const uint8_t *p;
  size_t left;
  int depth = 0;
  bool allow_count(size_t n) const { return n <= left / 4; }
  template <class T> bool read(T *out, size_t n) {
    if (n > left / sizeof(T))
      return false;
    size_t bytes = n * sizeof(T);
    if (bytes)
      std::memcpy(out, p, bytes);
    p += bytes;
    left -= bytes;
    return true;
  }
  template <class T> T one() {
    T v{};
    if (!read(&v, 1))
      throw std::invalid_argument("truncated descriptor");
    return v;
  }
  template <class T> std::vector<T> vec() {
    int32_t n = one<int32_t>();
    if (n < 0 || size_t(n) > left / sizeof(T))
      throw std::invalid_argument("invalid vector length");
    std::vector<T> v(n);
    if (!read(v.data(), v.size()))
      throw std::invalid_argument("short vector");
    return v;
  }
  void end() {
    if (left)
      throw std::invalid_argument("trailing descriptor bytes");
  }
};
struct Writer {
  std::vector<uint8_t> data;
  template <class T> void one(T v) {
    const auto *p = reinterpret_cast<const uint8_t *>(&v);
    data.insert(data.end(), p, p + sizeof(T));
  }
  template <class T> void vec(const std::vector<T> &v) {
    one(int32_t(v.size()));
    const auto *p = reinterpret_cast<const uint8_t *>(v.data());
    if (!v.empty())
      data.insert(data.end(), p, p + v.size() * sizeof(T));
  }
};
void require(bool ok, const char *why) {
  if (!ok)
    throw std::invalid_argument(why);
}
struct NativeStatusError : std::runtime_error {
  int code;
  explicit NativeStatusError(Status s)
      : std::runtime_error(s.message), code(int(s.code)) {}
};
void good(Status s) {
  if (!s)
    throw NativeStatusError(std::move(s));
}
welvet_status status(int code, const char *text) {
  welvet_status s{};
  s.code = code;
  if (text)
    std::snprintf(s.message, sizeof(s.message), "%s", text);
  return s;
}
template <class F> welvet_status guarded(F fn) {
  try {
    fn();
    return status(0, nullptr);
  } catch (const NativeStatusError &e) {
    return status(e.code, e.what());
  } catch (const std::invalid_argument &e) {
    return status(1, e.what());
  } catch (const std::exception &e) {
    return status(4, e.what());
  } catch (...) {
    return status(4, "unknown native session failure");
  }
}
TensorF32 tensor(Reader &r) {
  TensorF32 t;
  int32_t rank = r.one<int32_t>();
  require(rank > 0 && rank <= 5, "tensor rank must be 1..5");
  size_t n = 1;
  for (int i = 0; i < rank; ++i) {
    int d = r.one<int32_t>();
    require(d > 0 && n <= size_t(1 << 26) / d, "invalid tensor shape");
    n *= d;
    t.shape.push_back(d);
  }
  require(n <= r.left / 4, "short tensor data");
  t.data.resize(n);
  require(r.read(t.data.data(), n), "short tensor");
  return t;
}
void tensor_out(Writer &w, const TensorF32 &t) {
  w.one(int32_t(t.shape.size()));
  for (auto d : t.shape)
    w.one(int32_t(d));
  w.vec(t.data);
}
void blob(Writer &w, welvet_metal_blob *out) {
  require(out, "null output");
  out->data = nullptr;
  out->size = 0;
  void *p = std::malloc(w.data.size());
  if (!p && !w.data.empty())
    throw std::bad_alloc();
  if (p)
    std::memcpy(p, w.data.data(), w.data.size());
  out->data = static_cast<uint8_t *>(p);
  out->size = w.data.size();
}
void read_store(Reader &r, WeightStore &s) {
  int dt = r.one<int32_t>(), rows = r.one<int32_t>(), cols = r.one<int32_t>();
  float scale = r.one<float>();
  require(dt >= 0 && dt <= 33 && rows == s.rows && cols == s.cols &&
              std::isfinite(scale),
          "native store metadata mismatch");
  auto f = r.vec<float>();
  auto raw = r.vec<uint8_t>();
  auto bias = r.vec<double>();
  WeightStore fresh;
  good(WeightStore::make(rows, cols, DType(dt), nullptr, &fresh));
  require(f.size() == fresh.f32.size() && raw.size() == fresh.native.size(),
          "native store payload length mismatch");
  fresh.f32 = std::move(f);
  fresh.native = std::move(raw);
  fresh.scale = scale;
  fresh.bias = std::move(bias);
  s = std::move(fresh);
}
void write_store(Writer &w, const WeightStore &s) {
  w.one(int32_t(s.dtype));
  w.one(int32_t(s.rows));
  w.one(int32_t(s.cols));
  w.one(s.scale);
  w.vec(s.f32);
  w.vec(s.native);
  w.vec(s.bias);
}
void extra_nodes(stack::Child &c, std::vector<stack::Child *> &extra) {
  if (c.kind == stack::Kind::Mamba || c.kind == stack::Kind::GDN)
    extra.push_back(&c);
  for (auto &n : c.nested)
    extra_nodes(n, extra);
}
} // namespace
struct welvet_model_session {
  std::mutex mu;
  int root = 0;
  stack::Layer stack;
  parallel::Layer parallel;
  architecture::Grid grid;
  // The native model, proxy caches and Line flight state live until release.
};
namespace {
template <class F> void with_children(welvet_model_session &s, F fn) {
  if (s.root == 0) {
    fn(s.stack.children);
    return;
  }
  std::vector<stack::Child> children;
  if (s.root == 1) {
    architecture::Cell c;
    c.op_kind = architecture::OpKind::Parallel;
    c.parallel_op = std::move(s.parallel);
    children.push_back(interop::wire::take_mesh_child(c));
  } else
    for (auto &c : s.grid.cells)
      children.push_back(interop::wire::take_mesh_child(c));
  auto restore = [&] {
    if (s.root == 1) {
      auto &c = children[0];
      auto p = c.parallel_nest;
      s.parallel.cfg = {p.dim,
                        p.out_feat,
                        int(c.nested.size()),
                        parallel::Combine(p.combine),
                        p.seq_len,
                        p.sparse_k,
                        p.disagree_beta};
      s.parallel.branches = std::move(c.nested);
      s.parallel.controls = std::move(c.controls);
      s.parallel.accel = std::move(c.accel);
      if (p.has_gate)
        s.parallel.gate = std::move(c.dense_op);
    } else {
      auto hops = architecture::hop_order(s.grid);
      for (size_t i = 0; i < children.size(); ++i) {
        auto meta = s.grid.cells[i].meta;
        if (children[i].kind != stack::Kind::None)
          good(interop::wire::bind_mesh_child(s.grid, hops[i],
                                              std::move(children[i])));
        s.grid.cells[i].meta = meta;
      }
    }
  };
  try {
    fn(children);
  } catch (...) {
    restore();
    throw;
  }
  restore();
}
Status forward(welvet_model_session &s, const TensorF32 &x, TensorF32 *post) {
  TensorF32 pre;
  if (s.root == 0)
    return stack::forward(s.stack, x, &pre, post);
  if (s.root == 1)
    return parallel::forward(s.parallel, x, &pre, post);
  runtime::forward::Result f;
  auto st = runtime::forward::volume_dense(s.grid, x, &f);
  *post = std::move(f.output);
  return st;
}
Status train(welvet_model_session &s, const TensorF32 &x, const TensorF32 &t,
             TrainMode mode, double lr, int ticks, double *loss) {
  if (s.root == 0)
    return runtime::cameral_train::train_stack_mse(s.stack, x, t, mode, lr,
                                                   loss);
  if (s.root == 1)
    return runtime::cameral_train::train_mse(s.parallel, x, t, mode, lr, loss);
  if (mode == TrainMode::MeshBP || mode == TrainMode::MeshTween ||
      mode == TrainMode::MeshTweenChain ||
      (int(mode) >= 20 && int(mode) <= 23)) {
    int credit = mode == TrainMode::MeshTween                 ? 1
                 : mode == TrainMode::MeshTweenSplit          ? 2
                 : mode == TrainMode::MeshTweenAlt            ? 3
                 : mode == TrainMode::MeshTweenSplitFastProxy ? 4
                 : mode == TrainMode::MeshTweenSplitSparse    ? 5
                                                              : 0;
    return runtime::step::train_mse(s.grid, x, t, ticks, lr, loss,
                                    runtime::step::Credit(credit));
  }
  if (mode == TrainMode::NormalBP || mode == TrainMode::Inherit ||
      mode == TrainMode::TweenChain || train_mode_is_frozen(mode) ||
      mode == TrainMode::Adversarial || mode == TrainMode::Memory)
    return runtime::cameral_train::train_volume_mse(
        s.grid, x, t,
        mode == TrainMode::TweenChain ? TrainMode::NormalBP : mode, lr, loss);
  if (mode != TrainMode::Tween && mode != TrainMode::TweenAlt &&
      mode != TrainMode::TweenSplit && !(int(mode) >= 14 && int(mode) <= 19))
    return Status::Err(StatusCode::Unsupported,
                       "this TrainMode requires a Stack/Parallel session or "
                       "explicit mesh mode");
  runtime::forward::Result f;
  auto st = runtime::forward::volume_dense(s.grid, x, &f);
  if (!st)
    return st;
  if (mode == TrainMode::Tween)
    return systems::tween::train_volume_mse(s.grid, f, t, lr, loss);
  if (mode == TrainMode::TweenAlt)
    return systems::tween_alt::train_volume_mse(s.grid, x, t, lr, loss);
  if (mode == TrainMode::TweenSplit || (int(mode) >= 14 && int(mode) <= 19))
    return systems::tween_split::train_volume_variant(
        s.grid, f, t, lr, loss,
        systems::tween_split::Variant(
            mode == TrainMode::TweenSplit ? 0 : int(mode) - 13));
  return Status::Err(
      StatusCode::Unsupported,
      "this TrainMode requires a Stack/Parallel session or explicit mesh mode");
}
} // namespace
extern "C" {
int32_t welvet_model_abi_version(void) { return 1; }
welvet_status welvet_model_mount(const uint8_t *descriptor, size_t size,
                                 welvet_model_session **out) {
  if (out)
    *out = nullptr;
  return guarded([&] {
    require(out && descriptor && size >= 28 && size <= (1u << 28),
            "invalid mount arguments");
    Reader r{descriptor, size};
    require(r.one<uint32_t>() == 0x314e4d57,
            "unsupported model descriptor version");
    auto s = std::make_unique<welvet_model_session>();
    s->root = r.one<int32_t>();
    int count = r.one<int32_t>();
    require(s->root >= 0 && s->root <= 2 && count > 0 && count <= 4096,
            "invalid model root/count");
    int dims[4];
    for (auto &d : dims)
      d = r.one<int32_t>();
    s->stack.children.resize(count);
    for (auto &c : s->stack.children)
      require(interop::wire::read_child(r, &c), "invalid model child");
    require(interop::wire::read_cam_controls(r, &s->stack.controls),
            "invalid root controls");
    for (auto &c : s->stack.children)
      require(interop::wire::read_cam_tree(r, &c), "invalid child controls");
    std::vector<cameral::StoreRef> stores;
    for (auto &c : s->stack.children)
      cameral::collect_stores(c, &stores);
    require(r.one<int32_t>() == int(stores.size()), "store count mismatch");
    for (auto ref : stores)
      read_store(r, *ref.store);
    if (s->stack.controls.sync)
      cameral::set_sync(s->stack, *s->stack.controls.sync);
    if (s->root == 1) {
      require(count == 1 && s->stack.children[0].kind == stack::Kind::Parallel,
              "invalid Parallel root");
      auto c = std::move(s->stack.children[0]);
      auto p = c.parallel_nest;
      s->parallel.cfg = {p.dim,
                         p.out_feat,
                         int(c.nested.size()),
                         parallel::Combine(p.combine),
                         p.seq_len,
                         p.sparse_k,
                         p.disagree_beta};
      s->parallel.branches = std::move(c.nested);
      s->parallel.controls = std::move(c.controls);
      if (p.has_gate)
        s->parallel.gate = std::move(c.dense_op);
      s->stack.children.clear();
    }
    if (s->root == 2) {
      int64_t n = 1;
      for (int d : dims) {
        require(d > 0 && d <= 4096 && n <= 4096 / d, "invalid grid dimensions");
        n *= d;
      }
      require(n == count, "grid cell count mismatch");
      s->grid = architecture::new_grid(dims[0], dims[1], dims[2], dims[3]);
      auto hops = architecture::hop_order(s->grid);
      for (int i = 0; i < count; ++i) {
        if (s->stack.children[i].kind != stack::Kind::None)
          good(interop::wire::bind_mesh_child(s->grid, hops[i],
                                              std::move(s->stack.children[i])));
        auto &cell = s->grid.cells[i];
        int enabled = r.one<int32_t>();
        require(!enabled || cell.op_kind != architecture::OpKind::None,
                "enabled grid cell has no operation");
        cell.meta.is_disabled = enabled == 0;
        cell.is_remote_link = r.one<int32_t>() != 0;
        cell.target_z = r.one<int32_t>();
        cell.target_y = r.one<int32_t>();
        cell.target_x = r.one<int32_t>();
        cell.target_l = r.one<int32_t>();
        if (cell.is_remote_link)
          require(architecture::at(s->grid, cell.target_z, cell.target_y,
                                   cell.target_x, cell.target_l),
                  "invalid remote link");
      }
      s->stack.children.clear();
    }
    r.end();
    good(metal::init_default_device());
    *out = s.release();
  });
}
welvet_status welvet_model_forward(welvet_model_session *s,
                                   const uint8_t *input, size_t size,
                                   welvet_metal_blob *out) {
  return guarded([&] {
    require(s && input && out, "invalid forward arguments");
    std::lock_guard<std::mutex> lock(s->mu);
    Reader r{input, size};
    auto x = tensor(r);
    r.end();
    TensorF32 post;
    good(forward(*s, x, &post));
    Writer w;
    tensor_out(w, post);
    blob(w, out);
  });
}
// Segment APIs are additive to model ABI 1. They never compute a local loss.
welvet_status welvet_model_forward_pair(welvet_model_session *s,
    const uint8_t *data, size_t size, welvet_metal_blob *out) {
  return guarded([&] {
    require(s && data && out, "invalid segment forward arguments");
    std::lock_guard<std::mutex> lock(s->mu);
    require(s->root != 2, "segment forward expects Stack or Parallel");
    Reader r{data, size}; auto x = tensor(r); r.end();
    TensorF32 pre, post;
    good(s->root == 0 ? stack::forward(s->stack, x, &pre, &post)
                      : parallel::forward(s->parallel, x, &pre, &post));
    Writer w; tensor_out(w, pre); tensor_out(w, post); blob(w, out);
  });
}
welvet_status welvet_model_backward(welvet_model_session *s,
    const uint8_t *data, size_t size, welvet_metal_blob *out) {
  return guarded([&] {
    require(s && data && out, "invalid segment backward arguments");
    std::lock_guard<std::mutex> lock(s->mu);
    require(s->root != 2, "segment backward expects Stack or Parallel");
    Reader r{data, size}; auto x = tensor(r), gy = tensor(r); r.end();
    TensorF32 pre, post, gx, dw;
    good(s->root == 0 ? stack::forward(s->stack, x, &pre, &post)
                      : parallel::forward(s->parallel, x, &pre, &post));
    require(gy.shape == post.shape && gy.data.size() == post.data.size(),
            "segment output gradient shape mismatch");
    good(s->root == 0 ? stack::backward_f32(s->stack, gy, x, pre, &gx, &dw)
                      : parallel::backward_f32(s->parallel, gy, x, pre, &gx, &dw));
    if (dw.shape.empty()) dw.shape = {int(dw.data.size())};
    Writer w; tensor_out(w, gx); tensor_out(w, dw); blob(w, out);
  });
}
welvet_status welvet_model_train_credit(welvet_model_session *s, int32_t mode,
    double lr, const uint8_t *data, size_t size) {
  return guarded([&] {
    require(s && data && mode >= 0 && mode <= 33 && std::isfinite(lr), "invalid credit arguments");
    std::lock_guard<std::mutex> lock(s->mu);
    require(s->root != 2, "credit expects Stack or Parallel");
    Reader r{data, size}; auto x=tensor(r), gy=tensor(r); TensorF32 pre, post;
    int present=r.one<int32_t>();require(present==0 || present==1,"invalid pre flag");
    if(present)pre=tensor(r);r.end();
    if (train_mode_is_frozen(TrainMode(mode))) return;
    if (!present) good(s->root==0 ? stack::forward(s->stack,x,&pre,&post)
                                 : parallel::forward(s->parallel,x,&pre,&post));
    if (s->root==1) {
      good(runtime::cameral_train::train_parallel(s->parallel,gy,x,pre,TrainMode(mode),lr));
      return;
    }
    stack::Child c; c.kind=stack::Kind::NestedStack;
    c.nested=std::move(s->stack.children);c.controls=std::move(s->stack.controls);c.accel=std::move(s->stack.accel);
    auto restore=[&]{s->stack.children=std::move(c.nested);s->stack.controls=std::move(c.controls);s->stack.accel=std::move(c.accel);};
    Status st;
    try {TensorF32 gx;st=runtime::cameral_train::train_child(c,gy,x,pre,post,TrainMode(mode),lr,&gx);}
    catch(...){restore();throw;}
    restore();good(st);
  });
}
welvet_status welvet_model_dense_proxy(welvet_model_session *s, int32_t kind,
    const uint8_t *data, size_t size, welvet_metal_blob *out) {
  return guarded([&] {
    require(s && data && out && kind >= 0 && kind <= 1, "invalid Dense proxy arguments");
    std::lock_guard<std::mutex> lock(s->mu);
    require(s->root == 0 && s->stack.children.size() == 1 &&
            s->stack.children[0].kind == stack::Kind::Dense, "proxy expects one Dense");
    Reader r{data, size}; auto x = tensor(r), gy = tensor(r); TensorF32 pre, result;
    int present = r.one<int32_t>(); require(present == 0 || present == 1, "invalid pre flag");
    if (present) pre = tensor(r); r.end();
    auto &d = s->stack.children[0].dense_op;
    good(kind == 0 ? dense::linear_grad_in(d, gy, x, &result)
                   : dense::grad_w_only(d, gy, x, pre, &result));
    Writer w; tensor_out(w, result); blob(w, out);
  });
}
welvet_status welvet_model_apply_sgd(welvet_model_session *s, double lr,
    const uint8_t *data, size_t size) {
  return guarded([&] {
    require(s && data && std::isfinite(lr), "invalid segment SGD arguments");
    std::lock_guard<std::mutex> lock(s->mu);
    require(s->root != 2, "segment SGD expects Stack or Parallel");
    Reader r{data, size}; auto dw = tensor(r); r.end();
    const int n = s->root == 0 ? stack::grad_w_size(s->stack)
                             : parallel::grad_w_size(s->parallel);
    require(dw.data.size() == size_t(n), "segment weight gradient size mismatch");
    good(s->root == 0 ? stack::apply_grad_sgd(&s->stack, dw, lr)
                      : parallel::apply_grad_sgd(&s->parallel, dw, lr));
  });
}
welvet_status welvet_model_train_mse(welvet_model_session *s, int32_t mode,
                                     double lr, int32_t ticks,
                                     const uint8_t *data, size_t size,
                                     double *loss) {
  return guarded([&] {
    require(s && data && loss && mode >= 0 && mode <= 33 && std::isfinite(lr) &&
                ticks > 0 && ticks <= 4096,
            "invalid train arguments");
    std::lock_guard<std::mutex> lock(s->mu);
    Reader r{data, size};
    auto x = tensor(r), target = tensor(r);
    r.end();
    good(train(*s, x, target, TrainMode(mode), lr, ticks, loss));
  });
}
welvet_status welvet_model_export_weights(welvet_model_session *s,
                                          welvet_metal_blob *out) {
  return guarded([&] {
    require(s && out, "invalid export arguments");
    std::lock_guard<std::mutex> lock(s->mu);
    with_children(*s, [&](std::vector<stack::Child> &children) {
      std::vector<cameral::StoreRef> stores;
      std::vector<stack::Child *> extras;
      for (auto &c : children) {
        cameral::collect_stores(c, &stores);
        extra_nodes(c, extras);
      }
      Writer w;
      w.one(uint32_t(0x3153574e));
      w.one(int32_t(stores.size()));
      for (auto ref : stores)
        write_store(w, *ref.store);
      w.one(int32_t(extras.size()));
      for (auto *c : extras) {
        w.one(int32_t(c->kind));
        std::vector<float> p(stack::child_grad_w_size(*c));
        good(stack::child_export_f32(*c, p.data()));
        w.vec(p);
      }
      blob(w, out);
    });
  });
}
welvet_status welvet_model_sync(welvet_model_session *s, int32_t event) {
  return guarded([&] {
    require(s && event >= 0 && event <= 3, "invalid sync arguments");
    std::lock_guard<std::mutex> lock(s->mu);
    if (s->root == 0)
      good(event == 0
               ? cameral::sync_now(s->stack)
               : cameral::maybe_sync(s->stack, cameral::SyncWhen(event)));
    else if (s->root == 1)
      good(event == 0
               ? cameral::sync_now(s->parallel)
               : cameral::maybe_sync(s->parallel, cameral::SyncWhen(event)));
    else
      throw std::invalid_argument(
          "grid synchronization is configured per cell at mount");
  });
}
void welvet_model_release(welvet_model_session *s) { delete s; }
void welvet_metal_blob_free(welvet_metal_blob *b) {
  if (b) {
    std::free(b->data);
    b->data = nullptr;
    b->size = 0;
  }
}
}
