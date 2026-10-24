#include "welvet/dense_session.hpp"

#include "welvet/dense.hpp"
#include "welvet/metal/matvec.hpp"
#include "welvet/weights.hpp"

#include <string>
#include <vector>

namespace welvet::dense {

struct Session {
  metal::ResidentDense gpu;
  WeightStore weights; // keeps FormatNone truth; GPU holds soft-double mirror
  Activation act = Activation::Linear;
  int32_t batch = 0;
  int32_t in = 0;
  int32_t out = 0;
};

static Status decode_W_softdouble(const WeightStore& W, std::vector<float>* out_xy) {
  out_xy->resize(static_cast<std::size_t>(W.rows * W.cols) * 2);
  std::vector<double> row(static_cast<std::size_t>(W.cols));
  for (int r = 0; r < W.rows; ++r) {
    if (auto st = W.decode_row_f64(r, row.data()); !st) return st;
    for (int c = 0; c < W.cols; ++c) {
      float hi = 0, lo = 0;
      metal::split_double(row[static_cast<std::size_t>(c)], &hi, &lo);
      const std::size_t i = static_cast<std::size_t>(r * W.cols + c) * 2;
      (*out_xy)[i] = hi;
      (*out_xy)[i + 1] = lo;
    }
  }
  return Status::Ok();
}

Status session_mount(int32_t batch, int32_t in, int32_t out, Activation act, DType dt,
                     const float* W_init, const double* bias, size_t bias_len, Session** sess) {
  if (!sess || !W_init) return Status::Err(StatusCode::InvalidArgument, "null mount args");
  if (!dtype_supported_metal_forward(dt)) {
    return Status::Err(StatusCode::Unsupported,
                       std::string("dense Metal unsupported dtype ") + std::string(dtype_name(dt)));
  }
  if (auto st = metal::require_dense_compute(); !st) return st;

  auto* s = new Session();
  s->batch = batch;
  s->in = in;
  s->out = out;
  s->act = act;
  if (auto st = WeightStore::make(out, in, dt, W_init, &s->weights); !st) {
    delete s;
    return st;
  }
  if (bias && bias_len > 0) {
    s->weights.bias.assign(bias, bias + bias_len);
  }

  std::vector<float> W_xy;
  if (auto st = decode_W_softdouble(s->weights, &W_xy); !st) {
    delete s;
    return st;
  }
  std::vector<float> bias_xy;
  const int bl = static_cast<int>(s->weights.bias.size());
  if (bl > 0) {
    bias_xy.resize(static_cast<std::size_t>(bl) * 2);
    for (int i = 0; i < bl; ++i) {
      float hi = 0, lo = 0;
      metal::split_double(s->weights.bias[static_cast<std::size_t>(i)], &hi, &lo);
      bias_xy[static_cast<std::size_t>(i) * 2] = hi;
      bias_xy[static_cast<std::size_t>(i) * 2 + 1] = lo;
    }
  }

  if (auto st = metal::resident_mount(&s->gpu, W_xy.data(), bl ? bias_xy.data() : nullptr, bl,
                                      batch, out, in, act);
      !st) {
    delete s;
    return st;
  }
  *sess = s;
  return Status::Ok();
}

Status session_execute(Session* s, const float* x, float* pre, float* post) {
  if (!s || !x || !pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "null execute args");
  }
  return metal::resident_execute(&s->gpu, x, pre, post);
}

void session_release(Session* s) {
  if (!s) return;
  metal::resident_release(&s->gpu);
  delete s;
}

} // namespace welvet::dense
