#include "welvet/dense.hpp"

#include "welvet/activate.hpp"
#include "welvet/metal/matvec.hpp"

#include <string>
#include <vector>

namespace welvet::dense {

Status new_layer(int in_h, int out_h, Activation act, DType dt, const float* init_or_null,
                 Layer* layer) {
  if (!layer || in_h <= 0 || out_h <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bad dims");
  }
  if (!dtype_supported_metal_forward(dt)) {
    return Status::Err(StatusCode::Unsupported,
                       std::string("dense Metal FormatNone forward unsupported for dtype ") +
                           std::string(dtype_name(dt)) +
                           " (Go baseline; Metal soft-double path not validated for this dtype)");
  }
  layer->input_height = in_h;
  layer->output_height = out_h;
  layer->activation = act;
  return WeightStore::make(out_h, in_h, dt, init_or_null, &layer->weights);
}

bool dtype_supported_metal_forward(DType dt) {
  // All 34 FormatNone dtypes Go supports — Metal soft-double GEMV targets the same
  // decode_row_f64 → f64-acc contract. If a dtype's encode/decode cannot match Go,
  // WeightStore::make / decode fails or we mark it unsupported here later.
  const int id = static_cast<int>(dt);
  return id >= 0 && id < kNumDTypes;
}

static Status decode_W_softdouble(const WeightStore& W, std::vector<float>* out_xy) {
  if (!out_xy) return Status::Err(StatusCode::InvalidArgument, "null W out");
  out_xy->resize(static_cast<std::size_t>(W.rows * W.cols) * 2);
  std::vector<double> row(static_cast<std::size_t>(W.cols));
  for (int r = 0; r < W.rows; ++r) {
    if (auto st = W.decode_row_f64(r, row.data()); !st) return st;
    for (int c = 0; c < W.cols; ++c) {
      float hi = 0, lo = 0;
      welvet::metal::split_double(row[static_cast<std::size_t>(c)], &hi, &lo);
      const std::size_t i = static_cast<std::size_t>(r * W.cols + c) * 2;
      (*out_xy)[i] = hi;
      (*out_xy)[i + 1] = lo;
    }
  }
  return Status::Ok();
}

Status forward_f32(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "dense: nil out");
  }
  if (!dtype_supported_metal_forward(layer.weights.dtype)) {
    return Status::Err(StatusCode::Unsupported,
                       std::string("dense Metal unsupported dtype ") +
                           std::string(dtype_name(layer.weights.dtype)));
  }
  // Metal only — no host MatVec, no f32-acc substitute.
  if (auto st = welvet::metal::require_dense_compute(); !st) return st;

  if (input.shape.size() < 2) {
    return Status::Err(StatusCode::InvalidArgument, "dense: input shape need [batch,in]");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int in = layer.input_height;
  const int out = layer.output_height;
  if (static_cast<int>(input.shape[1]) != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: input width mismatch");
  }
  if (layer.weights.rows != out || layer.weights.cols != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: weight shape mismatch");
  }

  *pre = TensorF32::zeros(batch, out);
  *post = TensorF32::zeros(batch, out);

  std::vector<float> W_xy;
  if (auto st = decode_W_softdouble(layer.weights, &W_xy); !st) return st;

  if (auto st = welvet::metal::dense_matvec_softdouble_batch(
          W_xy.data(), input.data.data(), pre->data.data(), batch, out, in);
      !st) {
    return st;
  }

  std::vector<float> bias_xy;
  const int bias_len = static_cast<int>(layer.weights.bias.size());
  if (bias_len > 0) {
    bias_xy.resize(static_cast<std::size_t>(bias_len) * 2);
    for (int i = 0; i < bias_len; ++i) {
      float hi = 0, lo = 0;
      welvet::metal::split_double(layer.weights.bias[static_cast<std::size_t>(i)], &hi, &lo);
      bias_xy[static_cast<std::size_t>(i) * 2] = hi;
      bias_xy[static_cast<std::size_t>(i) * 2 + 1] = lo;
    }
  }

  return welvet::metal::dense_bias_act(pre->data.data(), post->data.data(),
                                       bias_len ? bias_xy.data() : nullptr, bias_len, batch, out,
                                       layer.activation);
}

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "dense: nil grad out");
  }
  if (input.shape.size() < 2 || grad_out.shape.size() < 2 || pre.shape.size() < 2) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bad shapes");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int in = layer.input_height;
  const int out = layer.output_height;
  if (static_cast<int>(input.shape[1]) != in || static_cast<int>(grad_out.shape[0]) != batch ||
      static_cast<int>(grad_out.shape[1]) != out || static_cast<int>(pre.shape[0]) != batch ||
      static_cast<int>(pre.shape[1]) != out) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bwd shape mismatch");
  }
  if (layer.weights.rows != out || layer.weights.cols != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: weight shape mismatch");
  }

  std::vector<float> d_pre(static_cast<std::size_t>(batch * out));
  for (int i = 0; i < batch * out; ++i) {
    const double g = static_cast<double>(grad_out.data[static_cast<std::size_t>(i)]) *
                     activate_deriv_f64(static_cast<double>(pre.data[static_cast<std::size_t>(i)]),
                                        layer.activation);
    d_pre[static_cast<std::size_t>(i)] = static_cast<float>(g);
  }

  *grad_in = TensorF32::zeros(batch, in);
  *grad_w = TensorF32::zeros(out, in);

  for (int b = 0; b < batch; ++b) {
    const float* gy = d_pre.data() + static_cast<std::size_t>(b * out);
    float* gx = grad_in->data.data() + static_cast<std::size_t>(b * in);
    if (auto st = layer.weights.matvect_f32(gy, gx); !st) {
      return st;
    }
    const float* x_row = input.data.data() + static_cast<std::size_t>(b * in);
    for (int o = 0; o < out; ++o) {
      const double g = static_cast<double>(gy[o]);
      float* dw = grad_w->data.data() + static_cast<std::size_t>(o * in);
      for (int i = 0; i < in; ++i) {
        dw[i] = static_cast<float>(static_cast<double>(dw[i]) + g * static_cast<double>(x_row[i]));
      }
    }
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "dense: ApplyGradSGD nil");
  }
  const int n = layer->weights.rows * layer->weights.cols;
  if (static_cast<int>(dW.data.size()) < n) {
    return Status::Err(StatusCode::InvalidArgument, "dense: dW short");
  }
  std::vector<double> buf(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    buf[static_cast<std::size_t>(i)] = static_cast<double>(dW.data[static_cast<std::size_t>(i)]);
  }
  return layer->weights.apply_sgd(buf.data(), lr);
}

} // namespace welvet::dense
