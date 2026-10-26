#include "welvet/dense.hpp"
#include "welvet/metal/layer_utils.hpp"
#include "welvet/metal/training.hpp"

#include "welvet/activate.hpp"
#include "welvet/metal/matvec.hpp"
#include "welvet/metal/ops.hpp"

#include <string>
#include <vector>

namespace welvet::dense {
namespace {

Status decode_W_softdouble(const WeightStore &W, std::vector<float> *out_xy) {
  if (!out_xy)
    return Status::Err(StatusCode::InvalidArgument, "null W out");
  out_xy->resize(static_cast<std::size_t>(W.rows * W.cols) * 2);
  std::vector<double> row(static_cast<std::size_t>(W.cols));
  for (int r = 0; r < W.rows; ++r) {
    if (auto st = W.decode_row_f64(r, row.data()); !st)
      return st;
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

Status decode_bias_softdouble(const WeightStore &W, std::vector<float> *bias_xy) {
  const int bl = static_cast<int>(W.bias.size());
  if (bl <= 0) {
    bias_xy->clear();
    return Status::Ok();
  }
  bias_xy->resize(static_cast<std::size_t>(bl) * 2);
  for (int i = 0; i < bl; ++i) {
    float hi = 0, lo = 0;
    welvet::metal::split_double(W.bias[static_cast<std::size_t>(i)], &hi, &lo);
    (*bias_xy)[static_cast<std::size_t>(i) * 2] = hi;
    (*bias_xy)[static_cast<std::size_t>(i) * 2 + 1] = lo;
  }
  return Status::Ok();
}

Status ensure_resident(const Layer &layer, int batch) {
  if (batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bad batch for resident");
  }
  if (auto st = welvet::metal::require_dense_compute(); !st)
    return st;

  const int in = layer.input_height;
  const int out = layer.output_height;
  const int bias_len = static_cast<int>(layer.weights.bias.size());

  const bool shape_ok = layer.gpu.buf_w && layer.gpu_batch == batch && layer.gpu.rows == out &&
                        layer.gpu.cols == in && layer.gpu.bias_len == bias_len &&
                        layer.gpu.act == layer.activation;

  // Hot path: resident W already matches FormatNone truth.
  if (layer.gpu_valid && shape_ok) {
    return Status::Ok();
  }

  std::vector<float> W_xy;
  if (auto st = decode_W_softdouble(layer.weights, &W_xy); !st)
    return st;
  std::vector<float> bias_xy;
  if (auto st = decode_bias_softdouble(layer.weights, &bias_xy); !st)
    return st;

  if (!shape_ok) {
    if (auto st =
            metal::resident_mount(&layer.gpu, W_xy.data(), bias_len ? bias_xy.data() : nullptr,
                                  bias_len, batch, out, in, layer.activation);
        !st) {
      layer.gpu_valid = false;
      return st;
    }
  } else {
    // Buffers exist; FormatNone changed (SGD / reset) — reload soft-double W only.
    if (auto st = metal::resident_reload_w(&layer.gpu, W_xy.data(),
                                           bias_len ? bias_xy.data() : nullptr, bias_len);
        !st) {
      layer.gpu_valid = false;
      return st;
    }
  }
  layer.gpu_batch = batch;
  layer.gpu_valid = true;
  return Status::Ok();
}

} // namespace

Status new_layer(int in_h, int out_h, Activation act, DType dt, const float *init_or_null,
                 Layer *layer) {
  if (!layer || in_h <= 0 || out_h <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bad dims");
  }
  if (!dtype_supported_metal_forward(dt)) {
    return Status::Err(StatusCode::Unsupported,
                       std::string("dense Metal FormatNone forward unsupported for dtype ") +
                           std::string(dtype_name(dt)) +
                           " (Go baseline; Metal soft-double path not validated for this dtype)");
  }
  metal::resident_release(&layer->gpu);
  layer->gpu_valid = false;
  layer->gpu_batch = -1;
  layer->input_height = in_h;
  layer->output_height = out_h;
  layer->activation = act;
  return WeightStore::make(out_h, in_h, dt, init_or_null, &layer->weights);
}

bool dtype_supported_metal_forward(DType dt) {
  const int id = static_cast<int>(dt);
  return id >= 0 && id < kNumDTypes;
}

Status sync_gpu_from_weights(const Layer &layer, int batch) {
  layer.invalidate_gpu();
  return ensure_resident(layer, batch);
}

Status forward_f32(const Layer &layer, const TensorF32 &input, TensorF32 *pre, TensorF32 *post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "dense: nil out");
  }
  if (!dtype_supported_metal_forward(layer.weights.dtype)) {
    return Status::Err(StatusCode::Unsupported, std::string("dense Metal unsupported dtype ") +
                                                    std::string(dtype_name(layer.weights.dtype)));
  }
  if (input.shape.size() != 2) {
    return Status::Err(StatusCode::InvalidArgument, "dense: input shape need [batch,in]");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int in = layer.input_height;
  const int out = layer.output_height;
  if (batch <= 0 || input.data.size() != static_cast<std::size_t>(batch) * in ||
      static_cast<int>(input.shape[1]) != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: input width mismatch");
  }
  if (layer.weights.rows != out || layer.weights.cols != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: weight shape mismatch");
  }

  // Mount once / reload after invalidate. Valid mirror → upload x only.
  if (auto st = ensure_resident(layer, batch); !st)
    return st;

  *pre = TensorF32::zeros(batch, out);
  *post = TensorF32::zeros(batch, out);
  return metal::resident_execute(&layer.gpu, input.data.data(), pre->data.data(),
                                 post->data.data());
}

Status backward_f32(const Layer &layer, const TensorF32 &grad_out, const TensorF32 &input,
                    const TensorF32 &pre, TensorF32 *grad_in, TensorF32 *grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "dense: nil grad out");
  }
  if (input.shape.size() != 2 || grad_out.shape.size() != 2 || pre.shape.size() != 2) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bad shapes");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int in = layer.input_height;
  const int out = layer.output_height;
  if (batch <= 0 || input.data.size() != static_cast<std::size_t>(batch) * in ||
      pre.data.size() != static_cast<std::size_t>(batch) * out ||
      grad_out.data.size() != static_cast<std::size_t>(batch) * out ||
      static_cast<int>(input.shape[1]) != in || static_cast<int>(grad_out.shape[0]) != batch ||
      static_cast<int>(grad_out.shape[1]) != out || static_cast<int>(pre.shape[0]) != batch ||
      static_cast<int>(pre.shape[1]) != out) {
    return Status::Err(StatusCode::InvalidArgument, "dense: bwd shape mismatch");
  }
  if (layer.weights.rows != out || layer.weights.cols != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: weight shape mismatch");
  }

  if (auto st = ensure_resident(layer, batch); !st)
    return st;

  *grad_in = TensorF32::zeros(batch, in);
  *grad_w = TensorF32::zeros(out, in);
  return metal::resident_backward(&layer.gpu, grad_out.data.data(), input.data.data(),
                                  pre.data.data(), grad_in->data.data(), grad_w->data.data());
}

Status linear_grad_in(const Layer& layer, const TensorF32& gy, const TensorF32& input, TensorF32* gx) {
  if (!gx || input.shape.size()!=2 || input.data.empty() || gy.data.empty())
    return Status::Err(StatusCode::InvalidArgument,"dense: proxy input");
  const int batch=int(input.shape[0]), in=layer.input_height, out=layer.output_height;
  if(input.data.size()!=size_t(batch)*in)return Status::Err(StatusCode::InvalidArgument,"dense: proxy shape");
  TensorF32 projected; projected.shape={batch,out};projected.data.resize(size_t(batch)*out);
  if(auto st=metal::run("op_project_gap",{metal::read(gy.data),metal::write(projected.data)},
      {int(gy.data.size()),int(projected.data.size())},{},projected.data.size());!st)return st;
  if(auto st=ensure_resident(layer,batch);!st)return st;
  TensorF32 pre=TensorF32::zeros(batch,out),dw=TensorF32::zeros(out,in);
  *gx=TensorF32::zeros(batch,in);
  return metal::resident_backward(&layer.gpu,projected.data.data(),input.data.data(),pre.data.data(),
                                  gx->data.data(),dw.data.data(),true);
}
Status grad_w_only(const Layer& layer, const TensorF32& gy, const TensorF32& input,
                   const TensorF32& pre, TensorF32* dw) {
  if(!dw || input.shape.size()!=2 || gy.data.empty())return Status::Err(StatusCode::InvalidArgument,"dense: dw-only input");
  const int batch=int(input.shape[0]), in=layer.input_height,out=layer.output_height;
  if(batch<=0 || input.data.size()!=size_t(batch)*in)return Status::Err(StatusCode::InvalidArgument,"dense: dw-only shape");
  TensorF32 projected;projected.shape={batch,out};projected.data.resize(size_t(batch)*out);
  if(auto st=metal::run("op_project_gap",{metal::read(gy.data),metal::write(projected.data)},
      {int(gy.data.size()),int(projected.data.size())},{},projected.data.size());!st)return st;
  *dw=TensorF32::zeros(out,in);
  return metal::run("op_proxy_dw",{metal::read(projected.data),metal::read(input.data),metal::read(pre.data),metal::write(dw->data)},
      {batch,in,out,int(layer.activation),pre.data.size()>=size_t(batch)*out?1:0},{},dw->data.size());
}

Status apply_grad_sgd(Layer *layer, const TensorF32 &dW, double lr) {
  if (!layer)
    return Status::Err(StatusCode::InvalidArgument, "dense: nil SGD layer");
  if (auto st = metal::update(layer->weights, dW, lr); !st)
    return st;
  layer->invalidate_gpu();
  return Status::Ok();
}

} // namespace welvet::dense
