#include "welvet/dense.hpp"

namespace welvet::dense {

Status new_layer(int32_t in_dim, int32_t out_dim, Activation act, DType dt, const float* W_init,
                 Layer* layer_out) {
  if (!layer_out) return Status::Err(StatusCode::InvalidArgument, "null Layer");
  if (in_dim <= 0 || out_dim <= 0) return Status::Err(StatusCode::InvalidArgument, "bad dims");

  Layer layer;
  layer.in = in_dim;
  layer.out = out_dim;
  layer.act = act;
  layer.dtype = dt;
  auto st = weights::new_format_none(out_dim, in_dim, dt, W_init, &layer.weights);
  if (!st) return st;
  layer.bias.assign(static_cast<std::size_t>(out_dim), 0.0);
  *layer_out = std::move(layer);
  return Status::Ok();
}

Status forward_f32(const Layer&, const TensorF32&, TensorF32*, TensorF32*) {
  return Status::Gap("Dense Metal forward (upload → matvec → bias → act → download)");
}

} // namespace welvet::dense
