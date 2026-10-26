#include "welvet/kmeans.hpp"

#include "welvet/metal/layer_utils.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace welvet::kmeans {
int out_dim(const Config& cfg) {
  return cfg.output_mode == OutputMode::Features ? cfg.feature_dim : cfg.num_clusters;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil config");
  if (cfg->num_clusters <= 0 || cfg->feature_dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: need positive NumClusters/FeatureDim");
  }
  if (!std::isfinite(cfg->temperature) || cfg->temperature < 0.0) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: temperature must be finite and nonnegative");
  }
  if (cfg->output_mode != OutputMode::Probabilities && cfg->output_mode != OutputMode::Features) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: unknown OutputMode");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* init_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  // dense::new_layer(in_h=D, out_h=K) → Weights Rows=K, Cols=D
  if (auto st = dense::new_layer(cfg.feature_dim, cfg.num_clusters, Activation::Linear, dt,
                                 init_or_null, &layer.centers);
      !st) {
    return Status::Err(st.code, std::string("kmeans centers: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

namespace {
Status execute(const Layer& layer, const TensorF32& input, const TensorF32* grad_out,
               const TensorF32* saved_pre, TensorF32* pre, TensorF32* post,
               TensorF32* grad_in, TensorF32* grad_w) {
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  if (input.shape.empty() || input.shape.back() != cfg.feature_dim || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: input shape");
  }
  int batch = 1;
  for (size_t i = 0; i + 1 < input.shape.size(); ++i) {
    if (input.shape[i] <= 0) {
      return Status::Err(StatusCode::InvalidArgument, "kmeans: input extent");
    }
    batch *= input.shape[i];
  }
  const int d = cfg.feature_dim, k = cfg.num_clusters, od = out_dim(cfg);
  if (input.data.size() != size_t(batch * d) ||
      (grad_out && (grad_out->data.size() != size_t(batch * od) ||
                    saved_pre->data.size() != size_t(batch * od)))) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: buffer shape");
  }
  // Go materializes native centers as Float32 before computing distances.
  std::vector<float> centers(k * d);
  if (auto st = layer.centers.weights.export_f32(centers.data()); !st) return st;
  TensorF32 p, o, dx, dw;
  p.shape = input.shape;
  p.shape.back() = od;
  o.shape = p.shape;
  p.data.resize(batch * od);
  o.data.resize(batch * od);
  dx.shape = input.shape;
  dx.data.resize(batch * d);
  dw.shape = {k, d};
  dw.data.resize(k * d);
  float temp_hi, temp_lo;
  metal::split_double(cfg.temperature == 0 ? 1 : cfg.temperature, &temp_hi, &temp_lo);
  std::vector<float> probabilities(2 * batch * k);
  auto st = metal::run("op_kmeans_assign",
      {metal::read(input.data), metal::read(centers), metal::write(probabilities),
       metal::write(p.data), metal::write(o.data)},
      {batch, d, k, od, int(cfg.output_mode), int(cfg.activation)}, {temp_hi, temp_lo}, batch);
  if (!st) return st;
  if (grad_out) {
    // Dynamic scratch has no fixed cluster-count or feature-width limit.
    std::vector<float> scratch(2 * (k * d + batch * d + k + od));
    st = metal::run("op_kmeans_backward",
        {metal::read(input.data), metal::read(centers), metal::read(grad_out->data),
         metal::read(saved_pre->data), metal::read(probabilities), metal::write(scratch),
         metal::write(dx.data), metal::write(dw.data)},
        {batch, d, k, od, int(cfg.output_mode), int(cfg.activation)}, {temp_hi, temp_lo});
    if (!st) return st;
  }
  if (pre) *pre = std::move(p);
  if (post) *post = std::move(o);
  if (grad_in) *grad_in = std::move(dx);
  if (grad_w) *grad_w = std::move(dw);
  return Status::Ok();
}
} // namespace

Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil output");
  return execute(layer, input, nullptr, nullptr, pre, post, nullptr, nullptr);
}

int grad_w_size(const Layer& layer) { return layer.cfg.num_clusters * layer.cfg.feature_dim; }

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: nil gradient");
  }
  return execute(layer, input, &grad_out, &pre, nullptr, nullptr, grad_in, grad_w);
}

Status apply_grad_sgd(Layer* layer, const TensorF32& grad_w, double lr) {
  if (!layer) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil layer");
  return dense::apply_grad_sgd(&layer->centers, grad_w, lr);
}

Status export_packed_f32(const Layer& layer, float* dst) {
  return layer.centers.weights.export_f32(dst);
}
} // namespace welvet::kmeans
