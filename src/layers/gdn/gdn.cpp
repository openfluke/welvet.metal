#include "welvet/gdn.hpp"
#include "welvet/metal/layer_utils.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace welvet::gdn {

int key_dim(const Config &cfg) { return cfg.num_key_heads * cfg.key_head_dim; }

int value_dim(const Config &cfg) { return cfg.num_value_heads * cfg.value_head_dim; }

int conv_dim(const Config &cfg) { return key_dim(cfg) * 2 + value_dim(cfg); }

int weight_count(const Config &cfg) {
  Config c = cfg;
  if (c.conv_kernel < 1)
    c.conv_kernel = 4;
  const int cd = conv_dim(c);
  const int vd = value_dim(c);
  const int h = c.hidden_size;
  const int nh = c.num_value_heads;
  return cd * h + vd * h + nh * h + nh * h + h * vd + cd * c.conv_kernel + nh + nh +
         c.value_head_dim;
}

Status validate(Config *cfg) {
  if (!cfg)
    return Status::Err(StatusCode::InvalidArgument, "gdn: nil config");
  if (cfg->hidden_size <= 0 || cfg->num_key_heads <= 0 || cfg->num_value_heads <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: need positive HiddenSize/heads");
  }
  if (cfg->key_head_dim <= 0 || cfg->value_head_dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "gdn: need positive head dims");
  }
  if (cfg->num_value_heads % cfg->num_key_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument,
                       "gdn: NumValueHeads must divide by NumKeyHeads");
  }
  if (cfg->conv_kernel < 1)
    cfg->conv_kernel = 4;
  if (cfg->eps <= 0)
    cfg->eps = 1e-6;
  return Status::Ok();
}

Status new_configured(const Config &cfg_in, const float *packed_or_null, Layer *out) {
  if (!out)
    return Status::Err(StatusCode::InvalidArgument, "gdn: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st)
    return st;

  const int h = cfg.hidden_size;
  const int cd = conv_dim(cfg);
  const int vd = value_dim(cfg);
  const int nh = cfg.num_value_heads;
  const int k = cfg.conv_kernel;
  const int hd_v = cfg.value_head_dim;

  const int n_qkv = cd * h;
  const int n_z = vd * h;
  const int n_b = nh * h;
  const int n_a = nh * h;
  const int n_out = h * vd;
  const int n_conv = cd * k;

  Layer layer;
  layer.cfg = cfg;
  layer.in_qkv.assign(static_cast<std::size_t>(n_qkv), 0.0f);
  layer.in_z.assign(static_cast<std::size_t>(n_z), 0.0f);
  layer.in_b.assign(static_cast<std::size_t>(n_b), 0.0f);
  layer.in_a.assign(static_cast<std::size_t>(n_a), 0.0f);
  layer.out.assign(static_cast<std::size_t>(n_out), 0.0f);
  layer.conv_weight.assign(static_cast<std::size_t>(n_conv), 0.0f);
  layer.a_log.assign(static_cast<std::size_t>(nh), -1.0f);
  layer.dt_bias.assign(static_cast<std::size_t>(nh), 0.0f);
  layer.norm_gamma.assign(static_cast<std::size_t>(hd_v), 1.0f);

  if (packed_or_null) {
    const float *p = packed_or_null;
    std::memcpy(layer.in_qkv.data(), p, static_cast<std::size_t>(n_qkv) * sizeof(float));
    p += n_qkv;
    std::memcpy(layer.in_z.data(), p, static_cast<std::size_t>(n_z) * sizeof(float));
    p += n_z;
    std::memcpy(layer.in_b.data(), p, static_cast<std::size_t>(n_b) * sizeof(float));
    p += n_b;
    std::memcpy(layer.in_a.data(), p, static_cast<std::size_t>(n_a) * sizeof(float));
    p += n_a;
    std::memcpy(layer.out.data(), p, static_cast<std::size_t>(n_out) * sizeof(float));
    p += n_out;
    std::memcpy(layer.conv_weight.data(), p, static_cast<std::size_t>(n_conv) * sizeof(float));
    p += n_conv;
    std::memcpy(layer.a_log.data(), p, static_cast<std::size_t>(nh) * sizeof(float));
    p += nh;
    std::memcpy(layer.dt_bias.data(), p, static_cast<std::size_t>(nh) * sizeof(float));
    p += nh;
    std::memcpy(layer.norm_gamma.data(), p, static_cast<std::size_t>(hd_v) * sizeof(float));
  }

  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer &layer, const TensorF32 &input, TensorF32 *pre, TensorF32 *post) {
  if (!pre || !post)
    return Status::Err(StatusCode::InvalidArgument, "gdn: nil output");
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  if (input.shape.size() != 3 || input.shape[2] != cfg.hidden_size || input.shape[0] <= 0 ||
      input.shape[1] <= 0)
    return Status::Err(StatusCode::InvalidArgument, "gdn: input shape");
  int B = int(input.shape[0]), T = int(input.shape[1]), H = cfg.hidden_size;
  if (input.data.size() != static_cast<std::size_t>(B) * T * H)
    return Status::Err(StatusCode::InvalidArgument, "gdn: input data");
  std::vector<float> packed(weight_count(cfg));
  if (auto st = export_packed_f32(layer, packed.data()); !st)
    return st;
  std::size_t units = H + conv_dim(cfg) * (cfg.conv_kernel + 8) + value_dim(cfg) * 16 +
                      cfg.num_value_heads * cfg.key_head_dim * (cfg.value_head_dim + 8) +
                      cfg.num_value_heads * 16;
  std::vector<float> scratch(units * (T + 1) * 16, 0), empty;
  *pre = input;
  std::vector<float> grads(weight_count(cfg), 0);
  if (auto st = metal::run("op_gdn",
                           {metal::read(input.data), metal::read(packed), metal::read(empty),
                            metal::write(pre->data), metal::read(empty), metal::write(grads),
                            metal::write(scratch)},
                           {H, cfg.num_key_heads, cfg.num_value_heads, cfg.key_head_dim,
                            cfg.value_head_dim, cfg.conv_kernel, B, T, 0},
                           {float(cfg.eps)});
      !st)
    return st;
  *post = *pre;
  return Status::Ok();
}

int grad_w_size(const Layer &layer) { return weight_count(layer.cfg); }

Status backward_f32(Layer &layer, const TensorF32 &grad_out, const TensorF32 &input,
                    const TensorF32 & /*pre*/, TensorF32 *grad_in, TensorF32 *grad_w) {
  if (!grad_in || !grad_w)
    return Status::Err(StatusCode::InvalidArgument, "gdn: nil gradients");
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st)
    return st;
  if (input.shape.size() != 3 || input.shape[2] != cfg.hidden_size || input.shape[0] <= 0 ||
      input.shape[1] <= 0)
    return Status::Err(StatusCode::InvalidArgument, "gdn: input shape");
  int B = int(input.shape[0]), T = int(input.shape[1]), H = cfg.hidden_size;
  if (input.data.size() != static_cast<std::size_t>(B) * T * H)
    return Status::Err(StatusCode::InvalidArgument, "gdn: input data");
  std::vector<float> packed(weight_count(cfg));
  if (auto st = export_packed_f32(layer, packed.data()); !st)
    return st;
  std::size_t units = H + conv_dim(cfg) * (cfg.conv_kernel + 8) + value_dim(cfg) * 16 +
                      cfg.num_value_heads * cfg.key_head_dim * (cfg.value_head_dim + 8) +
                      cfg.num_value_heads * 16;
  std::vector<float> scratch(units * (T + 1) * 16, 0), empty;
  if (grad_out.shape != input.shape || grad_out.data.size() != input.data.size())
    return Status::Err(StatusCode::InvalidArgument, "gdn: gradient shape");
  *grad_in = input;
  *grad_w = TensorF32::zeros(1, weight_count(cfg));
  return metal::run("op_gdn",
                    {metal::read(input.data), metal::read(packed), metal::read(grad_out.data),
                     metal::read(empty), metal::write(grad_in->data), metal::write(grad_w->data),
                     metal::write(scratch)},
                    {H, cfg.num_key_heads, cfg.num_value_heads, cfg.key_head_dim,
                     cfg.value_head_dim, cfg.conv_kernel, B, T, 1},
                    {float(cfg.eps)});
}

Status apply_grad_sgd(Layer *layer, const TensorF32 &dW, double lr) {
  if (!layer)
    return Status::Err(StatusCode::InvalidArgument, "gdn: nil layer");
  if (dW.data.size() != static_cast<std::size_t>(weight_count(layer->cfg)))
    return Status::Err(StatusCode::InvalidArgument, "gdn: gradient shape");
  std::vector<float> packed(dW.data.size());
  if (auto st = export_packed_f32(*layer, packed.data()); !st)
    return st;
  if (auto st = metal::run("op_gdn_sgd", {metal::write(packed), metal::read(dW.data)},
                           {int(packed.size())}, {float(lr)}, packed.size());
      !st)
    return st;
  Layer updated;
  if (auto st = new_configured(layer->cfg, packed.data(), &updated); !st)
    return st;
  *layer = std::move(updated);
  return Status::Ok();
}

Status export_packed_f32(const Layer &layer, float *dst) {
  if (!dst)
    return Status::Err(StatusCode::InvalidArgument, "gdn: nil export dst");
  const Config &c = layer.cfg;
  const int cd = conv_dim(c);
  const int vd = value_dim(c);
  const int nh = c.num_value_heads;
  const int h = c.hidden_size;
  const int k = c.conv_kernel < 1 ? 1 : c.conv_kernel;
  int off = 0;
  auto put = [&](const std::vector<float> &w) {
    if (!w.empty()) {
      std::memcpy(dst + off, w.data(), w.size() * sizeof(float));
      off += static_cast<int>(w.size());
    }
  };
  put(layer.in_qkv);
  put(layer.in_z);
  put(layer.in_b);
  put(layer.in_a);
  put(layer.out);
  put(layer.conv_weight);
  put(layer.a_log);
  put(layer.dt_bias);
  put(layer.norm_gamma);
  (void)cd;
  (void)vd;
  (void)nh;
  (void)h;
  (void)k;
  return Status::Ok();
}

} // namespace welvet::gdn
