// Shadow consistency gradient: round the KD term before adding to float32 gy.
kernel void op_cam_shadow(OP_ARGS) {
  if (tid >= uint(c[0]))
    return;
  float2 coef =
      native_gpu::unpack_sd(ulong(uint(c[1])) | (ulong(uint(c[2])) << 32));
  float2 diff = sd_add(float2(b[tid], 0), -float2(d[tid], 0));
  float2 term = sd_div_uint(sd_mul(sd_mul(coef, float2(2, 0)), diff), c[0]);
  e[tid] = sd_to_float(sd_add(float2(a[tid], 0), float2(sd_to_float(term), 0)));
}
kernel void op_cam_gate_gap(OP_ARGS) {
  if (tid >= uint(c[1]))
    return;
  float2 sum = 0;
  for (int i = 0; i < c[0]; ++i)
    sum = sd_add(sum, float2(a[i], 0));
  b[tid] = sd_to_float(sd_mul(
      sd_div_uint(sum, c[0]),
      native_gpu::unpack_sd(ulong(uint(c[2])) | (ulong(uint(c[3])) << 32))));
}
// Snapshot all source stores before writing any of them. Mean rounds once per
// contributing cam, matching Go BlendStoresWeighted's Float32 mean accumulator.
kernel void op_cam_blend(OP_ARGS) {
  int n = c[0], nb = c[1];
  if (tid >= uint(n))
    return;
  pn64 inv = pn_inverse(nb);
  float mean = 0;
  for (int k = 0; k < nb; ++k) {
    float term = pn_to_float(pn_mul(pn_float(a[k * n + tid]), inv));
    mean = sd_to_float(sd_add(float2(mean, 0), float2(term, 0)));
  }
  device uchar *alpha = (device uchar *)b;
  for (int k = 0; k < nb; ++k) {
    float2 aa = native_gpu::unpack_sd(native_gpu::get_u64(alpha + k * 8));
    if (aa.x <= 0) {
      d[k * n + tid] = a[k * n + tid];
      continue;
    }
    if (aa.x > 1)
      aa = float2(1, 0);
    pn64 exact = pn_bits(native_gpu::get_u64(alpha + k * 8));
    if (aa.x >= 1)
      exact = pn_float(1);
    float af = pn_to_float(exact),
          oma = pn_to_float(pn_add(pn_float(1), pn_neg(exact)));
    float lhs = sd_to_float(sd_mul(float2(oma, 0), float2(a[k * n + tid], 0)));
    // Go/arm64 fuses the second product into the final add.
    d[k * n + tid] = fma(af, mean, lhs);
  }
}
kernel void op_cam_bias_blend(OP_ARGS) {
  using namespace native_gpu;
  int n = c[0], nb = c[1];
  if (tid >= uint(n))
    return;
  device uchar *src = (device uchar *)a;
  device uchar *alpha = (device uchar *)b;
  Raw out = {(device uchar *)d, n * nb * 8};
  float2 mean = 0;
  for (int k = 0; k < nb; ++k)
    mean = sd_add(mean,
                  sd_div_uint(unpack_sd(get_u64(src + (k * n + tid) * 8)), nb));
  for (int k = 0; k < nb; ++k) {
    float2 aa = unpack_sd(get_u64(alpha + k * 8));
    float2 old = unpack_sd(get_u64(src + (k * n + tid) * 8));
    float2 value = old;
    if (aa.x > 0) {
      if (aa.x > 1)
        aa = float2(1, 0);
      value = sd_add(sd_mul(sd_add(float2(1, 0), -aa), old), sd_mul(aa, mean));
    }
    put_u64(out, (k * n + tid) * 8, pack_sd(value));
  }
}
