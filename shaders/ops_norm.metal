// a=x b=gamma(float2), d=beta(float2), e=pre f=post; backwards g=dy h=dx j=packed gradients.
kernel void op_norm_forward(OP_ARGS) {
  int dim = c[0], nt = c[1];
  if (tid >= uint(nt))
    return;
  int base = tid * dim;
  float2 sum = float2(0), sq = float2(0);
  for (int i = 0; i < dim; ++i) {
    float v = a[base + i];
    sum = sd_add(sum, float2(v, 0));
    sq = sd_add(sq, sd_mul_float(float2(v, 0), v));
  }
  float2 mean = c[2] ? sd_div_uint(sum, dim) : float2(0);
  float2 var = sd_add(sd_div_uint(sq, dim), -sd_mul(mean, mean));
  float2 inv = sd_div(float2(1, 0), sd_sqrt(sd_add(var, float2(r[0], r[1]))));
  device float2 *gamma = (device float2 *)b;
  device float2 *beta = (device float2 *)d;
  for (int i = 0; i < dim; ++i) {
    float2 xh = sd_mul(sd_add(float2(a[base + i], 0), -mean), inv);
    e[base + i] = sd_to_float(xh);
    float2 y = sd_mul(xh, gamma[i]);
    if (c[2])
      y = sd_add(y, beta[i]);
    f[base + i] = sd_to_float(y);
  }
}
kernel void op_norm_backward(OP_ARGS) {
  int dim = c[0], nt = c[1];
  if (tid >= uint(nt))
    return;
  int base = tid * dim;
  device float2 *gamma = (device float2 *)b;
  float2 sum = float2(0), sq = float2(0);
  for (int i = 0; i < dim; ++i) {
    float v = a[base + i];
    sum = sd_add(sum, float2(v, 0));
    sq = sd_add(sq, sd_mul_float(float2(v, 0), v));
  }
  float2 mean = c[2] ? sd_div_uint(sum, dim) : float2(0),
         var = sd_add(sd_div_uint(sq, dim), -sd_mul(mean, mean));
  float2 inv = sd_div(float2(1, 0), sd_sqrt(sd_add(var, float2(r[0], r[1]))));
  float2 s = float2(0), sx = float2(0);
  for (int i = 0; i < dim; ++i) {
    float2 u = sd_mul_float(gamma[i], g[base + i]);
    s = sd_add(s, u);
    sx = sd_add(sx, sd_mul_float(u, e[base + i]));
  }
  for (int i = 0; i < dim; ++i) {
    float2 u = sd_mul_float(gamma[i], g[base + i]);
    float2 v = sd_add(u, -sd_mul_float(sd_div_uint(sx, dim), e[base + i]));
    if (c[2])
      v = sd_add(v, -sd_div_uint(s, dim));
    h[base + i] = sd_to_float(sd_mul(inv, v));
  }
}
kernel void op_norm_param_grad(OP_ARGS) {
  int dim = c[0], nt = c[1];
  if (tid >= uint(dim))
    return;
  float2 dg = float2(0), db = float2(0);
  for (int t = 0; t < nt; ++t) {
    float dy = g[t * dim + tid];
    dg = sd_add(dg, sd_mul_float(float2(dy, 0), e[t * dim + tid]));
    db = sd_add(db, float2(dy, 0));
  }
  j[tid] = sd_to_float(dg);
  if (c[2])
    j[dim + tid] = sd_to_float(db);
}
