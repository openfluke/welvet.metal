inline int conv_kvol(constant int *c) {
  return c[14] == 3 ? c[9] * c[9] * c[9] : c[14] == 2 ? c[9] * c[9] : c[9];
}
inline int conv_map(int id, int kd, int kh, int kw, int od, int oh, int ow, constant int *c) {
  int z, y, x;
  if (c[12]) {
    int zz = od + c[11] - kd, yy = oh + c[11] - kh, xx = ow + c[11] - kw;
    if (c[14] < 3)
      zz = 0;
    if (c[14] < 2)
      yy = 0;
    if (zz % c[10] || yy % c[10] || xx % c[10])
      return -1;
    z = zz / c[10];
    y = yy / c[10];
    x = xx / c[10];
  } else {
    z = c[14] == 3 ? od * c[10] + kd - c[11] : 0;
    y = c[14] >= 2 ? oh * c[10] + kh - c[11] : 0;
    x = ow * c[10] + kw - c[11];
  }
  if (z < 0 || z >= c[3] || y < 0 || y >= c[4] || x < 0 || x >= c[5])
    return -1;
  return (z * c[4] + y) * c[5] + x;
}
kernel void op_conv_forward(OP_ARGS) {
  int isp = c[3] * c[4] * c[5], osp = c[6] * c[7] * c[8], kv = conv_kvol(c);
  if (tid >= uint(c[0] * c[2] * osp))
    return;
  int batch = tid / (c[2] * osp), filt = (tid / osp) % c[2], o = tid % osp, ow = o % c[8],
      oh = (o / c[8]) % c[7], od = o / (c[8] * c[7]);
  device float2 *w = (device float2 *)b;
  float2 sum = float2(0);
  for (int ic = 0; ic < c[1]; ++ic)
    for (int k = 0; k < kv; ++k) {
      int kd = c[14] == 3 ? k / (c[9] * c[9]) : 0, kh = c[14] >= 2 ? (k / c[9]) % c[9] : 0,
          kw = k % c[9];
      int pos = conv_map(0, kd, kh, kw, od, oh, ow, c);
      if (pos < 0)
        continue;
      float2 ww = w[(filt * c[1] + ic) * kv + k];
      if (c[12])
        ww = float2(sd_to_float(ww), 0);
      sum = sd_add(sum, sd_mul_float(ww, a[(batch * c[1] + ic) * isp + pos]));
    }
  float2 pre = float2(sd_to_float(sum), 0);
  if (!c[12] && filt < c[15])
    pre = sd_add(pre, ((device float2 *)f)[filt]);
  d[tid] = sd_to_float(pre);
  e[tid] = act_value(d[tid], c[13]);
}
kernel void op_conv_backward_x(OP_ARGS) {
  int isp = c[3] * c[4] * c[5], osp = c[6] * c[7] * c[8], kv = conv_kvol(c);
  if (tid >= uint(c[0] * c[1] * isp))
    return;
  int batch = tid / (c[1] * isp), ic = (tid / isp) % c[1], pos = tid % isp;
  device float2 *w = (device float2 *)b;
  float2 sum = float2(0);
  for (int o = 0; o < osp; ++o)
    for (int k = 0; k < kv; ++k) {
      int kd = c[14] == 3 ? k / (c[9] * c[9]) : 0, kh = c[14] >= 2 ? (k / c[9]) % c[9] : 0,
          kw = k % c[9];
      if (conv_map(0, kd, kh, kw, o / (c[7] * c[8]), (o / c[8]) % c[7], o % c[8], c) != pos)
        continue;
      float2 col = float2(0);
      for (int filt = 0; filt < c[2]; ++filt) {
        int oi = (batch * c[2] + filt) * osp + o;
        float dp = d[oi] * act_prime(e[oi], c[13]);
        float2 ww = w[(filt * c[1] + ic) * kv + k];
        if (c[12])
          ww = float2(sd_to_float(ww), 0);
        col = sd_add(col, sd_mul_float(ww, dp));
      }
      sum = sd_add(sum, c[12] ? col : float2(sd_to_float(col), 0));
    }
  f[tid] = sd_to_float(sum);
}
kernel void op_conv_backward_w(OP_ARGS) {
  int isp = c[3] * c[4] * c[5], osp = c[6] * c[7] * c[8], kv = conv_kvol(c);
  if (tid >= uint(c[2] * c[1] * kv))
    return;
  int filt = tid / (c[1] * kv), ic = (tid / kv) % c[1], k = tid % kv;
  int kd = c[14] == 3 ? k / (c[9] * c[9]) : 0, kh = c[14] >= 2 ? (k / c[9]) % c[9] : 0,
      kw = k % c[9];
  float2 sum = float2(0);
  for (int batch = 0; batch < c[0]; ++batch)
    for (int o = 0; o < osp; ++o) {
      int pos = conv_map(0, kd, kh, kw, o / (c[7] * c[8]), (o / c[8]) % c[7], o % c[8], c);
      if (pos < 0)
        continue;
      int oi = (batch * c[2] + filt) * osp + o;
      float dp = d[oi] * act_prime(e[oi], c[13]);
      sum = sd_add(sum, sd_mul_float(float2(a[(batch * c[1] + ic) * isp + pos], 0), dp));
      if (!c[12])
        sum = float2(sd_to_float(sum), 0);
    }
  g[tid] = sd_to_float(sum);
}
