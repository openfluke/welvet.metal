kernel void op_meta_rules(OP_ARGS) {
  if (tid)
    return;
  int n = c[0], nr = c[1];
  device int *rules = (device int *)d;
  device float2 *thresholds = (device float2 *)e;
  float2 isum = float2(0), sum = float2(0), sq = float2(0);
  float mx = b[0];
  int active = 0;
  for (int i = 0; i < n; ++i) {
    isum = sd_add(isum, float2(a[i], 0));
    sum = sd_add(sum, float2(b[i], 0));
    sq = sd_add(sq, sd_mul_float(float2(b[i], 0), b[i]));
    mx = max(mx, b[i]);
    if (abs(b[i]) > 1e-6f)
      ++active;
  }
  float2 avg = sd_div_uint(sum, n), ia = sd_div_uint(isum, n);
  float2 variance = sd_add(sd_div_uint(sq, n), -sd_mul(avg, avg));
  float stddev = n > 1 ? sqrt(max(0.0f, sd_to_float(variance))) : 0;
  int effect = 0;
  for (int i = 0; i < nr; ++i) {
    int ci = i * 4, condition = rules[ci], cool = rules[ci + 2], fires = rules[ci + 3];
    if (cool > 0 && fires < cool) {
      rules[ci + 3]++;
      continue;
    }
    float th = sd_to_float(thresholds[i]);
    bool fire = false;
    if (condition == 1)
      fire = stddev > th;
    if (condition == 2)
      fire = stddev < th && stddev >= 0;
    if (condition == 3)
      fire = abs(sd_to_float(avg)) > th;
    if (condition == 4)
      fire = abs(sd_to_float(avg)) < th;
    if (condition == 5)
      fire = abs(mx) > th;
    if (condition == 6)
      fire = float(active) / n < th;
    if (condition == 7 && abs(sd_to_float(ia)) >= 1e-6f)
      fire = abs(abs(sd_to_float(sd_div(avg, ia))) - 1) > th;
    if (fire) {
      effect = rules[ci + 1];
      rules[ci + 3] = 0;
    } else
      rules[ci + 3]++;
  }
  f[0] = float(effect);
}
kernel void op_identity(OP_ARGS) {
  if (tid < uint(c[0] * c[0]))
    a[tid] = (tid / uint(c[0]) == tid % uint(c[0])) ? 1 : 0;
}
