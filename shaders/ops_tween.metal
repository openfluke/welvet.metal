// Global gap -> local output shape, matching Go parallel.projectGap.
// Use 64-bit integer indexing and float2 sums; no host numerical projection.
kernel void op_project_gap(OP_ARGS) {
  ulong sn = ulong(c[0]), dn = ulong(c[1]), i = ulong(tid);
  if (i >= dn) return;
  ulong lo = i * sn / dn;
  ulong hi = min(sn, max((i + 1) * sn / dn, lo + 1));
  float2 sum = float2(0);
  for (ulong j = lo; j < hi; ++j) sum = sd_add(sum, float2(a[j], 0));
  float width_hi = float(hi - lo);
  float width_lo = float(long(hi - lo) - long(width_hi));
  b[i] = sd_to_float(sd_div(sum, float2(width_hi, width_lo)));
}

// Round after multiplying by the promoted 1/N share, matching Go Split credit.
kernel void op_split_scale(OP_ARGS) {
  if (tid >= uint(c[0])) return;
  float n_hi = float(c[1]);
  float n_lo = float(long(c[1])-long(n_hi));
  float2 share = sd_div(float2(1,0), float2(n_hi,n_lo));
  a[tid] = sd_to_float(sd_mul(float2(a[tid],0), share));
}
