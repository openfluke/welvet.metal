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

// Go dense.GradWOnly keeps gy*act' promoted until its F64 SAXPY accumulator.
kernel void op_proxy_dw(OP_ARGS) {
  const int B=c[0],I=c[1],O=c[2],o=int(tid)/I,i=int(tid)%I;
  if(o>=O)return;
  pn64 sum={0,0};
  for(int row=0;row<B;++row){
    pn64 g=pn_float(a[row*O+o]);
    if(c[4])g=pn_mul(g,pn_float(act_prime(d[row*O+o],c[3])));
    sum=pn_fma(g,pn_float(b[row*I+i]),sum);
  }
  e[tid]=pn_to_float(sum);
}
// Software-promoted RMS norm stays on the GPU; host only reads a control flag.
kernel void op_proxy_norm(OP_ARGS) {
  float2 sum=0;
  for(int i=0;i<c[0];++i)sum=sd_add(sum,sd_mul(float2(a[i],0),float2(a[i],0)));
  float2 norm=sd_sqrt(sd_div_uint(sum,c[0]));
  b[0]=norm.x;b[1]=norm.y;b[2]=sd_to_float(norm)>1e-12f?1.0f:0.0f;
}
kernel void op_proxy_rescale(OP_ARGS) {
  if(tid>=uint(c[0]))return;
  float2 ratio=sd_div(float2(b[0],b[1]),float2(d[0],d[1]));
  e[tid]=sd_to_float(sd_mul(float2(a[tid],0),ratio));
}
