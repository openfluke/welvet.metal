kernel void op_embedding_forward(OP_ARGS) {
  if (tid >= uint(c[0] * c[1]))
    return;
  int tok = tid / c[1], col = tid % c[1];
  int id = abs(int(a[tok * c[3]])) % c[2];
  device float2 *w = (device float2 *)b;
  d[tid] = act_value(sd_to_float(w[id * c[1] + col]), c[4]);
}
kernel void op_embedding_backward(OP_ARGS) {
  if (tid >= uint(c[1] * c[2]))
    return;
  int row = tid / c[1], col = tid % c[1];
  float sum = 0;
  for (int t = 0; t < c[0]; ++t) {
    int id = abs(int(a[t * c[3]])) % c[2];
    if (id == row)
      sum += b[t * c[1] + col];
  }
  d[tid] = sum;
}
