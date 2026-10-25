kernel void op_mamba_forward(OP_ARGS) {
  int B = c[0], T = c[1], N = c[2];
  if (tid >= uint(B * N))
    return;
  int batch = tid / N, i = tid % N;
  device float2 *tape = (device float2 *)f;
  float2 state = float2(0), ea = sd_exp(float2(b[i], 0));
  for (int t = 0; t < T; ++t) {
    int row = (batch * T + t) * 2 * N;
    float2 x = float2(a[row + i], 0), zd = float2(a[row + N + i], 0);
    float2 dt = zd.x > 20 ? zd : sd_log(sd_add(float2(1, 0), sd_exp(zd)));
    float2 decay = sd_exp(-sd_mul(dt, ea));
    tape[((batch * T + t) * N + i) * 3] = state;
    tape[((batch * T + t) * N + i) * 3 + 1] = dt;
    tape[((batch * T + t) * N + i) * 3 + 2] = decay;
    state = sd_add(sd_mul(decay, state), sd_mul(dt, x));
    e[(batch * T + t) * N + i] = sd_to_float(sd_add(state, sd_mul_float(x, d[i])));
  }
}
kernel void op_mamba_backward(OP_ARGS) {
  int B = c[0], T = c[1], N = c[2];
  if (tid >= uint(N))
    return;
  int i = tid;
  device float2 *tape = (device float2 *)f;
  float2 dA = float2(0), dD = float2(0), ea = sd_exp(float2(b[i], 0));
  for (int batch = 0; batch < B; ++batch) {
    float2 gh = float2(0);
    for (int t = T - 1; t >= 0; --t) {
      int row = (batch * T + t) * 2 * N, ti = ((batch * T + t) * N + i) * 3;
      float2 hp = tape[ti], dt = tape[ti + 1], decay = tape[ti + 2], x = float2(a[row + i], 0),
             gy = float2(g[(batch * T + t) * N + i], 0);
      dD = sd_add(dD, sd_mul(gy, x));
      float2 hn = sd_add(gy, gh), ga = sd_mul(hn, hp), gdt = sd_mul(hn, x);
      float2 gx = sd_add(sd_mul(hn, dt), sd_mul_float(gy, d[i]));
      gh = sd_mul(hn, decay);
      dA = sd_add(dA, -sd_mul(sd_mul(sd_mul(ga, decay), dt), ea));
      h[row + i] = sd_to_float(gx);
      h[row + N + i] = sd_to_float(sd_mul(sd_add(gdt, -sd_mul(sd_mul(ga, decay), ea)),
                                          sd_sigmoid(float2(a[row + N + i], 0))));
    }
  }
  h[B * T * 2 * N + i] = sd_to_float(dA);
  h[B * T * 2 * N + N + i] = sd_to_float(dD);
}
