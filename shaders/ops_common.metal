// Generic typed-buffer ABI for numerical kernels. All math runs on the GPU.
#define OP_ARGS                                                                                    \
  device float *a [[buffer(0)]], device float *b [[buffer(1)]], device float *d [[buffer(2)]],     \
      device float *e [[buffer(3)]], device float *f [[buffer(4)]], device float *g [[buffer(5)]], \
      device float *h [[buffer(6)]], device float *j [[buffer(7)]], constant int *c [[buffer(8)]], \
      constant float *r [[buffer(9)]], uint tid [[thread_position_in_grid]]
inline float2 sd_div(float2 x, float2 y) {
  float q = x.x / y.x;
  return sd_add(float2(q, 0), float2(sd_to_float(sd_add(x, -sd_mul_float(y, q))) / y.x, 0));
}
inline float2 sd_sqrt(float2 x) {
  float q = sqrt(x.x);
  if (q == 0)
    return float2(0);
  return sd_add(float2(q, 0), sd_div(sd_add(x, -sd_mul_float(float2(q, 0), q)), float2(2 * q, 0)));
}
inline float act_value(float v, int act);
inline float act_prime(float v, int act);
kernel void op_add(OP_ARGS) {
  if (tid < uint(c[0]))
    d[tid] = a[tid] + b[tid];
}
kernel void op_scale(OP_ARGS) {
  if (tid < uint(c[0]))
    b[tid] = a[tid] * r[0];
}
kernel void op_swiglu(OP_ARGS) {
  if (tid >= uint(c[0]))
    return;
  float s = act_value(a[tid], 1);
  if (c[1] == 0)
    d[tid] = s * b[tid];
  else {
    e[tid] = d[tid] * s;
    f[tid] =
        sd_to_float(sd_mul_float(sd_mul_float(float2(d[tid], 0), b[tid]), act_prime(a[tid], 1)));
  }
}
kernel void op_update_xy(OP_ARGS) {
  if (tid >= uint(c[0]))
    return;
  device float2 *w = (device float2 *)a;
  device float2 *grad = (device float2 *)b;
  device float2 *out = (device float2 *)d;
  float2 step = sd_mul(grad[tid], float2(r[0], r[1]));
  out[tid] = sd_add(w[tid], -step);
}
// Float2 transcendental math: range reduction and compensated series keep f32
// rounding independent of the GPU's native transcendental approximation.
inline float2 sd_exp(float2 x) {
  if (x.x > 88.0f)
    return float2(INFINITY, 0);
  if (x.x < -100.0f)
    return float2(0);
  int k = int(round(x.x * 1.4426950408889634f));
  float2 ln2 = float2(0.6931471824645996f, -1.904654323148236e-9f);
  float2 z = sd_add(x, -sd_mul_float(ln2, float(k))), term = float2(1, 0), sum = term;
  for (int i = 1; i <= 18; ++i) {
    term = sd_div_uint(sd_mul(term, z), i);
    sum = sd_add(sum, term);
  }
  return float2(ldexp(sum.x, k), ldexp(sum.y, k));
}
inline float2 sd_sigmoid(float2 x) {
  return sd_div(float2(1, 0), sd_add(float2(1, 0), sd_exp(-x)));
}
inline float2 sd_tanh(float2 x) {
  if (x.x > 18)
    return float2(1, 0);
  if (x.x < -18)
    return float2(-1, 0);
  float2 ex = sd_exp(sd_mul_float(x, 2));
  return sd_div(sd_add(ex, float2(-1, 0)), sd_add(ex, float2(1, 0)));
}
inline float2 sd_log(float2 x) {
  if (x.x <= 0)
    return float2(log(x.x), 0);
  float q = log(x.x);
  float2 ex = sd_exp(float2(q, 0));
  return sd_add(float2(q, 0), sd_div(sd_add(x, -ex), ex));
}

inline float act_value(float v, int act) {
  float2 x = float2(v, 0);
  if (act == 1)
    return sd_to_float(sd_mul(x, sd_sigmoid(x)));
  if (act == 3)
    return sd_to_float(sd_tanh(x));
  if (act == 4)
    return sd_to_float(sd_sigmoid(x));
  if (act == 2) {
    float2 cubic =
        sd_mul(sd_mul(sd_mul(x, x), x), float2(0.044714998453855515f, 1.5461444832776117e-09f));
    float2 u = sd_mul(sd_add(x, cubic), float2(0.7978845834732056f, -2.2670340160679814e-08f));
    return sd_to_float(sd_mul(sd_mul_float(x, 0.5f), sd_add(float2(1, 0), sd_tanh(u))));
  }
  if (act == 0)
    return max(v, 0.0f);
  if (act == 6)
    return v < 0 ? 0 : v * v;
  if (act == 5)
    return v < 0 ? sd_to_float(sd_mul(x, float2(0.009999999776482582f, 2.2351741811588166e-10f)))
                 : v;
  return v;
}
inline float act_prime(float v, int act) {
  float2 x = float2(v, 0), one = float2(1, 0);
  if (act == 1) {
    float2 z = sd_sigmoid(x);
    return sd_to_float(sd_mul(z, sd_add(one, sd_mul(x, sd_add(one, -z)))));
  }
  if (act == 2) {
    const float2 k = float2(0.7978845834732056f, -2.2670340160679814e-08f);
    const float2 c = float2(0.044714998453855515f, 1.5461444832776117e-09f);
    float2 xx = sd_mul(x, x);
    float2 u = sd_mul(k, sd_add(x, sd_mul(c, sd_mul(xx, x))));
    float2 t = sd_tanh(u);
    float2 du = sd_mul(k, sd_add(one, sd_mul_float(sd_mul(c, xx), 3)));
    return sd_to_float(
        sd_add(sd_mul_float(sd_add(one, t), 0.5f),
               sd_mul(sd_mul(sd_mul_float(x, 0.5f), sd_add(one, -sd_mul(t, t))), du)));
  }
  if (act == 3) {
    float2 t = sd_tanh(x);
    return sd_to_float(sd_add(one, -sd_mul(t, t)));
  }
  if (act == 4) {
    float2 z = sd_sigmoid(x);
    return sd_to_float(sd_mul(z, sd_add(one, -z)));
  }
  if (act == 0)
    return v > 0 ? 1 : 0;
  if (act == 6)
    return v > 0 ? 2 * v : 0;
  if (act == 5)
    return v > 0 ? 1 : 0.01f;
  return 1;
}
inline float2 sd_sin(float2 x) {
  float2 pi2 = float2(6.2831854820251465f, -1.7484555314695172e-7f);
  float k = round(x.x / 6.283185307179586f);
  x = sd_add(x, -sd_mul_float(pi2, k));
  float2 term = x, sum = x, xx = sd_mul(x, x);
  for (int n = 1; n <= 15; ++n) {
    term = -sd_div_uint(sd_mul(term, xx), (2 * n) * (2 * n + 1));
    sum = sd_add(sum, term);
  }
  return sum;
}
inline float2 sd_cos(float2 x) {
  return sd_sin(sd_add(x, float2(1.5707963705062866f, -4.371138828673793e-8f)));
}
kernel void op_add3(OP_ARGS) {
  if (tid < uint(c[0]))
    e[tid] = sd_to_float(sd_add(sd_add(float2(a[tid], 0), float2(b[tid], 0)), float2(d[tid], 0)));
}
