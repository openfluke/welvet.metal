inline bool attn_allow(int q, int k, constant int *c) {
  int mask = c[6];
  if (mask == 2)
    return true;
  if (mask == 3) {
    int w = c[9] > 0 ? c[9] : c[5];
    return c[11] ? k <= q && k >= q - w + 1 : abs(q - k) < w;
  }
  if (mask == 4 && q < c[10] && k < c[10])
    return true;
  return k <= q;
}
inline void rope_gpu(device float2 *v, int pos, int nh, int hd, float2 theta, bool reverse) {
  int half_dim = hd / 2;
  if (half_dim < 1 || theta.x <= 0)
    return;
  for (int head = 0; head < nh; ++head)
    for (int i = 0; i < half_dim; ++i) {
      float2 power = sd_exp(sd_mul_float(sd_log(theta), float(2 * i) / hd));
      float2 angle = sd_div(float2(float(pos), 0), power), cs = sd_cos(angle), sn = sd_sin(angle);
      float2 x = v[head * hd + i], y = v[head * hd + i + half_dim];
      if (reverse) {
        v[head * hd + i] = sd_add(sd_mul(x, cs), sd_mul(y, sn));
        v[head * hd + i + half_dim] = sd_add(-sd_mul(x, sn), sd_mul(y, cs));
      } else {
        v[head * hd + i] = sd_add(sd_mul(x, cs), -sd_mul(y, sn));
        v[head * hd + i + half_dim] = sd_add(sd_mul(x, sn), sd_mul(y, cs));
      }
    }
}
kernel void op_attention(OP_ARGS) {
  if (tid)
    return;
  int B = c[0], T = c[1], NH = c[2], NK = c[3], HD = c[4], MSL = c[5], QD = NH * HD, KD = NK * HD,
      back = c[12];
  device float2 *work = (device float2 *)h;
  device float2 *ck = work;
  device float2 *cv = ck + MSL * KD;
  device float2 *q = cv + MSL * KD;
  device float2 *gq = q + T * QD;
  device float2 *gk = gq + T * QD;
  device float2 *gv = gk + MSL * KD;
  device float2 *score = gv + MSL * KD;
  device float2 *dw = score + B * T;
  device float2 *ds = dw + B * T;
  float2 theta = float2(r[0], r[1]), scale = float2(r[2], r[3]);
  if (scale.x == 0)
    scale = sd_div(float2(1, 0), sd_sqrt(float2(float(HD), 0)));
  int grad_shift = 0;
  if (back) {
    float maxgrad = 0;
    for (int i = 0; i < B * T * QD; ++i)
      maxgrad = max(maxgrad, abs(e[i]));
    if (maxgrad < 1e-20f)
      grad_shift = 64;
  }
  bool rope = c[7] == 0 || c[7] == 3, alibi = c[7] == 2 || c[7] == 3;
  for (int batch = 0; batch < B; ++batch) {
    int base = batch * T, kvlen = base + T;
    if (back) {
      for (int i = 0; i < MSL * KD; ++i) {
        ck[i] = float2(0);
        cv[i] = float2(0);
      }
      for (int i = 0; i < T * QD; ++i)
        gq[i] = float2(0);
    }
    for (int t = 0; t < T; ++t) {
      int pos = base + t, tok = base + t;
      for (int i = 0; i < QD; ++i)
        q[t * QD + i] = float2(a[tok * QD + i], 0);
      for (int i = 0; i < KD; ++i) {
        ck[(pos % MSL) * KD + i] = float2(b[tok * KD + i], 0);
        cv[(pos % MSL) * KD + i] = float2(d[tok * KD + i], 0);
      }
      if (rope) {
        rope_gpu(q + t * QD, pos, NH, HD, theta, false);
        rope_gpu(ck + (pos % MSL) * KD, pos, NK, HD, theta, false);
      }
    }
    for (int head = 0; head < NH; ++head)
      for (int t = 0; t < T; ++t) {
        int pos = base + t, kh = head / (NH / NK), qo = t * QD + head * HD;
        float2 mx = float2(-1e9f, 0), sum = float2(0);
        int count = 0;
        for (int k = 0; k < kvlen; ++k) {
          if (!attn_allow(pos, k, c)) {
            score[k] = float2(0);
            continue;
          }
          float2 dot = float2(0);
          for (int z = 0; z < HD; ++z)
            dot = sd_add(dot, sd_mul(q[qo + z], ck[(k % MSL) * KD + kh * HD + z]));
          float2 sc = sd_mul(dot, scale);
          if (alibi) {
            float2 slope = sd_exp(sd_mul_float(float2(0.6931471824645996f, -1.904654323148236e-9f),
                                               -float(head + 1) * 8 / NH));
            sc = sd_add(sc, -sd_mul_float(slope, float(abs(pos - k))));
          }
          score[k] = sc;
          if (sd_to_float(sc) > sd_to_float(mx))
            mx = sc;
          ++count;
        }
        if (!count) {
          if (!back)
            for (int z = 0; z < HD; ++z)
              f[(base + t) * QD + head * HD + z] = 0;
          continue;
        }
        for (int k = 0; k < kvlen; ++k)
          if (attn_allow(pos, k, c)) {
            score[k] = c[8] == 1 ? sd_sigmoid(score[k]) : sd_exp(sd_add(score[k], -mx));
            sum = sd_add(sum, score[k]);
          }
        if (c[8] != 1)
          for (int k = 0; k < kvlen; ++k)
            if (attn_allow(pos, k, c))
              score[k] = sd_div(score[k], sum);
        if (!back) {
          for (int z = 0; z < HD; ++z) {
            float2 out = float2(0);
            for (int k = 0; k < kvlen; ++k)
              if (attn_allow(pos, k, c))
                out = sd_add(out, sd_mul(score[k], cv[(k % MSL) * KD + kh * HD + z]));
            f[(base + t) * QD + head * HD + z] = sd_to_float(out);
          }
        } else {
          float2 sumdw = float2(0);
          for (int k = 0; k < kvlen; ++k) {
            dw[k] = float2(0);
            if (!attn_allow(pos, k, c))
              continue;
            for (int z = 0; z < HD; ++z) {
              float gy = ldexp(e[(base + t) * QD + head * HD + z], grad_shift);
              int vi = (k % MSL) * KD + kh * HD + z;
              gv[vi] = sd_add(gv[vi], sd_mul_float(score[k], gy));
              dw[k] = sd_add(dw[k], sd_mul_float(cv[vi], gy));
            }
            sumdw = sd_add(sumdw, sd_mul(dw[k], score[k]));
          }
          for (int k = 0; k < kvlen; ++k)
            if (attn_allow(pos, k, c)) {
              float2 delta = c[8] == 1
                                 ? sd_mul(sd_mul(dw[k], score[k]), sd_add(float2(1, 0), -score[k]))
                                 : sd_mul(score[k], sd_add(dw[k], -sumdw));
              delta = sd_mul(delta, scale);
              for (int z = 0; z < HD; ++z) {
                int ki = (k % MSL) * KD + kh * HD + z;
                gq[qo + z] = sd_add(gq[qo + z], sd_mul(delta, ck[ki]));
                gk[ki] = sd_add(gk[ki], sd_mul(delta, q[qo + z]));
              }
            }
        }
      }
    if (back) {
      for (int t = 0; t < T; ++t) {
        int pos = base + t;
        if (rope)
          rope_gpu(gq + t * QD, pos, NH, HD, theta, true);
        for (int i = 0; i < QD; ++i)
          g[(base + t) * QD + i] = ldexp(sd_to_float(gq[t * QD + i]), -grad_shift);
      }
      if (rope)
        for (int t = 0; t < T; ++t)
          rope_gpu(gk + ((base + t) % MSL) * KD, base + t, NK, HD, theta, true);
    }
  }
  if (back)
    for (int tok = 0; tok < B * T; ++tok)
      for (int i = 0; i < KD; ++i) {
        g[B * T * QD + tok * KD + i] = ldexp(sd_to_float(gk[(tok % MSL) * KD + i]), -grad_shift);
        g[B * T * (QD + KD) + tok * KD + i] =
            ldexp(sd_to_float(gv[(tok % MSL) * KD + i]), -grad_shift);
      }
}
