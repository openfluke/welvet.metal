// Fused serial schedule, dynamic scratch. No host activation/BPTT arithmetic.
inline float2 rec_linear(device float2 *w, device float *x, int rows, int cols, int row) {
  float2 s = float2(0);
  for (int i = 0; i < cols; ++i)
    s = sd_add(s, sd_mul_float(w[row * cols + i], x[i]));
  return float2(sd_to_float(s), 0);
}
kernel void op_recurrent_forward(OP_ARGS) {
  if (tid)
    return;
  int B = c[0], T = c[1], I = c[2], H = c[3], G = c[4], gn = H * I + H * H + H;
  device float2 *w = (device float2 *)b;
  device float2 *cell = (device float2 *)g;
  device float *hp = f;
  device float2 *sums = (device float2 *)h;
  for (int t = 0; t < T; ++t) {
    for (int batch = 0; batch < B; ++batch) {
      for (int gate = 0; gate < G; ++gate)
        for (int o = 0; o < H; ++o) {
          device float2 *wg = w + gate * gn;
          float2 ih = rec_linear(wg, a + (batch * T + t) * I, H, I, o);
          ih = float2(sd_to_float(sd_add(ih, wg[H * I + H * H + o])), 0);
          float2 hh = rec_linear(wg + H * I, hp + batch * H, H, H, o);
          sums[(batch * G + gate) * H + o] = sd_add(ih, hh);
        }
      for (int o = 0; o < H; ++o) {
        int index = batch * H + o, pi = (batch * T + t) * G * H;
        float2 x = sums[(batch * G) * H + o];
        if (G == 1) {
          d[(batch * T + t) * H + o] = sd_to_float(x);
          float y = sd_to_float(sd_tanh(x));
          e[(batch * T + t) * H + o] = y;
          hp[index] = y;
        } else {
          float2 is = sums[(batch * G) * H + o], fs = sums[(batch * G + 1) * H + o],
                 gs = sums[(batch * G + 2) * H + o], os = sums[(batch * G + 3) * H + o];
          float2 cc =
              sd_add(sd_mul(sd_sigmoid(fs), cell[index]), sd_mul(sd_sigmoid(is), sd_tanh(gs)));
          float y = sd_to_float(sd_mul(sd_sigmoid(os), sd_tanh(cc)));
          int pp = (batch * T + t) * 5 * H;
          d[pp + o] = sd_to_float(is);
          d[pp + H + o] = sd_to_float(fs);
          d[pp + 2 * H + o] = sd_to_float(gs);
          d[pp + 3 * H + o] = sd_to_float(os);
          d[pp + 4 * H + o] = sd_to_float(cc);
          e[(batch * T + t) * H + o] = y;
          hp[index] = y;
          cell[index] = cc;
        }
      }
    }
  }
}
kernel void op_recurrent_backward(OP_ARGS) {
  if (tid)
    return;
  int B = c[0], T = c[1], I = c[2], H = c[3], G = c[4], gn = H * I + H * H + H, nw = G * gn;
  device float2 *w = (device float2 *)b;
  device float2 *work = (device float2 *)j;
  device float2 *pack = work;
  device float2 *gh = pack + nw;
  device float2 *gc = gh + B * H;
  device float2 *ngh = gc + B * H;
  device float2 *gin = ngh + B * H;
  device float *delta = (device float *)(gin + B * T * I);
  device float *hp = delta + B * G * H;
  for (int t = T - 1; t >= 0; --t) {
    for (int batch = 0; batch < B; ++batch)
      for (int o = 0; o < H; ++o) {
        int i = batch * H + o, oi = (batch * T + t) * H + o;
        float2 dh = sd_add(gh[i], float2(d[oi], 0));
        ngh[i] = float2(0);
        if (G == 1) {
          float2 hv = sd_tanh(float2(e[oi], 0));
          float2 dp = sd_mul(dh, sd_add(float2(1, 0), -sd_mul(hv, hv)));
          delta[batch * H + o] = sd_to_float(dp);
          pack[H * I + H * H + o] = sd_add(pack[H * I + H * H + o], dp);
          hp[i] = t ? sd_to_float(sd_tanh(float2(e[(batch * T + t - 1) * H + o], 0))) : 0;
        } else {
          int pp = (batch * T + t) * 5 * H;
          float2 ig = sd_sigmoid(float2(e[pp + o], 0)), fg = sd_sigmoid(float2(e[pp + H + o], 0)),
                 gg = sd_tanh(float2(e[pp + 2 * H + o], 0)),
                 og = sd_sigmoid(float2(e[pp + 3 * H + o], 0)),
                 cc = sd_tanh(float2(e[pp + 4 * H + o], 0));
          float2 cp = t ? float2(e[pp - H + o], 0) : float2(0);
          float2 dc = sd_add(gc[i], sd_mul(sd_mul(dh, og), sd_add(float2(1, 0), -sd_mul(cc, cc))));
          float2 di = sd_mul(sd_mul(sd_mul(dc, gg), ig), sd_add(float2(1, 0), -ig));
          float2 df = sd_mul(sd_mul(sd_mul(dc, cp), fg), sd_add(float2(1, 0), -fg));
          float2 dg = sd_mul(sd_mul(dc, ig), sd_add(float2(1, 0), -sd_mul(gg, gg)));
          float2 do_ = sd_mul(sd_mul(sd_mul(dh, cc), og), sd_add(float2(1, 0), -og));
          delta[(batch * 4) * H + o] = sd_to_float(di);
          delta[(batch * 4 + 1) * H + o] = sd_to_float(df);
          delta[(batch * 4 + 2) * H + o] = sd_to_float(dg);
          delta[(batch * 4 + 3) * H + o] = sd_to_float(do_);
          gc[i] = sd_mul(dc, fg);
          hp[i] = t ? sd_to_float(sd_mul(sd_sigmoid(float2(e[pp - 5 * H + 3 * H + o], 0)),
                                         sd_tanh(float2(e[pp - 5 * H + 4 * H + o], 0))))
                    : 0;
        }
      }
    for (int gate = 0; gate < G; ++gate) {
      device float2 *wg = w + gate * gn;
      for (int batch = 0; batch < B; ++batch) {
        for (int i = 0; i < I; ++i) {
          float2 s = float2(0);
          for (int o = 0; o < H; ++o)
            s = sd_add(s, sd_mul_float(wg[o * I + i], delta[(batch * G + gate) * H + o]));
          gin[(batch * T + t) * I + i] =
              sd_add(gin[(batch * T + t) * I + i], float2(sd_to_float(s), 0));
        }
        for (int i = 0; i < H; ++i) {
          float2 s = float2(0);
          for (int o = 0; o < H; ++o)
            s = sd_add(s, sd_mul_float(wg[H * I + o * H + i], delta[(batch * G + gate) * H + o]));
          ngh[batch * H + i] = sd_add(ngh[batch * H + i], float2(sd_to_float(s), 0));
        }
      }
      for (int o = 0; o < H; ++o) {
        for (int i = 0; i < I; ++i) {
          float2 s = float2(0);
          for (int batch = 0; batch < B; ++batch)
            s = float2(
                sd_to_float(sd_add(s, sd_mul_float(float2(delta[(batch * G + gate) * H + o], 0),
                                                   a[(batch * T + t) * I + i]))),
                0);
          int wi = gate * gn + o * I + i;
          pack[wi] = sd_add(pack[wi], float2(sd_to_float(s), 0));
        }
        for (int i = 0; i < H; ++i) {
          float2 s = float2(0);
          for (int batch = 0; batch < B; ++batch)
            s = float2(
                sd_to_float(sd_add(s, sd_mul_float(float2(delta[(batch * G + gate) * H + o], 0),
                                                   hp[batch * H + i]))),
                0);
          int wi = gate * gn + H * I + o * H + i;
          pack[wi] = sd_add(pack[wi], float2(sd_to_float(s), 0));
        }
        if (G == 4) {
          float2 s = float2(0);
          for (int batch = 0; batch < B; ++batch)
            s = sd_add(s, float2(delta[(batch * G + gate) * H + o], 0));
          int wi = gate * gn + H * I + H * H + o;
          pack[wi] = sd_add(pack[wi], s);
        }
      }
    }
    for (int i = 0; i < B * H; ++i)
      gh[i] = ngh[i];
  }
  for (int i = 0; i < nw; ++i)
    g[i] = sd_to_float(pack[i]);
  for (int i = 0; i < B * T * I; ++i)
    f[i] = sd_to_float(gin[i]);
}
