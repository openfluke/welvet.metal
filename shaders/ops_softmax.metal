// One thread per row; global scratch has no fixed dimension limit.
kernel void op_softmax_forward(OP_ARGS) {
  int rows = c[0], cols = c[1], kind = c[2], maskn = c[3];
  if (tid >= uint(rows))
    return;
  int start = tid * cols;
  device float *logits = d + start;
  device float *sorted = e + start;
  device float *result = f + start;
  device int *mask = (device int *)b;
  float2 inv = sd_div(float2(1, 0), float2(r[0], r[1]));
  uint state = uint(c[4]);
  if (!state)
    state = 0xC0FFEEu;
  if (kind == 2) {
    for (int i = 0; i < start; ++i)
      state = state * 1664525u + 1013904223u;
  }
  float mx = -INFINITY;
  for (int i = 0; i < cols; ++i) {
    float v = sd_to_float(sd_mul_float(inv, a[start + i]));
    if (kind == 3 && start + i < maskn && !mask[start + i])
      v = -1e9f;
    if (kind == 2) {
      state = state * 1664525u + 1013904223u;
      float u = max(float(state >> 8) * (1.0f / 16777216.0f), 1e-10f);
      v -= log(-log(u));
    }
    logits[i] = v;
    sorted[i] = v;
    mx = max(mx, v);
  }
  float se = 0;
  for (int i = 0; i < cols; ++i) {
    // Go evaluates exp in float64, then rounds each term to Float32.
    result[i] = sd_to_float(sd_exp(float2(logits[i] - mx, 0)));
    se += result[i];
  }
  for (int i = 0; i < cols; ++i)
    result[i] /= se;
  float alpha = r[2];
  if (alpha == 0)
    alpha = 1.5f;
  if (kind == 4 || kind == 5 && alpha > 1) {
    for (int i = 0; i < cols; ++i)
      for (int k = i + 1; k < cols; ++k)
        if (sorted[i] < sorted[k]) {
          float tmp = sorted[i];
          sorted[i] = sorted[k];
          sorted[k] = tmp;
        }
    float cum = 0;
    int count = 0;
    for (int i = 0; i < cols; ++i) {
      cum += sorted[i];
      if (sorted[i] - (cum - 1) / float(i + 1) > 0)
        count = i + 1;
      else
        break;
    }
    float tau = 0;
    if (count) {
      cum = 0;
      for (int i = 0; i < count; ++i)
        cum += sorted[i];
      tau = (cum - 1) / float(count);
    }
    float sum = 0;
    for (int i = 0; i < cols; ++i) {
      float sp = max(0.0f, logits[i] - tau);
      if (kind == 4 || alpha >= 2)
        result[i] = sp;
      else
        result[i] = (2 - alpha) * result[i] + (alpha - 1) * sp;
      sum += result[i];
    }
    if (kind == 5 && alpha < 2)
      for (int i = 0; i < cols; ++i)
        result[i] /= sum;
  }
}
kernel void op_softmax_backward(OP_ARGS) {
  int rows = c[0], cols = c[1];
  if (tid >= uint(rows))
    return;
  int start = tid * cols;
  device int *mask = (device int *)d;
  float dot = 0;
  for (int i = 0; i < cols; ++i) {
    float gy = b[start + i];
    if (c[2] == 3 && start + i < c[3] && !mask[start + i])
      gy = 0;
    dot += gy * a[start + i];
  }
  float2 inv = sd_div(float2(1, 0), float2(r[0], r[1]));
  for (int i = 0; i < cols; ++i) {
    float gy = b[start + i];
    if (c[2] == 3 && start + i < c[3] && !mask[start + i])
      gy = 0;
    float gl = a[start + i] * (gy - dot);
    e[start + i] = sd_to_float(sd_mul_float(inv, gl));
  }
}
