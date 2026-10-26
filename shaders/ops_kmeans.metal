// Soft assignments use float2 reductions. Centers and activation derivatives
// follow Go's Float32 materialization points before promoted arithmetic.
kernel void op_kmeans_assign(OP_ARGS) {
  int B=c[0], D=c[1], K=c[2], O=c[3], mode=c[4], act=c[5];
  int row=int(tid);
  if(row>=B)return;
  device float2* probs=(device float2*)d;
  float2 temperature=float2(r[0],r[1]);
  float2 denominator=sd_mul_float(sd_mul(temperature,temperature),2);
  float2 maximum=0;
  for(int k=0;k<K;++k) {
    float2 sq=0;
    for(int z=0;z<D;++z) {
      float2 diff=sd_add(float2(a[row*D+z],0),float2(-b[k*D+z],0));
      sq=sd_add(sq,sd_mul(diff,diff));
    }
    float2 logit=-sd_div(sq,denominator);
    probs[row*K+k]=logit;
    if(k==0||logit.x>maximum.x||(logit.x==maximum.x&&logit.y>maximum.y))maximum=logit;
  }
  float2 sum=0;
  for(int k=0;k<K;++k) {
    float2 value=sd_exp(sd_add(probs[row*K+k],-maximum));
    probs[row*K+k]=value;
    sum=sd_add(sum,value);
  }
  for(int k=0;k<K;++k)probs[row*K+k]=sd_div(probs[row*K+k],sum);
  for(int z=0;z<O;++z) {
    float2 value=0;
    if(mode==0)value=probs[row*K+z];
    else for(int k=0;k<K;++k)value=sd_add(value,sd_mul_float(probs[row*K+k],b[k*D+z]));
    e[row*O+z]=sd_to_float(value);
    f[row*O+z]=act_value(e[row*O+z],act);
  }
}

kernel void op_kmeans_backward(OP_ARGS) {
  int B=c[0], D=c[1], K=c[2], O=c[3], mode=c[4], act=c[5];
  device float2* probs=(device float2*)f;
  device float2* wx=(device float2*)g;
  device float2* xx=wx+K*D;
  device float2* logits=xx+B*D;
  device float2* gy=logits+K;
  float2 temperature=float2(r[0],r[1]);
  float2 temp_sq=sd_mul(temperature,temperature);
  for(int i=0;i<K*D;++i)wx[i]=0;
  for(int i=0;i<B*D;++i)xx[i]=0;
  for(int row=0;row<B;++row) {
    for(int z=0;z<O;++z)
      gy[z]=sd_mul(float2(d[row*O+z],0),float2(act_prime(e[row*O+z],act),0));
    float2 dot=0;
    for(int k=0;k<K;++k) {
      float2 v=0;
      if(mode==0)v=gy[k];
      else for(int z=0;z<D;++z)v=sd_add(v,sd_mul_float(gy[z],b[k*D+z]));
      logits[k]=v;
      dot=sd_add(dot,sd_mul(probs[row*K+k],v));
    }
    for(int k=0;k<K;++k) {
      float2 prob=probs[row*K+k];
      float2 gl=sd_mul(prob,sd_add(logits[k],-dot));
      for(int z=0;z<D;++z) {
        if(mode==1)wx[k*D+z]=sd_add(wx[k*D+z],sd_mul(prob,gy[z]));
        float2 diff=sd_add(float2(a[row*D+z],0),float2(-b[k*D+z],0));
        float2 value=sd_div(sd_mul(gl,diff),temp_sq);
        wx[k*D+z]=sd_add(wx[k*D+z],value);
        xx[row*D+z]=sd_add(xx[row*D+z],-value);
      }
    }
  }
  for(int i=0;i<B*D;++i)h[i]=sd_to_float(xx[i]);
  for(int i=0;i<K*D;++i)j[i]=sd_to_float(wx[i]);
}
