// Serial combine/routing preserves Go accumulation and deterministic tie order.
// Branch outputs are packed branch-major; widths may differ for Concat.
kernel void op_parallel(OP_ARGS) {
  const int rows=c[0], branches=c[1], out_dim=c[2], mode=c[3], selected=c[4];
  device int* widths=(device int*)b;
  device float2* work=(device float2*)j;
  const float2 inv=sd_div_uint(float2(1,0),branches);
  const pn64 exact_beta=pn_bits(ulong(uint(c[6]))|(ulong(uint(c[7]))<<32));
  const pn64 exact_inv=pn_inverse(branches);

  // SparseK ranks by global branch norm, then by original branch index.
  if(mode==5) {
    int offset=0;
    for(int i=0;i<branches;++i) {
      float2 sum=0;
      for(int z=0;z<rows*widths[i];++z) {
        float2 value=float2(a[offset+z],0);
        sum=sd_add(sum,sd_mul(value,value));
      }
      work[i]=sd_sqrt(sum);
      offset+=rows*widths[i];
    }
    for(int i=0;i<branches;++i) {
      int rank=0;
      for(int k=0;k<branches;++k) {
        if(work[k].x>work[i].x ||
           (work[k].x==work[i].x && (work[k].y>work[i].y ||
            (work[k].y==work[i].y && k<i)))) ++rank;
      }
      j[2*branches+i]=rank;
    }
  }
  for(int row=0;row<rows;++row) {
    float2 maximum=0,gate_sum=0;
    if(mode==3) {
      maximum=float2(d[row*branches],0);
      for(int i=1;i<branches;++i)
        if(d[row*branches+i]>maximum.x)maximum=float2(d[row*branches+i],0);
      for(int i=0;i<branches;++i)
        gate_sum=sd_add(gate_sum,sd_exp(sd_add(float2(d[row*branches+i],0),-maximum)));
    }
    for(int z=0;z<out_dim;++z) {
      float out=0,best=0;
      int best_branch=0;
      pn64 mean={0,0};
      if(mode==0) {
        int offset=0,feature_offset=0;
        for(int i=0;i<branches;++i) {
          if(z>=feature_offset && z<feature_offset+widths[i]) {
            const int index=offset+row*widths[i]+z-feature_offset;
            out=a[index];
            if(c[5])g[index]=e[row*out_dim+z];
          }
          offset+=rows*widths[i];
          feature_offset+=widths[i];
        }
        f[row*out_dim+z]=out;
        continue;
      }
      if(mode==5) {
        // The forward sum follows descending score order, not branch order.
        for(int rank=0;rank<selected;++rank)
          for(int i=0;i<branches;++i)
            if(j[2*branches+i]==rank)out+=a[i*rows*out_dim+row*out_dim+z];
        out*=sd_to_float(sd_div_uint(float2(1,0),selected));
      }
      for(int i=0;i<branches;++i) {
        const float value=a[i*rows*out_dim+row*out_dim+z];
        if(mode==1||mode==2)out+=value;
        if(mode==4&&(i==0||value>best)){best=value;best_branch=i;}
        if(mode==6)mean=pn_fma(pn_float(value),exact_inv,mean);
        if(mode==3) {
          float2 weight=sd_div(sd_exp(sd_add(float2(d[row*branches+i],0),-maximum)),gate_sum);
          out+=sd_to_float(sd_mul_float(weight,value));
        }
      }
      if(mode==2)out*=sd_to_float(inv);
      if(mode==4)out=best;
      if(mode==6) {
        pn64 delta=branches==2
            ? pn_add(pn_float(a[row*out_dim+z]),pn_neg(pn_float(a[rows*out_dim+row*out_dim+z])))
            : pn_add(pn_float(a[row*out_dim+z]),pn_neg(mean));
        out=pn_to_float(pn_fma(exact_beta,delta,mean));
      }
      f[row*out_dim+z]=out;
      if(!c[5])continue;
      for(int i=0;i<branches;++i) {
        float factor=1;
        if(mode==4)factor=i==best_branch?1:0;
        if(mode==5)factor=j[2*branches+i]<selected
            ? sd_to_float(sd_div_uint(float2(1,0),selected)):0;
        if(mode==6) {
          pn64 coefficient;
          if(branches==2)coefficient=pn_add(exact_inv,i==0?exact_beta:pn_neg(exact_beta));
          else {
            coefficient=pn_mul(pn_add(pn_float(1),pn_neg(exact_beta)),exact_inv);
            if(i==0)coefficient=pn_add(exact_beta,coefficient);
          }
          factor=pn_to_float(coefficient);
        }
        const float gy=e[row*out_dim+z];
        float routed;
        if(mode==3) {
          float2 weight=sd_div(sd_exp(sd_add(float2(d[row*branches+i],0),-maximum)),gate_sum);
          routed=sd_to_float(sd_mul_float(weight,gy));
        } else if(mode==2)routed=pn_to_float(pn_mul(exact_inv,pn_float(gy)));
        else routed=gy*factor;
        g[i*rows*out_dim+row*out_dim+z]=routed;
      }
    }
    if(c[5]&&mode==3) {
      float2 dot=0;
      for(int i=0;i<branches;++i) {
        float2 value=0;
        for(int z=0;z<out_dim;++z)
          value=sd_add(value,sd_mul(float2(e[row*out_dim+z],0),
                                  float2(a[i*rows*out_dim+row*out_dim+z],0)));
        work[i]=value;
        float2 weight=sd_div(sd_exp(sd_add(float2(d[row*branches+i],0),-maximum)),gate_sum);
        dot=sd_add(dot,sd_mul(weight,value));
      }
      for(int i=0;i<branches;++i) {
        float2 weight=sd_div(sd_exp(sd_add(float2(d[row*branches+i],0),-maximum)),gate_sum);
        h[row*branches+i]=sd_to_float(sd_mul(weight,sd_add(work[i],-dot)));
      }
    }
  }
}

// Go accumulates already-rounded branch dX in float64 and casts only once.
kernel void op_parallel_sum(OP_ARGS) {
  if(tid>=uint(c[0]))return;
  pn64 sum={0,0};
  if(tid<uint(c[2]))
    for(int i=0;i<c[1];++i)sum=pn_add(sum,pn_float(a[i*c[0]+tid]));
  b[tid]=pn_to_float(sum);
}
