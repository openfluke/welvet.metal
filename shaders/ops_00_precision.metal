// Small integer-backed binary64 ALU for cancellation-sensitive Parallel
// Disagree arithmetic. Inputs are finite FP32 branch outputs and finite beta.
// Keeps 53 significant bits and round-to-nearest-even after each operation.
struct pn64 { long sig; int exponent; };
inline pn64 pn_round(ulong magnitude, int exponent, bool negative) {
  if(!magnitude)return {0,0};
  while(magnitude<(1ul<<55)){magnitude<<=1;--exponent;}
  while(magnitude>=(1ul<<56)){magnitude=(magnitude>>1)|(magnitude&1);++exponent;}
  ulong mant=magnitude>>3,rem=magnitude&7;
  if(rem>4||(rem==4&&(mant&1)))++mant;
  if(mant==(1ul<<53)){mant>>=1;++exponent;}
  return {negative?-long(mant):long(mant),exponent+3};
}
inline pn64 pn_bits(ulong bits) {
  bool negative=bits>>63;int exponent=int((bits>>52)&2047);
  ulong mant=bits&0xffffffffffffful;
  if(exponent)mant|=1ul<<52;
  return {negative?-long(mant):long(mant),(exponent?exponent-1023:1-1023)-52};
}
inline pn64 pn_float(float value) {
  uint bits=as_type<uint>(value),e=(bits>>23)&255,m=bits&0x7fffff;
  if(!e)return {0,0}; // FP32 subnormals follow the documented GPU limit.
  ulong mant=ulong(m|0x800000)<<29;
  return {bits>>31?-long(mant):long(mant),int(e)-127-52};
}
inline pn64 pn_neg(pn64 x){x.sig=-x.sig;return x;}
inline pn64 pn_add(pn64 x,pn64 y) {
  if(!x.sig)return y;if(!y.sig)return x;
  if(x.exponent<y.exponent){pn64 tmp=x;x=y;y=tmp;}
  int shift=x.exponent-y.exponent;
  ulong ax=ulong(abs(x.sig))<<3,ay=ulong(abs(y.sig))<<3;
  if(shift>=64)ay=ay?1:0;
  else if(shift>0)ay=(ay>>shift)|((ay&((1ul<<shift)-1))?1ul:0ul);
  long sum=(x.sig<0?-long(ax):long(ax))+(y.sig<0?-long(ay):long(ay));
  return pn_round(ulong(abs(sum)),x.exponent-3,sum<0);
}
inline pn64 pn_mul(pn64 x,pn64 y) {
  if(!x.sig||!y.sig)return {0,0};
  ulong a=ulong(abs(x.sig)),b=ulong(abs(y.sig));
  ulong a0=a&0xfffffffful,a1=a>>32,b0=b&0xfffffffful,b1=b>>32;
  ulong lo=a0*b0,hi=a1*b1,p=a0*b1,q=a1*b0;
  ulong old=lo;lo+=p<<32;hi+=(p>>32)+(lo<old?1ul:0ul);
  old=lo;lo+=q<<32;hi+=(q>>32)+(lo<old?1ul:0ul);
  int shift=(hi&(1ul<<41))?53:52;
  ulong mant=(hi<<(64-shift))|(lo>>shift);
  ulong rem=lo&((1ul<<shift)-1),halfway=1ul<<(shift-1);
  if(rem>halfway||(rem==halfway&&(mant&1)))++mant;
  int exponent=x.exponent+y.exponent+shift;
  if(mant==(1ul<<53)){mant>>=1;++exponent;}
  return {(x.sig<0)!=(y.sig<0)?-long(mant):long(mant),exponent};
}
inline pn64 pn_inverse(uint divisor) {
  ulong mant=(1ul<<52)/divisor,rem=(1ul<<52)%divisor;
  int exponent=-52;
  while(mant<(1ul<<52)){
    mant<<=1;rem<<=1;--exponent;
    if(rem>=divisor){++mant;rem-=divisor;}
  }
  if(rem*2>divisor||(rem*2==divisor&&(mant&1)))++mant;
  if(mant==(1ul<<53)){mant>>=1;++exponent;}
  return {long(mant),exponent};
}
inline float pn_to_float(pn64 x) {
  if(!x.sig)return 0;
  ulong magnitude=ulong(abs(x.sig)),mant=magnitude>>29,rem=magnitude&((1ul<<29)-1);
  if(rem>(1ul<<28)||(rem==(1ul<<28)&&(mant&1)))++mant;
  float value=ldexp(float(mant),x.exponent+29);
  return x.sig<0?-value:value;
}

struct pn128 { ulong lo; ulong hi; };
inline pn128 pn_wide_product(ulong a,ulong b) {
  ulong a0=a&0xfffffffful,a1=a>>32,b0=b&0xfffffffful,b1=b>>32;
  ulong lo=a0*b0,hi=a1*b1,p=a0*b1,q=a1*b0;
  ulong old=lo;lo+=p<<32;hi+=(p>>32)+(lo<old?1ul:0ul);
  old=lo;lo+=q<<32;hi+=(q>>32)+(lo<old?1ul:0ul);
  return {lo,hi};
}
inline bool pn_low_bits(pn128 x,int count) {
  if(count<=0)return false;
  if(count>=128)return x.lo||x.hi;
  if(count>=64)return x.lo||(count>64&&(x.hi&((1ul<<(count-64))-1)));
  return x.lo&((1ul<<count)-1);
}
inline pn128 pn_shift(pn128 x,int shift) {
  if(shift==0)return x;
  if(shift>0) {
    if(shift>=128)return {0,0};
    if(shift>=64)return {0,x.lo<<(shift-64)};
    return {x.lo<<shift,(x.hi<<shift)|(x.lo>>(64-shift))};
  }
  int right=-shift;
  if(right>=128)return {ulong(x.lo||x.hi),0};
  pn128 out;
  if(right>=64)out={x.hi>>(right-64),0};
  else out={(x.lo>>right)|(x.hi<<(64-right)),x.hi>>right};
  out.lo|=pn_low_bits(x,right)?1ul:0ul;
  return out;
}
inline int pn_top(pn128 x) {
  ulong v=x.hi?x.hi:x.lo;int bit=x.hi?64:0;
  while(v>1){v>>=1;++bit;}
  return bit;
}
// Go on arm64 contracts multiply/add in these expressions. Keep the exact
// product through the addition, then round once to binary64.
inline pn64 pn_fma(pn64 x,pn64 y,pn64 z) {
  if(!x.sig||!y.sig)return z;
  pn128 product=pn_wide_product(ulong(abs(x.sig)),ulong(abs(y.sig)));
  int pe=x.exponent+y.exponent;
  if(!z.sig)return pn_mul(x,y);
  int top=max(pe+pn_top(product),z.exponent+52),exponent=top-120;
  pn128 a=pn_shift(product,pe-exponent);
  pn128 b=pn_shift({ulong(abs(z.sig)),0},z.exponent-exponent);
  bool negative=(x.sig<0)!=(y.sig<0),z_negative=z.sig<0;
  pn128 sum;
  if(negative==z_negative) {
    sum.lo=a.lo+b.lo;sum.hi=a.hi+b.hi+(sum.lo<a.lo?1ul:0ul);
  } else {
    if(a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo)){pn128 swap=a;a=b;b=swap;negative=z_negative;}
    sum.lo=a.lo-b.lo;sum.hi=a.hi-b.hi-(a.lo<b.lo?1ul:0ul);
  }
  if(!sum.lo&&!sum.hi)return {0,0};
  int shift=pn_top(sum)-52;
  ulong mant;
  if(shift<=0)mant=sum.lo<<(-shift);
  else {
    // Extract without jamming: guard/sticky decide ties separately.
    mant=shift>=64?sum.hi>>(shift-64):(sum.lo>>shift)|(sum.hi<<(64-shift));
    bool guard=shift>64?((sum.hi>>(shift-65))&1):((sum.lo>>(shift-1))&1);
    bool sticky=pn_low_bits(sum,shift-1);
    if(guard&&(sticky||(mant&1)))++mant;
  }
  exponent+=shift;
  if(mant==(1ul<<53)){mant>>=1;++exponent;}
  return {negative?-long(mant):long(mant),exponent};
}
