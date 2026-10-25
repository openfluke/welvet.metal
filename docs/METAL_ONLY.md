# Metal execution contract

`welvet.metal` is a standalone native Apple GPU backend. `welvet.cpp` is the
separate CPU backend. Each can eventually ship to Go Welvet through its own
native artifact and C ABI. There is no dependency from one backend to the other.

## Supported computation

- Dense forward and backward: Metal soft-double GEMV, activation, derivative,
  transpose GEMV, and outer-product kernels. All 34 FormatNone dtypes can be
  decoded at the storage boundary for these operations.
- MSE loss and MSE gradient: Metal kernels. Loss reduction uses a fixed serial
  order and float2 arithmetic. It is a correctness implementation, not a tuned
  parallel reduction. Validation band is `1e-12 * max(1, abs(reference))` for the
  finite Float32 fixtures in w2a; float2 is not IEEE float64 and exact-bit loss
  parity is not promised for arbitrary data.
- NormalBP forward/backward: Dense, RMSNorm, LayerNorm, SwiGLU, Softmax,
  Embedding, MHA, CNN1/2/3, ConvT1/2/3, RNN, LSTM, Sequential, Residual,
  Mamba, Metacognition, GDN and Stack use Metal numerical kernels.
- SGD and native encoding: Metal arithmetic and native FormatNone storage updates
  across all 34 dtypes. Float32 rounds the update step before subtraction to follow
  Go's semantics; other dtypes retain their native encoding. Bias updates also run
  on Metal. Explicit float2/software precision does not provide arbitrary IEEE
  float64 arithmetic precision or exponent range.
- Sequential and NestedStack: host scheduling and copies around supported GPU
  children; no host numerical training fallback.
- Dense volume NormalBP, line FamilyBP, and mesh FamilyBP use GPU kernels.
  Mesh gradient addition runs on Metal. Line/mesh smoke validation currently
  covers Float32 Dense, Sequential and NestedStack, not full dtype sign-off.
- GDN uses Metal Float32 truncated-BPTT with state/conv history stop-gradient.

**Validated 2026-10-26:** all 17 NormalBP evidence sections / 128 case checks pass,
including supported dtype loops, leaves, chains and volumes. Exact bits are
required where specified, with a finite loss band of
`1e-12 * max(1, abs(reference))`. Weight comparisons permit FP32 subnormal
flushing when both values are below the smallest normal FP32. MHA leaf dtype
cases retain their ≥strict gradient floor; Softmax retains ≥loose, and GDN
retains ≥loose leaf / ≥industry volume. Stack binary 211 retains its existing
post-only `1e-18` allowance. See [the checklist](TRAINING_CHECKLIST.md) for status.

KMeans and Parallel NormalBP remain unsupported. Other training modes need their
own validation; NormalBP sign-off does not transfer to them. No build flag or
runtime setting enables a CPU numerical fallback.

Host code handles allocation, validation, graph scheduling, tensor copies, storage
encoding/decoding, and ABI export. Training still stages tensors between kernel
calls and exports updated native weights before refreshing resident mirrors.
Full graph/tape residency and fused training are unfinished. This is GPU-only
numerical execution, not a claim of zero host involvement or peak performance.
Metal targets the Apple GPU. ANE is a separate, unimplemented backend gate.

## Build and delivery

```sh
./scripts/build_macos.sh
./scripts/macos_build_w2a.sh --normalbp-all
```

Tests and CPU parity references live in sibling `w2a`, not the engine repository.
`TestGoVMetalOnly` checks rejected CPU APIs, GPU loss/gradients/updates,
accumulation, repeated weight updates/cache refresh, Dense volumes, and the
supported Float32 line/mesh cases. `TestGoVMetalNormalBP` runs all 17 requested
NormalBP groups / 128 case checks. The combined runner executes both gates and
defaults to one measured iteration per fixture; set
`WELVET_METAL_BENCH_ITERS=2000` for longer timing runs.

Shared artifact: `libwelvet_metal.dylib`. Static artifact:

```sh
cmake -S . -B build-static -DWELVET_BUILD_SHARED=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-static -j 8
```

Shader source is embedded at build time and compiled by the system Metal runtime.
Default execution does not read the source checkout or need a separate `.metal`
file. Explicit shader reload paths remain a development API.

Building this native backend currently requires CMake, an Apple Clang C++/ObjC++
toolchain and Apple's SDK/frameworks. Go is needed for w2a and future Go integration.
A generic LLVM installation alone does not replace the Apple SDK. A distributed
prebuilt backend should remove the need for end users to build CMake sources;
Go embedding/loading and release packaging remain future work.
