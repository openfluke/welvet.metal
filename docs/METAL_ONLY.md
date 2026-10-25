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
- SGD: Metal kernel for Float32 FormatNone weights. The learning rate is supplied
  as float2, and the step is rounded to Float32 before subtraction to follow Go's
  Float32 update semantics. Nonzero learning rates and double gradients must
  fall within the normal Float32 exponent range used by float2 scratch. Non-Float32 weight updates and float64 bias updates
  return `Unsupported`.
- Sequential and NestedStack: host scheduling and copies around supported Dense
  children. Unported children return `Unsupported` through direct and runtime APIs.
- Dense volume NormalBP, line FamilyBP, and mesh FamilyBP use the same kernels.
  Mesh gradient addition runs on Metal.

MHA, SwiGLU, RMSNorm, LayerNorm, Softmax, Embedding, CNN1/2/3, ConvT1/2/3, RNN,
LSTM, Residual, Mamba, Metacognition, GDN, KMeans, and Parallel currently return
`Unsupported`. Their previous CPU numerical entry points have been retired.
They need real Metal kernels before availability can be restored. No build flag
or runtime setting enables a CPU fallback.

Host code handles allocation, validation, graph scheduling, tensor copies, storage
encoding/decoding, and ABI export. Training still stages tensors between kernel
calls and exports updated Float32 weights before refreshing the resident mirror.
Full graph/tape residency and fused training are unfinished. This is GPU-only
numerical execution, not a claim of zero host involvement or peak performance.
Metal targets the Apple GPU. ANE is a separate, unimplemented backend gate.

## Build and delivery

```sh
./scripts/build_macos.sh
./scripts/macos_build_w2a.sh --metal-only
```

Tests and CPU parity references live in sibling `w2a`, not the engine repository.
`TestGoVMetalOnly` checks rejected CPU APIs, GPU loss/gradients/updates,
accumulation, repeated weight updates/cache refresh, Dense volumes, and the
supported line/mesh cases. Existing broad parity suites remain backlog probes;
they will fail on operations that are now explicitly unsupported.

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
