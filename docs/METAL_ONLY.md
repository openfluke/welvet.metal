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
  KMeans, Mamba, Metacognition, GDN, Parallel and Stack use Metal numerical kernels.
- SGD and native encoding: Metal arithmetic and native FormatNone storage updates
  across all 34 dtypes. Float32 rounds the update step before subtraction to follow
  Go's semantics; other dtypes retain their native encoding. Bias updates also run
  on Metal. Explicit float2/software precision does not provide arbitrary IEEE
  float64 arithmetic precision or exponent range.
- Sequential and NestedStack: host scheduling and copies around supported GPU
  children; no host numerical training fallback.
- Line FamilyBP uses GPU kernels for all 23 supported families, including all
  supported dtypes. CNN2/3 and ConvT2/3 have explicit rank-4/rank-5 tests.
- Mesh FamilyBP forward, reverse-time BPTT, gradient addition and SGD run on
  Metal for all 23 supported families. Clock fixtures cover 111/211 ×34 and
  222 Float32 (GDN Float32 only), disabled ingress, recurrent remote links,
  all seven Parallel combines and both KMeans outputs. Every captured node
  input/pre/post, accumulated parameter gradient, oldest input gradient and
  updated weight is compared to Go under the checklist precision policy.
  Zero lr preserves native weights. Benchmark replay must match exactly and
  runs at least two measured iterations to verify reset between repetitions.
  Shared ModeMeshBP volume Forward+Step fixtures retain separate coverage.
- Tween ModeTween uses Metal block-average gap projection and local backward/SGD
  at lr×0.5. Sequential/Residual expand Dense leaves; NestedStack retains the
  original forward inputs during local updates. All 21 supported families have
  leaf/volume evidence, including all supported dtypes (GDN Float32 only).
  GDN leaf updated weights use a ≥strict floor; other Tween leaf weights are
  exact, with the finite loss band above. Further variants retain separate gates.
- TweenChain ModeTweenChain uses Metal chain-rule backward and SGD at full lr.
  All 21 supported families have leaf/volume evidence, including all supported
  dtypes (GDN Float32 only). The volume path shares NormalBP credit, with actual
  input gradients checked through TweenChain wires. GDN leaf updated weights
  retain ≥strict; GDN volume retains ≥industry. Step/Mesh variants have separate coverage below.
- TweenSplit uses one collect tape and GPU projection/scaling by 1/N at full lr.
  TweenAlt refreshes prediction and gap between Split and Tween phases (AltTimes=1
  tested). StepTween/Split/Alt use frozen popped gaps; StepTweenChain uses true BP.
  The six leaf/Line sections each cover 80 groups, including supported leaf and
  volume dtypes across all 21 supported families (GDN Float32 only).
- MeshTween uses captured last-tick local credit at half lr; MeshTweenChain uses
  BPTT at full lr. Each has 48 groups: Dense clock 111/211/222 Float32, clock
  111/211 ×34, and all supported shared volume families/dtypes. Non-Dense clock
  parity remains open. Proxy/cache/sparse credit modes remain unsupported.
- Float32 SGD encodes tiny steps in a scaled GPU representation so stored
  subnormals survive later phases that produce normal-range weights. Other
  kernel subnormal limits remain as documented. GDN Alt refreshed loss uses
  its finite relative 1e-5 band; GDN weights still require ≥strict.
- GDN uses Metal Float32 truncated-BPTT with state/conv history stop-gradient.
  Promoted reductions and transcendentals use float2 before rounding to Float32.

**Validated 2026-10-26:** all 19 NormalBP evidence sections / 219 case groups pass,
including supported dtype loops, leaves, chains and volumes. Exact bits are
required where specified, with a finite loss band of
`1e-12 * max(1, abs(reference))`. Weight comparisons permit FP32 subnormal
flushing when both values are below the smallest normal FP32. MHA leaf dtype
cases retain their ≥strict gradient floor; Softmax retains ≥loose, and GDN
retains ≥loose leaf / ≥industry volume. Stack binary 211 retains its existing
post-only `1e-18` allowance. See [the checklist](TRAINING_CHECKLIST.md) for status.

KMeans computes distances, assignment softmax, reconstruction and complete
NormalBP gradients on Metal. Both output modes have ×34 evidence. Its only
additional allowance is finite near-zero input-gradient cancellation: both
values must have magnitude ≤`1e-12` and differ by ≤`1e-14`. Other tensor gates
remain exact and loss uses the common finite band.

Parallel supports polymorphic/nested branches and all seven combine modes.
Direct pre/post/dX/dW/SGD checks cover each combine ×34, including Filter gate
updates and rank-3 inputs. Metal preserves promoted branch-input accumulation,
Avg backward scaling, deterministic Max/SparseK ties and Go arm64 Disagree FMA
rounding. Host branch loops only schedule kernels and stage buffers.

KMeans and Parallel are also validated for StepBP, Tween, TweenChain,
StepTween and StepTweenChain. Each mode adds 50 groups: both KMeans outputs
and all seven Parallel combines ×34 in multi-hop fixtures, all 23 polymorphic
branch kinds Float32, stateful Filter capture, and 111/chain/211 ×34 plus 222
Float32 shared volume credit. GDN branches retain their existing ≥strict floor.
Parallel's Tween collect tape captures every leaf and gate before applying SGD;
its Filter forward preserves rule state transitions. All numerical operations
remain on Metal. Shared volume credit and mesh-clock parity have separate gates.

The complete `--step-tween-all` run passed 654 requested-mode groups, all 219
NormalBP groups and the Metal-only guard. The guard also executes mesh SGD for
KMeans/Parallel; full Go MeshBP clock parity is covered by `--meshbp-train`.
Other modes retain their own KMeans/Parallel validation gates. No build flag or runtime
setting enables a CPU numerical fallback.

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
./scripts/macos_build_w2a.sh --stepbp-train
./scripts/macos_build_w2a.sh --meshbp-train
./scripts/macos_build_w2a.sh --tween-train
./scripts/macos_build_w2a.sh --tweenchain-train
./scripts/macos_build_w2a.sh --tween-next-all
```

Tests and CPU parity references live in sibling `w2a`, not the engine repository.
`TestGoVMetalOnly` checks rejected CPU APIs, GPU loss/gradients/updates,
accumulation, repeated weight updates/cache refresh, Dense volumes, and the
supported Float32 line/mesh cases. `TestGoVMetalNormalBP` runs all 19 evidence sections / 219 case groups. The combined runner executes both gates and
defaults to one measured iteration per fixture. The StepBP runner executes
`TestGoVMetalStepBPTrain` (134 case groups, with dtype sweeps) and `TestGoVMetalOnly`.
The MeshBP runner executes 103 all-family clock case groups plus StepBP/shared-volume,
NormalBP and Metal-only regressions. The Tween runner executes
`TestGoVMetalTweenTrain` (130 case groups, including leaf/volume dtype sweeps) and `TestGoVMetalOnly`, which also checks projection
and nested local credit. The TweenChain runner executes
`TestGoVMetalTweenChainTrain` (130 case groups with leaf/volume dtype sweeps) plus
`TestGoVMetalOnly`, including hand-calculated chain-rule/full-rate checks.
The `--tween-next-all` batch covers the eight requested Split/Alt/Step/Mesh sections
(576 groups) plus GPU-only primitives for collect shares, refreshed/frozen gaps,
action flights, unsupported credit rejection and subnormal SGD storage.
Line, Tween and TweenChain benchmark iterations reset all
weights/configuration before each run and
return the real final loss; reset cost is included. All runners default to one
measured iteration; set
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
