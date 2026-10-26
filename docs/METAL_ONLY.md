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
- TweenSplit includes KMeans and Parallel across all supported native dtypes.
  Parallel recursively captures every trainable branch leaf and its Filter gate
  before 1/N scaling and full-lr GPU SGD. Weightless leaves do not enter N.
  Its 131 groups include all 23 families, both KMeans outputs, seven combines,
  all polymorphic branch kinds and placed-cell volume credit. New leaf and
  volume cases require exact replay across at least two measured iterations.
- TweenSplit uses one collect tape and GPU projection/scaling by 1/N at full lr.
  TweenAlt refreshes prediction and gap between Split and Tween phases (AltTimes=1
  tested). StepTween/Split/Alt use frozen popped gaps; StepTweenChain uses true BP.
  StepTweenSplit, TweenAlt and StepTweenAlt each cover 130 groups across all
  23 families, including KMeans/Parallel, supported native dtypes and shared
  volumes (GDN Float32 only). Alt uses ordinary Dense backward during its
  Split phase; standalone Split uses GradWOnly, matching Go.
- MeshTween uses captured last-tick local credit at half lr. Its production
  training entry point and generic clock wire share `step::apply_tween`.
  All 23 actual node kinds have 111/211 supported-dtype and 222 Float32 clock
  coverage, plus recurrent links, KMeans outputs, all Parallel combines and
  exact replay. Zero lr preserves native weights. The 159 groups also include
  shared volume credit. Sequential/Residual project onto Dense leaves;
  Stack/Parallel nodes use concrete local backward/SGD. No temporal BPTT or
  chain-rule input gradient is claimed for this mode.
- MeshTweenChain uses the existing mesh FamilyBP BPTT and native SGD path at
  full lr. The named `MGC1` clock wire covers all 23 actual node kinds, supported
  dtypes, recurrent links, both KMeans outputs and seven Parallel combines.
  Its 160 groups include shared volumes and exact MeshBP equivalence across
  all 23 families. Checks include every captured tensor, oldest input gradient,
  accumulated parameter gradients, updated weights and exact benchmark replay.
- Sections 14–19 implement HeadProxy, Linear, FastProxy, LinearCache,
  HeadProxyAsync and Sparse with Metal projection, local gradients and SGD.
  Cache norm/rescaling stays on the GPU; 22-epoch tests cross the default
  20-epoch refresh boundary. Async and Sparse tests exercise persistent proxy
  and rotation state. All 23 families have native leaf and actual-type volume
  parity; GDN is Float32 only.
- MeshTweenSplit, MeshTweenAlt, MeshTweenSplitFastProxy and MeshTweenSplitSparse
  apply final-tick local credit, preserving Go's frozen-gap routing. Each has
  103 clock groups plus shared-volume coverage, supported native dtypes,
  recurrent links, zero-lr preservation and exact repeated replay.
- All Line credits are supported: BP, Tween, Split, Alt and six proxy variants.
  Sections 25–29 add Linear, FastProxy, LinearCache, HeadProxyAsync and Sparse
  through the shared Metal credit implementation. Their 352 groups cover all
  23 families, supported native dtypes, warmup/pops and replay. Seven 26-tick
  Parallel cases cross the cache-refresh boundary. Eligible Parallel hops keep
  cache/proxy/rotation state across pops; flattened Dense hops have no accelerator
  state. Shared volumes reuse the corresponding parent proxy modes.
  Line HeadProxy retains its separate 65 groups. Invalid credit values fail
  before mutating the pipeline.
- Float32 SGD encodes tiny steps in a scaled GPU representation so stored
  subnormals survive later phases that produce normal-range weights. Other
  kernel subnormal limits remain as documented. GDN loss uses
  its existing finite relative 1e-5 band, including long cache-refresh fixtures; existing non-Step leaf GDN weight gates remain ≥strict; the new Line modes
  use their declared 60–100% spectrum.
- GDN uses Metal Float32 truncated-BPTT with state/conv history stop-gradient.
  Promoted reductions and transcendentals use float2 before rounding to Float32.

**Parity target:** best-effort agreement with Go, aiming for 100% and accepting
60–100% on w2a's determinism spectrum (`industry` to `exact_bit`). Larger models
and longer training runs need not follow bit-identical CPU trajectories. See
[the README policy](../README.md#numerical-parity-target). The fixture gates
below describe measured regression evidence, including stronger checks where
practical; numerical parity and performance are separate measures.

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
./scripts/macos_build_w2a.sh --training-11-24-all
./scripts/macos_build_w2a.sh --training-25-29-all
./scripts/macos_build_w2a.sh --training-all
./scripts/macos_build_w2a.sh --meshtween-train
./scripts/macos_build_w2a.sh --meshtweenchain-train
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
(1100 groups, including the expanded TweenSplit, StepTween/Chain and mesh suites) plus GPU-only primitives for collect shares, refreshed/frozen gaps,
action flights, unsupported credit rejection and subnormal SGD storage.
The `--training-all` runner executes every numbered training mode (1–29),
NormalBP and GPU-only regression guards. `--training-25-29-all` executes the
352 new Line groups, the shared proxy volume matrix and GPU-only guards.
The new Line comparisons use the 60–100% spectrum for CPU/GPU loss and weights,
report deviations, and retain independent same-backend replay checks.
The `--training-11-24-all` runner covers sections 11–24 plus TweenSplit,
NormalBP, GPU-only primitives and 16 Line Alt pop controls. Six proxy modes
cover 65 leaf groups each (LinearCache adds 23 refresh-boundary groups), plus
65 Line HeadProxy groups. Each proxy volume mode covers 69 groups (LinearCache
adds 23 refresh-boundary groups). Generic volume/clock fixtures require exact
replay across at least two measured iterations; leaf fixtures also replay from
independently rebuilt state. Proxy volume comparisons accept the project
industry-to-exact spectrum (60–100%) and report deviations; the observed cache/async
weight differences reached strict (80%). Shapes and counts remain exact.
GDN updated weights in leaf fixtures retain ≥strict; its loss uses its existing
finite relative 1e-5 gate. Other regression fixture gates remain as declared above.
The MeshTween runner covers its 159 groups plus MeshBP, ModeTween and Metal-only
regressions. The MeshTweenChain runner covers its 160 groups plus MeshTween,
ModeTweenChain and Metal-only regressions. Generic clock replay uses at least
two measured iterations and
requires exact repeated Metal outputs and weights.
The TweenSplit runner covers its 131 groups plus ModeTween, TweenAlt and
Metal-only regressions. Split reuses Tween's nested leaf collector; shared
volume Split counts placed KMeans/Parallel cells as local update units.
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
