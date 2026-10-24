# welvet.metal — Welvet (3D volumetric AI engine) on Apple Metal

**welvet.metal is a full rebuild of Welvet — the 3D volumetric AI engine — in C++ / Apple Metal.** Same engine shape as Go Welvet: numerical types, layers, volumetric architecture (cells in a 3D grid), forward/backward, training modes, systems, model loading, inference, checkpoints. Execution is Metal compute + resident `MTLBuffer`s. Host orchestration is C++ (ObjC++ at the Metal boundary).

A `1 × 1 × 1` volume is the ordinary single-cell neural-network case (layers inside that cell). Larger volumes connect cells that each hold their own layer stacks. This Metal rebuild must own and execute that same topology — not “a Dense GEMV demo with a GPU sticker.”

It will run directly from C++ and through a stable C ABI. Go Welvet can later delegate via **NAC** (Native Acceleration Compute). The Metal engine must also build, load models, execute volumes, and train **without** a Go process.

| | |
|--|--|
| **Version** | **`0.01`** |
| **Engine board** | **~early** — Dense FormatNone×34 Metal forward exact_bit vs Go (w2a `go_v_metal`) |
| **Compute** | Apple GPU (Metal) — primary · Apple Neural Engine — separate gate |
| **Parity reference** | [welvet](https://github.com/openfluke/welvet) Go **v1.2.0** (100/100) |
| **CPU sibling** | [welvet.cpp](https://github.com/openfluke/welvet.cpp) **0.67** (semantics/wire to mirror — **score not inherited**) |
| **Validation** | [w2a](https://github.com/openfluke/w2a) planned `Go_v_Metal` (determinism + speed) |

**This README is the runway.** Every unchecked cell is planned Metal work. Tick ⬜ → 🚧 → ✅ here as gates land. Do not keep a parallel “real” checklist elsewhere. [GAPS.md](GAPS.md) is the machine-oriented census mirror; [VERSIONING.md](VERSIONING.md) is the `0.x` formula.

### What exists today (M0 host only — not engine credit)

```bash
./scripts/build_macos.sh
./build/example_metal_smoke          # device + shader compile + buffer RT
./build/welvet_dense_fwd --version   # 0.00
./build/welvet_dense_fwd --identity  # metal:<Apple GPU>…
./build/welvet_dense_fwd --gaps
```

| M0 host | Status |
|---|:---:|
| macOS arm64 CMake → `libwelvet_metal.dylib` | ✅ |
| `MTLDevice` probe + runtime identity | ✅ |
| Runtime compile `shaders/welvet_dense.metal` | ✅ |
| Shared buffer upload/download | ✅ |
| Smoke PSO `welvet_smoke_saxpy` | ✅ |
| C ABI version / init / gaps | ✅ |
| Dense FormatNone×34 Metal forward (soft-double GEMV, exact_bit vs Go) | ✅ |
| Remaining layers / volume / train / systems / model | ⬜ |

---

## Contents

- [Relationship](#relationship)
- [3D volumetric shape](#3d-volumetric-shape)
- [Non-negotiable rules](#non-negotiable-rules)
- [Scoreboard and credit rules](#scoreboard-and-credit-rules)
- [Engine rebuild scoreboard — 100 points](#engine-rebuild-scoreboard--100-points)
- [Full engine subsystem runway](#full-engine-subsystem-runway)
- [Coverage axes](#coverage-axes)
- [DTypes — 34 numerical types](#dtypes-coredtype--34-numerical-types)
- [Quant formats — 20](#quant-formats-quantformat--20)
- [Layer runway — 23 kinds + seqmix](#layer-runway--23-kinds--seqmix)
- [Kernel and training runway](#kernel-and-training-runway)
- [Training modes — all 29 + cameral](#training-modes--all-29--cameral)
- [Volumetric execution and placement](#volumetric-execution-and-placement)
- [Metal platform runway](#metal-platform-runway)
- [Systems — dna / evolution / tween / tanhi / telemetry](#systems--engine-7)
- [Model / IO](#model--io--engine-8)
- [Lucy / stubs / apps](#lucy--stubs--apps)
- [FFI and C ABI](#ffi-and-c-abi)
- [Resident weights and memory](#resident-weights-and-memory)
- [Validation and performance (w2a)](#validation-and-performance-w2a)
- [Implementation sequence](#implementation-sequence)
- [Source layout](#source-layout)

---

## Relationship

| Repository | Responsibility |
|---|---|
| [welvet](https://github.com/openfluke/welvet) | Go engine — volumetric architecture, layers, dtypes, quants, train modes, ENTITY, CPU/SIMD/WebGPU |
| [welvet.cpp](https://github.com/openfluke/welvet.cpp) | Portable C++ CPU engine + C ABI (port **semantics**, not CPU kernels) |
| **welvet.metal** | **This repo** — full Welvet engine on Apple Metal |
| [welvet.qnn](https://github.com/openfluke/welvet.qnn) | Qualcomm Hexagon HTP sibling pattern |
| [w2a](https://github.com/openfluke/w2a) | Parity / speed / determinism (`Go_v_Metal` planned like `Go_v_Cpp`) |

Go is the parity reference. Metal owns its own topology, model state, numerical storage, forward/backward, and training on GPU. A native Metal run does not need Go to orchestrate each layer.

### Architecture sketch — intended flow

```text
C++ application / CLI                 Go Welvet or w2a / future NAC
          |                                       |
     C++ engine API                          stable C ABI
          |                                       |
          +-------- welvet.metal engine ----------+
                              |
          model / ENTITY / tokenizer / transformer / sampling
                              |
             3D architecture: cells, stacks, branches, links
                              |
          forward / backward / training / Step / Mesh / systems
                              |
                MTLDevice + library + pipelines + capability queries
                              |
             resident MTLBuffer weights + tensors + training state
                              |
                    Apple GPU (Metal compute)
                    Apple Neural Engine (separate gate — never implied)
```

---

## 3D volumetric shape

Welvet is **not** “a list of layers.” It is a **3D volumetric AI engine**:

| Concept | Meaning |
|---|---|
| **Volume / Grid** | `Depth × Rows × Cols` cells; each cell holds a layer stack |
| **`1×1×1`** | Ordinary single-network case (still a volume of one cell) |
| **`2×2×2`, `3×3×3`, non-cubic** | Multi-cell topologies with hops and remote links |
| **Line / Step\*** | 1D systolic pipe through a stack (fill/drain ticks) |
| **Mesh\*** | Credit and scheduling across volumetric links |
| **Cameral / Parallel** | Branch modes, hemispheres, combine — still volumetric-aware |

Metal must keep weights, activations, tapes, and credit **resident** across cells where the schedule requires it. Host orchestration without device residency is not “volumetric Metal done.”

---

## Non-negotiable rules

Same spirit as Go Welvet:

1. **No engine tests in this tree** — validation lives in w2a (`Go_v_Metal`).
2. **No silent fallbacks** — unsupported dtype/layer/mode → hard error (no fake “Metal” that ran on host ALU unmarked).
3. **Nothing hardcoded to float32 as storage truth** — FormatNone + DType (or packed quant) is authoritative; scratch f32 at the Metal ALU boundary must be explicit and accounted.
4. **No QAT** — `DType` + `QuantFormat` are storage truth.
5. **Train keeps storage truth** — updates preserve declared encoding (or unpack→update→re-pack with documented scratch).
6. **One Go feature folder → one Metal runway row** (tick in this README).
7. **v1.0.0 = engine board 100/100** on Metal. Apps/stubs/ANE are tracked but do not invent board points unless a gate is written first.
8. **Determinism honesty** — prefer `exact_bit` vs Go; if Metal reductions cannot match, document the band **before** ✅.

---

## Scoreboard and credit rules

| Mark | Meaning | Credit |
|---|---|---:|
| ⬜ | Planned; no validated Metal implementation | 0% |
| 🚧 | Implemented subset with reproducible evidence + listed gaps | 50% of gate |
| ✅ | Declared acceptance gate passes (w2a + device identity) | 100% of gate |
| N/A | Outside Metal scope (reason required) | No credit |
| BLOCKED | Missing device / SDK / unresolved semantics | 0% |

**Version:** `welvet_version() = 0.{round(earned)}` until **100 → 1.0.0**. Only this engine board moves the tag. M0 host scaffolding does **not** earn engine points.

---

## Engine rebuild scoreboard — 100 points

Identical ten categories / weights as Go Welvet. Credit only from **this** Metal tree’s evidence.

| # | Category (Go board) | Wt | Metal today | Status | Earned |
|---:|---|---:|---|:---:|---:|
| 1 | **Foundation** — `core`, `weights`, quant, Metal compute base, tiling policy | 15 | FormatNone×34 encode/decode + Metal soft-double compute | 🚧 | **~8** |
| 2 | **Dense MatVec** — FormatNone×34 + quants; fwd/bwd/update on Metal | 15 | FormatNone×34 **forward** exact_bit (w2a); quants/bwd gap | 🚧 | **~7** |
| 3 | **Transformer stack** — MHA, SwiGLU, norms, Softmax, Emb, Sequential, Residual, seqmix | 14 | — | ⬜ | **0** |
| 4 | **CNN / RNN / LSTM** | 6 | — | ⬜ | **0** |
| 5 | **Extended layers** — GDN, ConvT1–3, Mamba, KMeans, Parallel, Metacognition | 7 | — | ⬜ | **0** |
| 6 | **Runtime + architecture** — volumetric grid, forward, backward, training, step/mesh | 8 | Grid dims stub only | ⬜ | **0** |
| 7 | **Systems** — dna, evolution, tween, tanhi, telemetry | 5 | — | ⬜ | **0** |
| 8 | **Model / IO** — tokenizer, entity, transformer, sampling, hf | 8 | — | ⬜ | **0** |
| 9 | **Training credit** — 29 modes + cameral BranchModes / CamSync | 8 | TrainMode enum only | ⬜ | **0** |
| 10 | **Peak fused / no host ALU** — fused Metal kernels, decoder fuse, provenance | 14 | — | ⬜ | **0** |
| | **Total → version** | **100** | | **🚧** | **~15 / 100 → `0.01`** (recompute formally after more gates) |

**Path to 1.0:** same order Go used / welvet.cpp is following — Dense FormatNone + NormalBP → layers → Step/Mesh/Tween* → Systems + Model/IO → CamSync + peak fused Metal. welvet.cpp’s ✅ list is the **port order**, not transferred credit.

---

## Full engine subsystem runway

### Foundation, weights and Metal compute — §1 / §2

| Go reference | Metal rebuild target | Status |
|---|---|:---:|
| `core/` | Tensor representation, all 34 dtypes, activations, layer IDs, execution config | 🚧 enum/stub |
| `weights/` | Dtype-native parameter storage, MatVec/transpose, grad/update, snapshots — **MTLBuffer resident** | ⬜ |
| `quant/` | All 20 packed formats Pack/Unpack/MatVec on Metal (or honest unsupported) | ⬜ |
| `tiling/` | Threadgroup / grid dispatch policy, tails, workspace budgets, deterministic scheduling | ⬜ |
| `simd/` | N/A as Plan 9 — replaced by Metal kernels (do not ship host AVX as “Metal”) | N/A |
| `webgpu/` | Reference for device semantics only — implement as Metal, not WGSL | N/A |
| `fusedgpu/` | Decoder/op fusion with resident intermediates + execution provenance on MTL | ⬜ |
| Metal runtime | Device, library, pipeline, command buffer, sync, error surface | 🚧 M0 host |

### Architecture and runtime — §6

| Go reference | Metal rebuild target | Status |
|---|---|:---:|
| `architecture/` | Own the 3D volume, coordinates, cell layers, graph links, topology validation | 🚧 dims stub |
| `runtime/dispatch/` | Dispatch every supported op/variant with explicit Metal capability checks | ⬜ |
| `runtime/forward/` | Full model/volume execution without Go orchestration | ⬜ |
| `runtime/backward/` | Reverse dependency traversal, tapes, gradients, joins, shared params on device | ⬜ |
| `runtime/training/` | Loss/credit/update, batches, state, training-mode selection on Metal | ⬜ |
| `runtime/step/` | Systolic pipes, Line/Stack ticks, fill/drain, weight generations, RoleTrain/RoleAction | ⬜ |
| Mesh / volumetric scheduling | Volume dependencies, branch joins, cross-cell credit on Metal-resident state | ⬜ |

### Systems — engine §7

| Go reference | Wt | Metal rebuild target | Status | Earned |
|---|---:|---|:---:|---:|
| `systems/dna/` | 1 | Encode/decode topology + params; native dtype preserved | ⬜ | 0 |
| `systems/evolution/` | 1 | DNA splice, mutation, cloning, NEAT-compatible behavior | ⬜ | 0 |
| `systems/tween/` | 1 | Target-propagation / local-credit ops and update budgets on Metal | ⬜ | 0 |
| `systems/tanhi/` | 1 | TANHI structural/activation reporting + telemetry transport | ⬜ | 0 |
| `systems/telemetry/` | 1 | Blueprint, counts, estimates, execution/learning observability | ⬜ | 0 |
| **Subtotal** | **5** | | **⬜** | **0 / 5** |

### Model / IO — engine §8

| Go reference | Wt | Metal rebuild target | Status | Earned |
|---|---:|---|:---:|---:|
| `model/tokenizer/` | 1.5 | BPE/HF tokenization, vocab/specials, encode/decode parity | ⬜ | 0 |
| `model/entity/` | 2 | ENTITY inspect/load/write, dtype-native blobs, cameral/Stack round trips | ⬜ | 0 |
| `model/transformer/` | 2.5 | Decoder construction, generation, KV cache, LM head, runtime integration | ⬜ | 0 |
| `model/sampling/` | 1 | ArgMax, temperature/TopK, penalties, banned IDs, deterministic RNG | ⬜ | 0 |
| `model/hf/` | 1 | Snapshot inspect, architecture detect, declared tensor-loader conversions | ⬜ | 0 |
| **Subtotal** | **8** | | **⬜** | **0 / 8** |

### Lucy / stubs / apps

| Surface | Metal runway | Board | Status |
|---|---|---|:---:|
| `lucy/` | LPD / TrainStackCE / Lean density / Score | §8/§9 gates before credit | ⬜ |
| `model/wav2vec2/` | Inventory + port applicable speech paths | model scope | ⬜ |
| CamSync / BlendStores | Soft/hard blend across resident stores | §9 | ⬜ |
| `stub/` seed · serialization · hardware · memory · fountain · donate · … | Inventory; stubs ≠ done | off-board until gated | ⬜ |
| `apps/` (octo, …) | Native CLI shells on Metal engine | off-board | ⬜ |

**Standalone acceptance:** a native Metal-backed C++ app can construct or load a model, create its **volume**, execute forward, run the selected training rule, save/reload supported state, and continue — **no Go process**. FFI-only layer parity is insufficient.

---

## Coverage axes

A support claim is a tuple:

```text
(layer + variant + shape + numerical dtype + activation type
 + gradient type + accumulator type + pass + training mode
 + Metal device/path + OS/arch + layout + residency + execution mode)
```

| Axis | Runway |
|---|---|
| Topology | `1×1×1`, `2×2×2`, `3×3×3`, non-cubic, cell stacks, connected/remote boundaries |
| Layer inventory | 23 kinds in Go `core/layer.go` + `layers/seqmix` |
| Storage dtype | All 34 `core.DType` |
| Quant | FormatNone + all 20 `quant.Format` (or explicit unsupported) |
| Activation host type | Go `core.Numeric` kinds; ABI uses explicit representations |
| Pass | Forward; dX; dW/db; state gradients when applicable |
| Training | Full backward + independently callable credit/update primitives |
| Execution | Metal compute queues; documented host epilogue if any |
| Device | Named Apple GPU (+ ANE only if separately validated) |
| Platform | macOS arm64 first; iOS device later |
| Shape/layout | Batch, sequence, rank, stride, alignment, block tails |
| Precision | Stored / transferred / decoded / accumulated / output / grad / update |
| State | Stateless, recurrent, KV cache, saved activations, versioned weights, CamSync |

---

## DTypes (`core.DType`) — 34 numerical types

FormatNone storage. Metal column = validated on-device path (not host decode pretending to be Metal).

| # | DType | Pack/RT | Metal F | Metal B | Metal update | Notes |
|--:|-------|:------:|:------:|:------:|:------:|-------|
| 0 | Float64 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 1 | Float32 | ✅ | ✅ | ⬜ | ⬜ | soft-double GEMV |
| 2 | Float16 | ⬜ | ⬜ | ⬜ | ⬜ | native Metal half |
| 3 | BFloat16 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 4 | FP8E4M3 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 5 | FP8E5M2 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 6 | Int64 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 7 | Int32 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 8 | Int16 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 9 | Int8 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 10 | Uint64 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 11 | Uint32 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 12 | Uint16 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 13 | Uint8 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 14 | Int4 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 15 | Uint4 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 16 | FP4 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 17 | Int2 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 18 | Uint2 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 19 | Ternary | ⬜ | ⬜ | ⬜ | ⬜ | |
| 20 | Binary | ⬜ | ⬜ | ⬜ | ⬜ | |
| 21 | Int | ⬜ | ⬜ | ⬜ | ⬜ | ABI width explicit |
| 22 | Uint | ⬜ | ⬜ | ⬜ | ⬜ | |
| 23 | Uintptr | ⬜ | ⬜ | ⬜ | ⬜ | never serialize as tensor id |
| 24 | Complex64 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 25 | Complex128 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 26 | NF4 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 27 | FP6 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 28 | Int6 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 29 | Uint6 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 30 | Int5 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 31 | Uint5 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 32 | Int3 | ⬜ | ⬜ | ⬜ | ⬜ | |
| 33 | Uint3 | ⬜ | ⬜ | ⬜ | ⬜ | |

### Numerical semantics gates

| Concern | Status |
|---|:---:|
| Integer/low-bit learning rules match Go | ⬜ |
| Binary/ternary bit order, codes, masks | ⬜ |
| Complex real/imag layout | ⬜ |
| FP8/FP6/FP4/NF4 exact code tables | ⬜ |
| Accumulator dtype + reduction order + reproducibility policy | ⬜ |
| Explicit lossy conversion permission + scratch lifetime | ⬜ |
| Unsupported arithmetic → hard error (no silent widen) | ⬜ |

---

## Quant formats (`quant.Format`) — 20

Packed layouts. Row **None** = FormatNone + DType (not “FP32 master”).

| Format | Metal pack+MatVec | Metal Dense F | Metal Dense B/update |
|--------|:-----------------:|:-------------:|:--------------------:|
| None | ⬜ (via weights + DType) | ⬜ | ⬜ |
| Q8_0 | ⬜ | ⬜ | ⬜ |
| Q4_0 | ⬜ | ⬜ | ⬜ |
| Q4_1 | ⬜ | ⬜ | ⬜ |
| Q5_0 | ⬜ | ⬜ | ⬜ |
| Q5_1 | ⬜ | ⬜ | ⬜ |
| Q2_K | ⬜ | ⬜ | ⬜ |
| Q3_K | ⬜ | ⬜ | ⬜ |
| Q4_K | ⬜ | ⬜ | ⬜ |
| Q5_K | ⬜ | ⬜ | ⬜ |
| Q6_K | ⬜ | ⬜ | ⬜ |
| IQ1_S | ⬜ | ⬜ | ⬜ |
| IQ2_XXS | ⬜ | ⬜ | ⬜ |
| IQ2_XS | ⬜ | ⬜ | ⬜ |
| IQ3_XXS | ⬜ | ⬜ | ⬜ |
| IQ3_S | ⬜ | ⬜ | ⬜ |
| IQ4_NL | ⬜ | ⬜ | ⬜ |
| IQ4_XS | ⬜ | ⬜ | ⬜ |
| TernaryPacked | ⬜ | ⬜ | ⬜ |
| BinaryPacked | ⬜ | ⬜ | ⬜ |
| AffinePacked | ⬜ | ⬜ | ⬜ |

---

## Layer runway — 23 kinds + seqmix

`dX` = input-gradient contract; `dW/db` = every trainable parameter. Metal F/B = on-device (host epilogue must be labeled).

| ID | Layer | Forward target | Backward target | Metal F | Metal B |
|---:|---|---|---|:---:|:---:|
| 0 | **Dense** | GEMV/GEMM, bias, activation | dX, dW, act'; ApplyGradSGD | ✅ FormatNone×34 | ⬜ |
| 1 | **MultiHeadAttention** | Q/K/V/O, attention, masks, RoPE, cache | Proj + attn + RoPE grads (ModeSelf) | ⬜ | ⬜ |
| 2 | **SwiGLU** | Gate/up/down, SiLU product | All projs + product/SiLU′ | ⬜ | ⬜ |
| 3 | **RMSNorm** | RMS + scale | dX + scale grad | ⬜ | ⬜ |
| 4 | **CNN1** | 1D conv | dX, kernel grads | ⬜ | ⬜ |
| 5 | **CNN2** | 2D conv | dX, kernel grads | ⬜ | ⬜ |
| 6 | **CNN3** | 3D conv | dX, kernel grads | ⬜ | ⬜ |
| 7 | **RNN** | Recurrent + state | BPTT + state grads | ⬜ | ⬜ |
| 8 | **LSTM** | Gates + h/c | Gate/param/h/c grads | ⬜ | ⬜ |
| 9 | **LayerNorm** | Mean/var, scale, bias | dX, scale/bias | ⬜ | ⬜ |
| 10 | **ConvTransposed1D** | 1D transposed | dX, kernel | ⬜ | ⬜ |
| 11 | **ConvTransposed2D** | 2D transposed | dX, kernel | ⬜ | ⬜ |
| 12 | **ConvTransposed3D** | 3D transposed | dX, kernel | ⬜ | ⬜ |
| 13 | **Embedding** | Gather | Scatter/reduction; defined token dX | ⬜ | ⬜ |
| 14 | **KMeans** | Soft assignment + centroids | Declared differentiable paths | ⬜ | ⬜ |
| 15 | **Softmax** | Standard/grid/temp/variants | JVPs (dX; weightless) | ⬜ | ⬜ |
| 16 | **Parallel** | Branches, routing, combine, cameral | Branch credit + BranchModes | ⬜ | ⬜ |
| 17 | **Sequential** | Ordered child chain | Reverse child credit | ⬜ | ⬜ |
| 18 | **Residual** | F(x) + skip | F reverse + skip grad | ⬜ | ⬜ |
| 19 | **Metacognition** | Observed compute + rules | lastGate-scale + Observed bwd | ⬜ | ⬜ |
| 20 | **Mamba** | Selective SSM scan | Declared BPTT + projs | ⬜ | ⬜ |
| 21 | **GDN** | Gated DeltaNet | Truncated-BPTT contract (f32-first) | ⬜ | ⬜ |
| 22 | **Stack** | Heterogeneous nested chain | Nested child credit | ⬜ | ⬜ |
| helper | **seqmix** | Sequence-mixing helpers | Helper grads | ⬜ | ⬜ |

### Layer acceptance details

| Family | Required before ✅ | Status |
|---|---|:---:|
| Dense | Transpose, batch, tails, strides, bias, act, dX-only, dW-only, update-only, dtype boundaries | ⬜ |
| Attention | Causal/cross/GQA; masks; ALiBi; sliding; dropout/RNG; KV invalidation | ⬜ |
| Norms | Epsilon, axes, small variance, saved stats, param grads | ⬜ |
| Softmax | Stable reductions; masks; temperature; Sparsemax/Entmax/Gumbel where present | ⬜ |
| Activations | Linear, ReLU, SiLU, GELU, Tanh, Sigmoid, LeakyReLU, ReLU2 + derivatives | ⬜ |
| Convolution | Non-square/cubic, dilation, groups, overlap accum, transpose shapes | ⬜ |
| Recurrence/SSM | Initial/final state, sequence bounds, reset, tape, truncation | ⬜ |
| Composition | Nested non-Dense; shared stores; branch order; no duplicate updates | ⬜ |
| Discrete/index/rule | Explicit differentiability matching Go | ⬜ |

### Dense first slice (M1)

| # | Case | Status |
|---|---|:---:|
| 1 | Metal soft-double GEMV (Apple has **no HW float64**) | ✅ |
| 2 | Bias + activation on Metal (`welvet_dense_bias_act`) | ✅ |
| 3 | Leaf forward exact_bit vs Go Float32 | ✅ |
| 4 | Leaf forward ×34 FormatNone exact_bit vs Go | ✅ `go_v_metal` |
| 5 | 3-way speed Go / C++ / Metal + pick winner | ✅ (tiny shapes: C++ wins; Metal ~dispatch-bound) |
| 6 | **No host MatVec / no f32-acc substitute** | ✅ |
| 7 | NormalBP dX/dW/db + SGD Float32 resident | ⬜ |
| 8 | NormalBP ×34 + volume `1×1×1` / `2×2×2` | ⬜ |

---

## Kernel and training runway

### Shared compute primitives

| Primitive | Forward/use | Backward/update | Status |
|---|---|---|:---:|
| GEMV/GEMM | Dense + shared projections | GEMVᵀ, dX, outer/dW, db | ⬜ |
| Reduction | Sum, mean, var, max, norm, LSE | Reduction grads + broadcast | ⬜ |
| Elementwise | Bias, mul, add, acts, gates | Derivatives / JVPs | ⬜ |
| Layout/shape | View, transpose, reshape, concat, split | Inverse mapping | ⬜ |
| Gather/scatter | Embedding, indexing, routing | Duplicate-index policy | ⬜ |
| Convolution | im2col or native Metal conv | Overlap accum + filter grads | ⬜ |
| Attention/position | QK, scale, mask, AV, RoPE | Attn + positional grads | ⬜ |
| Scan/recurrence | RNN/LSTM/Mamba/GDN | Declared BPTT/truncation | ⬜ |
| Optimizer/update | Storage-preserving SGD first | Masks, LR, rounding | ⬜ |

---

## Training modes — all 29 + cameral

Canonical equations: [Welvet `training_modes.md`](https://github.com/openfluke/welvet/blob/main/training_modes.md).  
Tick **Inside node** (`1×1×1`) and **3D volumetric** separately. **Signed** = all supported FormatNone dtypes for that layer under the mode.

| # | TrainMode | Family | Inside node | 3D volumetric | Signed | Ver |
|---|---|---|:---:|:---:|:---:|---|
| 1 | NormalBP | BP | ⬜ | ⬜ | ⬜ | |
| 2 | StepBP | BP | ⬜ | ⬜ | ⬜ | |
| 3 | MeshBP | BP | ⬜ | ⬜ | ⬜ | |
| 4 | Tween | Tween | ⬜ | ⬜ | ⬜ | |
| 5 | StepTween | Tween | ⬜ | ⬜ | ⬜ | |
| 6 | MeshTween | Tween | ⬜ | ⬜ | ⬜ | |
| 7 | TweenChain | TweenChain | ⬜ | ⬜ | ⬜ | |
| 8 | StepTweenChain | TweenChain | ⬜ | ⬜ | ⬜ | |
| 9 | MeshTweenChain | TweenChain | ⬜ | ⬜ | ⬜ | |
| 10 | TweenSplit | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 11 | StepTweenSplit | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 12 | MeshTweenSplit | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 13 | TweenAlt | TweenAlt | ⬜ | ⬜ | ⬜ | |
| 14 | StepTweenAlt | TweenAlt | ⬜ | ⬜ | ⬜ | |
| 15 | MeshTweenAlt | TweenAlt | ⬜ | ⬜ | ⬜ | |
| 16 | TweenSplitHeadProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 17 | TweenSplitLinear | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 18 | TweenSplitFastProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 19 | TweenSplitLinearCache | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 20 | TweenSplitHeadProxyAsync | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 21 | TweenSplitSparse | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 22 | MeshTweenSplitFastProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 23 | MeshTweenSplitSparse | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 24 | StepTweenSplitHeadProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 25 | StepTweenSplitLinear | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 26 | StepTweenSplitFastProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 27 | StepTweenSplitLinearCache | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 28 | StepTweenSplitHeadProxyAsync | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 29 | StepTweenSplitSparse | TweenSplit | ⬜ | ⬜ | ⬜ | |
| — | Freeze | Freeze | ⬜ | ⬜ | ⬜ | |
| — | Shadow | Freeze | ⬜ | ⬜ | ⬜ | |
| — | Adversarial | BP-gated | ⬜ | ⬜ | ⬜ | |
| — | Memory | BP-gated | ⬜ | ⬜ | ⬜ | |
| — | CamSync / BlendStores | — | ⬜ | ⬜ | ⬜ | |
| — | Cameral BranchModes (Parallel) | — | ⬜ | ⬜ | ⬜ | |

Per-layer evidence tables (leaf ×34, volume shapes, speed) get appended under each mode as suites land — still **in this README** (or linked anchors in this file), not a replacement checklist.

### Sign-off template (copy per mode × layer)

| # | Case | Status |
|---|---|:---:|
| 1 | Leaf Float32 (grads/weights/loss as applicable) | ⬜ |
| 2 | Leaf × all 34 FormatNone dtypes | ⬜ |
| 3 | Volume `1×1×1×1` Float32 | ⬜ |
| 4 | Volume `1×1×1×2` chain Float32 | ⬜ |
| 5 | Volume `1×1×1×1` ×34 | ⬜ |
| 6 | Volume `2×2×2×1` Float32 | ⬜ |
| 7 | Volume `2×1×1×1` ×34 | ⬜ |
| 8 | Speed: goµs / metalµs / spd%Δ + `welvet_runtime_identity` | ⬜ |

---

## Volumetric execution and placement

| Capability | Acceptance | Status |
|---|---|:---:|
| Single cell | Layer stack in `1×1×1` on Metal with Go parity | ⬜ |
| Multi-cell volume | Coords, ordering, links, boundary tensors intact | ⬜ |
| Non-cubic volumes | No hidden cubic-only assumption | ⬜ |
| Per-layer placement | Cell/layer/child → queryable Metal path | ⬜ |
| Partition fusion | Fuse consecutive ops without changing visible semantics | ⬜ |
| Heterogeneous forward | Explicit host↔Metal handoff | ⬜ |
| Heterogeneous backward | Gradient edges + tape ownership across boundary | ⬜ |
| Step systolic | Fill/drain, ticks, Line/Stack, RoleTrain/RoleAction | ⬜ |
| Mesh execution | Dependency order, joins, credit over links | ⬜ |
| Placement cost | Upload/download, queue latency, compile, memory pressure | ⬜ |
| Plan invalidation | Rebuild on shape/topology/weights/device/precision/mode change | ⬜ |

**Placement:** discover → query exact support → estimate cost → select → record → execute. Tiny `1×1×1` may prefer host; that must be an explicit plan, not a silent lie labeled Metal.

---

## Metal platform runway

| Gate | Status |
|---|:---:|
| macOS Apple silicon build | ✅ |
| Device enumerate + identity string | ✅ |
| Runtime `.metal` compile | ✅ |
| Shared `MTLBuffer` RT | ✅ |
| Smoke compute PSO | ✅ |
| Command buffer dispatch helper (encode/commit/wait/errors) | ⬜ |
| Precompiled `.metallib` release path | ⬜ |
| Determinism policy doc (reduction order, threadgroups, bands) | ⬜ |
| iOS device framework / static artifact | ⬜ |
| iOS simulator slice (no device accel implied) | ⬜ |
| Apple Neural Engine / Core ML (separate — never implied by Metal ✅) | ⬜ |

| Platform | Artifact | Status |
|---|---|:---:|
| macOS arm64 | `.dylib` / headers / shaders | 🚧 M0 |
| macOS x86_64 | Explicitly bounded or dropped | ⬜ |
| iOS arm64 | Static/framework | ⬜ |

---

## FFI and C ABI

| Surface | Status |
|---|:---:|
| ABI negotiation / version | 🚧 `welvet_version` = `0.00` |
| Metal init / available / identity | ✅ |
| Gaps census counters | ✅ |
| Tensor / weight / layer descriptors | ⬜ |
| Engine/model handles (volume create/load/train/checkpoint) | ⬜ |
| Dense forward / NormalBP wire magics (w2a) | ⬜ |
| All layer NormalBP magics (mirror welvet.cpp names) | ⬜ |
| Volume / Step / Mesh / Tween* magics | ⬜ |
| Go NAC dispatcher selects Metal | ⬜ |

Boundary rules match welvet.cpp: versioned descriptors, opaque handles, no C++ exceptions through C, explicit ownership, async inputs owned/copied, provenance in traces.

---

## Resident weights and memory

| Capability | Status |
|---|:---:|
| Authoritative owner (host vs Metal generation) | ⬜ |
| Upload once / reuse across fwd/bwd/update | ⬜ |
| Preserve storage truth (no undeclared f32 master) | ⬜ |
| Derived device layout cache + invalidation | ⬜ |
| Resident activations / tapes across adjacent layers | ⬜ |
| Resident gradients + clear policy | ⬜ |
| Resident update in declared storage semantics | ⬜ |
| Host-update mode (export grads → refresh device) | ⬜ |
| CamSync on resident stores | ⬜ |
| Checkpoint/export for ENTITY flows | ⬜ |
| Migration before device/plan change | ⬜ |
| Memory budget accounting | ⬜ |
| Unified/shared memory only under explicit sync contract | ⬜ |
| Concurrency / device-loss honesty | ⬜ |

---

## Validation and performance (w2a)

| Gate | Status |
|---|:---:|
| Pin Welvet + welvet.cpp revisions in fixtures | ⬜ |
| `Go_v_Metal` harness (parity + speed) | ⬜ |
| Optional C++↔Metal cross-check on same Mac | ⬜ |
| Standalone construct→volume→train→save/reload | ⬜ |
| Numerical / forward / backward / train / volume / modes matrices | ⬜ |
| Capability honesty (unsupported fails clearly) | ⬜ |
| Device identity on every evidence log | 🚧 helper ready |
| Distribution smoke on released artifact | ⬜ |

### Performance template

| Workload | dtype | pass/mode | Go µs | Metal µs | spd%Δ | exact? | device | evidence |
|---|---|---|---:|---:|---:|:---:|---|---|
| — | — | — | — | — | — | — | — | ⬜ |

Measure cold compile, initial upload, warm resident, sync, export separately. Metal is not assumed faster than Go or welvet.cpp.

---

## Implementation sequence

| Milestone | Concrete outcome | Status |
|---|---|:---:|
| **M0** — Metal host + this README runway | Device/library/buffer/smoke + full Go inventory checklist | ✅ |
| **M1** — Dense FormatNone forward on Metal | w2a `go_v_metal` exact_bit×34 + 3-way speed | ✅ |
| **M2** — Dense NormalBP resident | Multi-step train parity | ⬜ |
| **M3** — Dtype / quant breadth | ×34 (+ quants or honest gaps) | ⬜ |
| **M4** — Layer breadth | Transformer → CNN/RNN → extended | ⬜ |
| **M5** — Full volumetric runtime | Grid + Step/Mesh + credit families | ⬜ |
| **M6** — Systems + Model/IO + Lucy | DNA…telemetry, ENTITY/HF/…, LPD | ⬜ |
| **M7** — Peak fused Metal | Minimal host ALU + decoder fuse | ⬜ |
| **M8** — iOS + ANE inventory | Device artifact; ANE separate | ⬜ |
| **M9** — NAC + release review | Go delegate + clean-env smoke; board → 1.0.0 | ⬜ |

---

## Source layout

```text
include/welvet/              public C++ engine API
include/welvet/metal/        device, buffer, library, pipeline
include/welvet_c.h           stable C ABI
src/core/                    dtype, activate, tensor, status
src/metal/                   ObjC++ Metal runtime (*.mm)
src/weights/                 FormatNone store → resident MTLBuffer
src/layers/…                 layer packages (Dense first)
src/architecture/            3D Grid / volume
src/runtime/                 forward, backward, training, step, mesh
src/systems/                 dna, evolution, tween, tanhi, telemetry
src/gaps/                    census (mirrors this README)
src/interop/                 C ABI
shaders/                     .metal kernels
tools/dense_fwd/             w2a wire tool
examples/metal_smoke/        M0 smoke
scripts/build_macos.sh
README.md                    ← you are here (authoritative checklist)
GAPS.md                      machine-oriented gap dump
VERSIONING.md                0.x formula
```

---

## How to use this README

1. Pick the next ⬜ (after M0: Dense MatVec dispatch → M1).
2. Implement Metal kernels + host wiring; preserve Go / welvet.cpp semantics.
3. Prove in w2a (`Go_v_Metal`): determinism + speed on named device.
4. Flip the mark in **this file**, sync `GAPS.md`, recompute earned points, bump `welvet_version` / CMake `VERSION`.
5. Repeat until the engine board is 100 and the volume trains end-to-end on Metal.

**Destination:** a complete standalone **3D volumetric** Welvet engine on Apple GPU — same model contract as Go Welvet — with resident state, native API, optional FFI/NAC, and w2a evidence for every ✅.
