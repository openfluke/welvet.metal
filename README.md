# welvet.metal — Welvet (3D volumetric AI engine) on Apple Metal

**GPU-only enforcement:** supported numerical forward/backward and training
paths run on Apple Metal. Host code handles scheduling, validation, allocation
and storage staging. Current support and per-mode evidence live in
[the execution contract](docs/METAL_ONLY.md) and
[the training checklist](docs/TRAINING_CHECKLIST.md); use those documents for
current per-mode evidence alongside the checklist below. Historical host
implementations do not establish GPU support.

`welvet.metal` and `welvet.cpp` are **separate native backends** for Go Welvet.
Neither backend is compiled into the other.

**welvet.metal is a full rebuild of Welvet — the 3D volumetric AI engine — in C++ / Apple Metal.** Same engine shape as Go Welvet: numerical types, layers, volumetric architecture (cells in a 3D grid), forward/backward, training modes, systems, model loading, inference, checkpoints. Execution is Metal compute + resident `MTLBuffer`s. Host orchestration is C++ (ObjC++ at the Metal boundary).

A `1 × 1 × 1` volume is the ordinary single-cell neural-network case (layers inside that cell). Larger volumes connect cells that each hold their own layer stacks. This Metal rebuild must own and execute that same topology — not “a Dense GEMV demo with a GPU sticker.”

It will run directly from C++ and through a stable C ABI. Go Welvet can later delegate via **NAC** (Native Acceleration Compute). The Metal engine must also build, load models, execute volumes, and train **without** a Go process.

| | |
|--|--|
| **Version** | **`0.48`** — working-tree milestone, 2026-10-26 |
| **Engine board** | **48 / 100** — supported FormatNone layer/training paths complete; broader engine gates remain |
| **Compute** | Apple GPU (Metal) — primary · Apple Neural Engine — separate gate |
| **Parity reference** | [welvet](https://github.com/openfluke/welvet) — local Go source used by w2a |
| **CPU sibling** | [welvet.cpp](https://github.com/openfluke/welvet.cpp) — separate backend; score is not inherited |
| **Validation** | [w2a](https://github.com/openfluke/w2a) `go_v_metal`: all 29 numbered modes, 23 layer families, supported native dtypes |

**This README is the runway.** Every unchecked cell is planned Metal work. Tick ⬜ → 🚧 → ✅ here as gates land. **Training port detail** (per-mode × layer evidence, wire names, version log) lives in [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md) — tick there as NormalBP / TrainModes land; this README keeps the mode summary table. [GAPS.md](GAPS.md) is the machine-oriented census mirror; [VERSIONING.md](VERSIONING.md) is the `0.x` formula.

### What exists today

```bash
./scripts/build_macos.sh
./build/example_metal_smoke          # device + shader compile + buffer RT
./build/welvet_dense_fwd --version   # 0.48
./build/welvet_dense_fwd --identity  # metal:<Apple GPU>…
./build/welvet_dense_fwd --gaps
./scripts/macos_build_w2a.sh --training-all
```

| Verified capability | Status |
|---|:---:|
| macOS Apple silicon shared/static builds, Metal device/library/buffers | ✅ |
| Metal forward/backward across all 23 supported layer families | ✅ |
| FormatNone native parameter storage and GPU SGD across all 34 dtypes | ✅ |
| NormalBP, all 29 numbered training modes, Grid/Line/Mesh scheduling | ✅ |
| KMeans outputs, seven Parallel combines, nested/polymorphic branches | ✅ |
| w2a parity, benchmark reset, GPU-only and rejected-host-API checks | ✅ |
| Persistent graph/tape residency and fused training | 🚧 |
| Packed quant formats, model/IO, remaining systems, Cameral/CamSync | ⬜ |

GDN is Float32-only. The 34-dtype sweeps validate native **parameter storage**
with the current Float32 tensor interface and documented software-promoted
arithmetic; they do not claim every activation type or packed quant format.
Parity aims for 100%, accepting the documented 60–100% spectrum. Numerical
training runs on Metal; host scheduling, allocation, copies and staging remain.

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
| [w2a](https://github.com/openfluke/w2a) | CPU-reference parity, determinism bands and timings in `go_v_metal` |

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
8. **Best-effort determinism** — aim for 100% (`exact_bit`) agreement with Go; the accepted project parity range is **60–100% on w2a's determinism spectrum**. Report the measured band and numerical deviations before ✅.

### Numerical parity target

Deterministic exactness against Go is **best effort, aiming for 100%, with an
accepted range of 60–100%**. These percentages are w2a spectrum scores:
`industry` = 60%, `strict` = 80%, and `exact_bit` = 100%. They are not model
accuracy or the percentage of tensor elements allowed to be wrong.

CPU and Metal GPU arithmetic can differ because of reduction order, rounding
and parallel execution. Those differences can accumulate in larger models and
longer training runs. Long-horizon training does not require bit-identical
weights or trajectories against Go: target industry-band agreement or better,
preserve the intended training semantics, and report the achieved band and
deviations. Pursue exactness where practical without making long-horizon bit
perfection a prerequisite for shipping.

Small regression fixtures may retain stronger exact checks. Their documented
gates record what has been verified; they do not impose universal exactness on
every model or training duration. The engine completion scoreboard below is
separate from this numerical parity spectrum.

---

## Scoreboard and credit rules

| Mark | Meaning | Credit |
|---|---|---:|
| ⬜ | Planned; no validated Metal implementation | 0% |
| 🚧 | Implemented subset with reproducible evidence + listed gaps | 50% of gate |
| ✅ | Declared acceptance gate passes (w2a + device identity) | 100% of gate |
| N/A | Outside Metal scope (reason required) | No credit |
| BLOCKED | Missing device / SDK / unresolved semantics | 0% |

**Version:** `welvet_version() = 0.{round-half-up(earned)}` until **100 → 1.0.0**. Only this engine board moves the tag. M0 host scaffolding does **not** earn engine points.

---

## Engine rebuild scoreboard — 100 points

Identical ten categories / weights as Go Welvet. Credit only from **this** Metal tree’s evidence.

| # | Category (Go board) | Wt | Metal today / remaining gate | Status | Earned |
|---:|---|---:|---|:---:|---:|
| 1 | **Foundation** — core, weights, quant, Metal runtime, tiling | 15 | Native FormatNone×34 and software-promoted GPU math; packed quant and broader tensor/tiling APIs remain | 🚧 | **7.5** |
| 2 | **Dense MatVec** — FormatNone×34 + quants; F/B/update | 15 | FormatNone F/B/native GPU SGD verified; packed quant kernels remain | 🚧 | **7.5** |
| 3 | **Transformer stack** — MHA, SwiGLU, norms, Softmax, Emb, Seq/Res, seqmix | 14 | Supported layer baselines verified; cross-attention, broader variants and ConvMix remain | 🚧 | **7** |
| 4 | **CNN / RNN / LSTM** | 6 | Supported forward/BPTT/SGD verified; broader convolution/state acceptance matrix remains | 🚧 | **3** |
| 5 | **Extended layers** — GDN, ConvT1–3, Mamba, KMeans, Parallel, Metacognition | 7 | Declared baseline F/B/training contracts verified; GDN Float32, ordinary Parallel combines | ✅ | **7** |
| 6 | **Runtime + architecture** — Grid, forward, backward, training, Line/Mesh | 8 | All-family numerical execution and schedules verified; persistent resident graph/tapes remain | 🚧 | **4** |
| 7 | **Systems** — dna, evolution, tween, tanhi, telemetry | 5 | Tween/local-credit subgate complete; other four subgates remain | 🚧 | **1** |
| 8 | **Model / IO** — tokenizer, entity, transformer, sampling, hf | 8 | Model construction/loading/checkpoint/generation pipeline remains | ⬜ | **0** |
| 9 | **Training credit** — 29 modes + Cameral / CamSync | 8 | All 29 numbered modes verified; advanced modifiers and Cameral/CamSync remain | 🚧 | **4** |
| 10 | **Peak fused / no host ALU** | 14 | GPU-only numerical execution enforced; full residency, fusion and decoder throughput remain | 🚧 | **7** |
| | **Total → version** | **100** | **48 points** | **🚧** | **`0.48`** |

**Scoring audit (2026-10-26):** partial categories receive the documented 50%
credit; the systems category uses its five explicit one-point subgates below
(Tween = 1, others = 0). A layer ✅ covers its declared supported baseline, not
all future variants. The engine score is separate from the 60–100% numerical
parity spectrum. All 29 training modes being complete does not make the whole
engine 1.0.0. No release tag or sibling-repository credit is implied.

**Path to 1.0:** same order Go used / welvet.cpp is following — Dense FormatNone + NormalBP → layers → Step/Mesh/Tween* → Systems + Model/IO → CamSync + peak fused Metal. welvet.cpp’s ✅ list is the **port order**, not transferred credit.

---

## Full engine subsystem runway

### Foundation, weights and Metal compute — §1 / §2

| Go reference | Metal rebuild target | Status |
|---|---|:---:|
| `core/` | Tensor representation, all 34 dtypes, activations, layer IDs, execution config | 🚧 native dtypes, Float32 tensor interface and GPU activations; broader tensor APIs open |
| `weights/` | Native storage, Metal projection/backward/SGD; full persistent graph residency remains | 🚧 |
| `quant/` | FormatNone×34 verified; 20 additional packed formats remain unsupported | 🚧 |
| `tiling/` | Basic dispatch/tail guards implemented; workspace budgets and tuned policies remain | 🚧 |
| `simd/` | N/A as Plan 9 — replaced by Metal kernels (do not ship host AVX as “Metal”) | N/A |
| `webgpu/` | Reference for device semantics only — implement as Metal, not WGSL | N/A |
| `fusedgpu/` | Decoder/op fusion with resident intermediates + execution provenance on MTL | ⬜ |
| Metal runtime | Device, embedded shaders, pipelines, command dispatch, sync and error handling | ✅ |

### Architecture and runtime — §6

| Go reference | Metal rebuild target | Status |
|---|---|:---:|
| `architecture/` | Own the 3D volume, coordinates, cell layers, graph links, topology validation | ✅ host-owned topology with Metal numerical execution |
| `runtime/dispatch/` | Dispatch supported layer baselines; unsupported APIs fail explicitly | ✅ |
| `runtime/forward/` | Supported volume execution without Go orchestration; model loading is separate | ✅ |
| `runtime/backward/` | Metal reverse traversal/gradient joins verified; persistent tapes/shared-store planning remain | 🚧 |
| `runtime/training/` | GPU loss/credit/native SGD; all 29 numbered modes | ✅ |
| `runtime/step/` | Line ticks, warmup/pops, frozen credit and Train/Action scheduling verified | ✅ |
| Mesh / volumetric scheduling | All-family clock/volume credit verified; tapes still stage through host buffers | 🚧 |

### Systems — engine §7

| Go reference | Wt | Metal rebuild target | Status | Earned |
|---|---:|---|:---:|---:|
| `systems/dna/` | 1 | Encode/decode topology + params; native dtype preserved | ⬜ | 0 |
| `systems/evolution/` | 1 | DNA splice, mutation, cloning, NEAT-compatible behavior | ⬜ | 0 |
| `systems/tween/` | 1 | Tween/Chain/Split/Alt, proxy/cache/sparse local credit on Metal | ✅ | 1 |
| `systems/tanhi/` | 1 | TANHI structural/activation reporting + telemetry transport | ⬜ | 0 |
| `systems/telemetry/` | 1 | Blueprint, counts, estimates, execution/learning observability | ⬜ | 0 |
| **Subtotal** | **5** | | **🚧** | **1 / 5** |

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

FormatNone native parameter storage and GPU F/B/SGD are verified across all
34 dtypes under the current Float32 tensor interface. Explicit staging/codecs
and software-promoted arithmetic preserve the declared Go contract. This table
does not claim generic complex-valued tensor arithmetic or hardware Float64.
GDN remains Float32-only; per-layer evidence is in the training checklist.

| # | DType | Pack/RT | Metal F | Metal B | Metal update | Notes |
|--:|-------|:------:|:------:|:------:|:------:|-------|
| 0 | Float64 | ✅ | ✅ | ✅ | ✅ | Native encoding; current real Float32 tensor bridge |
| 1 | Float32 | ✅ | ✅ | ✅ | ✅ | Float32 tensor interface; native parameter dtype |
| 2 | Float16 | ✅ | ✅ | ✅ | ✅ | Native parameter encoding; Float32 tensor interface |
| 3 | BFloat16 | ✅ | ✅ | ✅ | ✅ | |
| 4 | FP8E4M3 | ✅ | ✅ | ✅ | ✅ | |
| 5 | FP8E5M2 | ✅ | ✅ | ✅ | ✅ | |
| 6 | Int64 | ✅ | ✅ | ✅ | ✅ | |
| 7 | Int32 | ✅ | ✅ | ✅ | ✅ | |
| 8 | Int16 | ✅ | ✅ | ✅ | ✅ | |
| 9 | Int8 | ✅ | ✅ | ✅ | ✅ | |
| 10 | Uint64 | ✅ | ✅ | ✅ | ✅ | |
| 11 | Uint32 | ✅ | ✅ | ✅ | ✅ | |
| 12 | Uint16 | ✅ | ✅ | ✅ | ✅ | |
| 13 | Uint8 | ✅ | ✅ | ✅ | ✅ | |
| 14 | Int4 | ✅ | ✅ | ✅ | ✅ | |
| 15 | Uint4 | ✅ | ✅ | ✅ | ✅ | |
| 16 | FP4 | ✅ | ✅ | ✅ | ✅ | |
| 17 | Int2 | ✅ | ✅ | ✅ | ✅ | |
| 18 | Uint2 | ✅ | ✅ | ✅ | ✅ | |
| 19 | Ternary | ✅ | ✅ | ✅ | ✅ | |
| 20 | Binary | ✅ | ✅ | ✅ | ✅ | |
| 21 | Int | ✅ | ✅ | ✅ | ✅ | ABI width explicit |
| 22 | Uint | ✅ | ✅ | ✅ | ✅ | |
| 23 | Uintptr | ✅ | ✅ | ✅ | ✅ | never serialize as tensor id |
| 24 | Complex64 | ✅ | ✅ | ✅ | ✅ | Native encoding; current real Float32 tensor bridge |
| 25 | Complex128 | ✅ | ✅ | ✅ | ✅ | Native encoding; current real Float32 tensor bridge |
| 26 | NF4 | ✅ | ✅ | ✅ | ✅ | |
| 27 | FP6 | ✅ | ✅ | ✅ | ✅ | |
| 28 | Int6 | ✅ | ✅ | ✅ | ✅ | |
| 29 | Uint6 | ✅ | ✅ | ✅ | ✅ | |
| 30 | Int5 | ✅ | ✅ | ✅ | ✅ | |
| 31 | Uint5 | ✅ | ✅ | ✅ | ✅ | |
| 32 | Int3 | ✅ | ✅ | ✅ | ✅ | |
| 33 | Uint3 | ✅ | ✅ | ✅ | ✅ | |

### Numerical semantics gates

| Concern | Status |
|---|:---:|
| Integer/low-bit learning rules match Go | ✅ |
| Binary/ternary bit order, codes, masks | ✅ |
| Complex native encoding / real-valued Float32 bridge; broader complex math open | 🚧 |
| FP8/FP6/FP4/NF4 exact code tables | ✅ |
| Explicit software precision and parity policy; full execution-plan policy open | 🚧 |
| Declared Float32 staging/scratch; generic conversion/ownership API open | 🚧 |
| Unsupported arithmetic → hard error (no silent widen) | ✅ |

---

## Quant formats (`quant.Format`) — 20

Packed layouts. Row **None** = FormatNone + DType (not “FP32 master”).

| Format | Metal pack+MatVec | Metal Dense F | Metal Dense B/update |
|--------|:-----------------:|:-------------:|:--------------------:|
| None | ✅ weights + DType×34 | ✅ | ✅ |
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

`dX` = input-gradient contract; `dW/db` = every trainable parameter. Metal F/B = GPU numerical execution with host scheduling/staging. Checkmarks
cover supported baselines in all 29 modes (GDN Float32-only); broader variant
acceptance below remains separate.

| ID | Layer | Forward target | Backward target | Metal F | Metal B |
|---:|---|---|---|:---:|:---:|
| 0 | **Dense** | Batched projection, bias, activation | dX, dW, act'; ApplyGradSGD | ✅ | ✅ |
| 1 | **MultiHeadAttention** | Self-attention Q/K/V/O, masks, RoPE | Proj + attn + RoPE grads (ModeSelf) | ✅ | ✅ |
| 2 | **SwiGLU** | Gate/up/down, SiLU product | All projs + product/SiLU′ | ✅ | ✅ |
| 3 | **RMSNorm** | RMS + scale | dX + scale grad | ✅ | ✅ |
| 4 | **CNN1** | 1D conv | dX, kernel grads | ✅ | ✅ |
| 5 | **CNN2** | 2D conv | dX, kernel grads | ✅ | ✅ |
| 6 | **CNN3** | 3D conv | dX, kernel grads | ✅ | ✅ |
| 7 | **RNN** | Recurrent + state | BPTT + state grads | ✅ | ✅ |
| 8 | **LSTM** | Gates + h/c | Gate/param/h/c grads | ✅ | ✅ |
| 9 | **LayerNorm** | Mean/var, scale, bias | dX, scale/bias | ✅ | ✅ |
| 10 | **ConvTransposed1D** | 1D transposed | dX, kernel | ✅ | ✅ |
| 11 | **ConvTransposed2D** | 2D transposed | dX, kernel | ✅ | ✅ |
| 12 | **ConvTransposed3D** | 3D transposed | dX, kernel | ✅ | ✅ |
| 13 | **Embedding** | Gather | Scatter/reduction; defined token dX | ✅ | ✅ |
| 14 | **KMeans** | Soft assignment + centroids | Declared differentiable paths | ✅ | ✅ |
| 15 | **Softmax** | Standard/grid/temp/variants | JVPs (dX; weightless) | ✅ | ✅ |
| 16 | **Parallel** | Branches, routing, seven combines | Ordinary branch/gate credit; Cameral separate | ✅ | ✅ |
| 17 | **Sequential** | Ordered child chain | Reverse child credit | ✅ | ✅ |
| 18 | **Residual** | F(x) + skip | F reverse + skip grad | ✅ | ✅ |
| 19 | **Metacognition** | Observed compute + rules | lastGate-scale + Observed bwd | ✅ | ✅ |
| 20 | **Mamba** | Selective SSM scan | Declared BPTT + projs | ✅ | ✅ |
| 21 | **GDN** | Gated DeltaNet | Truncated-BPTT contract (f32-first) | ✅ | ✅ |
| 22 | **Stack** | Heterogeneous nested chain | Nested child credit | ✅ | ✅ |
| helper | **seqmix** | Kind/Contract metadata and MHA/Mamba/GDN tags | Concrete math in those layers; ConvMix open | 🚧 | ⬜ |

### Layer acceptance details

| Family | Required before ✅ | Status |
|---|---|:---:|
| Dense | Transpose, batch, tails, strides, bias, act, dX-only, dW-only, update-only, dtype boundaries | 🚧 supported baseline verified; broader matrix open |
| Attention | Causal/cross/GQA; masks; ALiBi; sliding; dropout/RNG; KV invalidation | 🚧 supported baseline verified; broader matrix open |
| Norms | Epsilon, axes, small variance, saved stats, param grads | 🚧 supported baseline verified; broader matrix open |
| Softmax | Stable reductions; masks; temperature; Sparsemax/Entmax/Gumbel where present | 🚧 supported baseline verified; broader matrix open |
| Activations | Linear, ReLU, SiLU, GELU, Tanh, Sigmoid, LeakyReLU, ReLU2 + derivatives | ✅ |
| Convolution | Non-square/cubic, dilation, groups, overlap accum, transpose shapes | 🚧 supported baseline verified; broader matrix open |
| Recurrence/SSM | Initial/final state, sequence bounds, reset, tape, truncation | 🚧 supported baseline verified; broader matrix open |
| Composition | Nested non-Dense; shared stores; branch order; no duplicate updates | 🚧 supported baseline verified; broader matrix open |
| Discrete/index/rule | Explicit differentiability matching Go | ✅ |

### Dense first slice (M1)

| # | Case | Status |
|---|---|:---:|
| 1 | Metal soft-double GEMV (Apple has **no HW float64**) | ✅ |
| 2 | Bias + activation on Metal (`welvet_dense_bias_act`) | ✅ |
| 3 | Leaf forward exact_bit vs Go Float32 | ✅ |
| 4 | Leaf forward ×34 FormatNone exact_bit vs Go | ✅ `go_v_metal` |
| 5 | 3-way speed Go / C++ / Metal + pick winner | ✅ (tiny shapes: C++ wins; Metal ~dispatch-bound) |
| 6 | **No host MatVec / no f32-acc substitute** | ✅ |
| 7 | NormalBP dX/dW/db + GPU SGD; per-kernel buffers, staged tapes | ✅ |
| 8 | NormalBP ×34 + volume `1×1×1` / `2×2×2` | ✅ |

---

## Kernel and training runway

### Shared compute primitives

These are kernels used by supported layer/training paths, not a claim of a
complete standalone tensor-operation API or fully fused execution.

| Primitive | Forward/use | Backward/update | Status |
|---|---|---|:---:|
| GEMV/GEMM | Dense + shared projections | GEMVᵀ, dX, outer/dW, db | ✅ supported layer paths |
| Reduction | Sum, mean, var, max, norm, LSE | Reduction grads + broadcast | ✅ supported layer paths |
| Elementwise | Bias, mul, add, acts, gates | Derivatives / JVPs | ✅ supported layer paths |
| Layout/shape | View, transpose, reshape, concat, split | Inverse mapping | ✅ supported layer paths |
| Gather/scatter | Embedding, indexing, routing | Duplicate-index policy | ✅ supported layer paths |
| Convolution | im2col or native Metal conv | Overlap accum + filter grads | ✅ supported layer paths |
| Attention/position | QK, scale, mask, AV, RoPE | Attn + positional grads | ✅ supported layer paths |
| Scan/recurrence | RNN/LSTM/Mamba/GDN | Declared BPTT/truncation | ✅ supported layer paths |
| Optimizer/update | Storage-preserving SGD first | Masks, LR, rounding | ✅ supported layer paths |

---

## Training modes — all 29 + cameral

**All 29 numbered modes are signed off** for their supported FormatNone
dtypes (GDN Float32 only), using the [60–100% parity target](#numerical-parity-target).
Run the complete w2a matrix:

```sh
./scripts/macos_build_w2a.sh --training-all
```

Canonical equations: [Welvet `training_modes.md`](https://github.com/openfluke/welvet/blob/main/training_modes.md).  
Tick **Inside node** (`1×1×1`) and **3D volumetric** separately. **Signed** = all supported FormatNone dtypes for that layer under the mode.

| # | TrainMode | Family | Inside node | 3D volumetric | Signed | Ver |
|---|---|---|:---:|:---:|:---:|---|
| 1 | NormalBP | BP | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 2 | StepBP | BP | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 3 | MeshBP | BP | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 4 | Tween | Tween | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 5 | StepTween | Tween | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 6 | MeshTween | Tween | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 7 | TweenChain | TweenChain | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 8 | StepTweenChain | TweenChain | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 9 | MeshTweenChain | TweenChain | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 10 | TweenSplit | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 11 | StepTweenSplit | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 12 | MeshTweenSplit | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 13 | TweenAlt | TweenAlt | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 14 | StepTweenAlt | TweenAlt | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 15 | MeshTweenAlt | TweenAlt | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 16 | TweenSplitHeadProxy | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 17 | TweenSplitLinear | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 18 | TweenSplitFastProxy | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 19 | TweenSplitLinearCache | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 20 | TweenSplitHeadProxyAsync | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 21 | TweenSplitSparse | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 22 | MeshTweenSplitFastProxy | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 23 | MeshTweenSplitSparse | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 24 | StepTweenSplitHeadProxy | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 25 | StepTweenSplitLinear | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 26 | StepTweenSplitFastProxy | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 27 | StepTweenSplitLinearCache | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 28 | StepTweenSplitHeadProxyAsync | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| 29 | StepTweenSplitSparse | TweenSplit | ✅ | ✅ | ✅ | 2026-10-26 (unreleased) |
| — | Freeze | Freeze | ⬜ | ⬜ | ⬜ | |
| — | Shadow | Freeze | ⬜ | ⬜ | ⬜ | |
| — | Adversarial | BP-gated | ⬜ | ⬜ | ⬜ | |
| — | Memory | BP-gated | ⬜ | ⬜ | ⬜ | |
| — | CamSync / BlendStores | — | ⬜ | ⬜ | ⬜ | |
| — | Cameral BranchModes (Parallel) | — | ⬜ | ⬜ | ⬜ | |

Per-layer evidence tables (leaf ×34, volume shapes, speed, wires): **[docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md)**. Mode summary table above stays the high-level board; tick detailed rows in the checklist.

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
| Single cell | Layer stack in `1×1×1` on Metal with Go parity | ✅ supported fixtures |
| Multi-cell volume | Coords, ordering, links, boundary tensors intact | ✅ supported fixtures |
| Non-cubic volumes | No hidden cubic-only assumption | ✅ supported fixtures |
| Per-layer placement | Cell/layer/child → queryable Metal path | ⬜ |
| Partition fusion | Fuse consecutive ops without changing visible semantics | ⬜ |
| Heterogeneous forward | Explicit host↔Metal handoff | ⬜ |
| Heterogeneous backward | Gradient edges + tape ownership across boundary | ⬜ |
| Step systolic | Fill/drain, ticks, Line/Stack, RoleTrain/RoleAction | ✅ supported fixtures |
| Mesh execution | Dependency order, joins, credit over links | ✅ supported fixtures |
| Placement cost | Upload/download, queue latency, compile, memory pressure | ⬜ |
| Plan invalidation | Rebuild on shape/topology/weights/device/precision/mode change | ⬜ |

**Placement roadmap:** support query, cost planning and cross-backend handoffs
remain open. This backend always uses Metal for supported numerical execution;
future Go-level backend selection may choose a separate CPU backend for small
workloads. Persistent graph/tape residency and fusion remain separate gates.

---

## Metal platform runway

| Gate | Status |
|---|:---:|
| macOS Apple silicon build | ✅ |
| Device enumerate + identity string | ✅ |
| Runtime `.metal` compile | ✅ |
| Shared `MTLBuffer` RT | ✅ |
| Smoke compute PSO | ✅ |
| Command buffer dispatch helper (encode/commit/wait/errors) | ✅ |
| Precompiled `.metallib` release path | ⬜ |
| Best-effort parity bands and arithmetic/staging contract; tuned planning open | ✅ documented current path |
| iOS device framework / static artifact | ⬜ |
| iOS simulator slice (no device accel implied) | ⬜ |
| Apple Neural Engine / Core ML (separate — never implied by Metal ✅) | ⬜ |

| Platform | Artifact | Status |
|---|---|:---:|
| macOS arm64 | Shared/static builds, headers, embedded shaders and wire tool | ✅ local builds |
| macOS x86_64 | Explicitly bounded or dropped | ⬜ |
| iOS arm64 | Static/framework | ⬜ |

---

## FFI and C ABI

| Surface | Status |
|---|:---:|
| ABI negotiation / version | 🚧 version `0.48`; general descriptor negotiation open |
| Metal init / available / identity | ✅ |
| Gaps census counters | ✅ |
| Tensor / weight / layer descriptors | ⬜ |
| Engine/model handles (volume create/load/train/checkpoint) | ⬜ |
| Dense forward / NormalBP wire magics (w2a) | ✅ w2a wire tool |
| All layer NormalBP magics (mirror welvet.cpp names) | ✅ w2a wire tool |
| Volume / Step / Mesh / Tween* magics | ✅ w2a wire tool |
| Go NAC dispatcher selects Metal | ⬜ |

Boundary rules match welvet.cpp: versioned descriptors, opaque handles, no C++ exceptions through C, explicit ownership, async inputs owned/copied, provenance in traces.

---

## Resident weights and memory

| Capability | Status |
|---|:---:|
| Authoritative owner (host vs Metal generation) | 🚧 FormatNone host + soft-double GPU mirror |
| Upload once / reuse across fwd/bwd/update | ✅ Dense resident W (reload after SGD) |
| Preserve storage truth (no undeclared f32 master) | ✅ FormatNone WeightStore |
| Derived device layout cache + invalidation | ✅ invalidate_gpu after ApplyGradSGD |
| Resident activations / tapes across adjacent layers | ⬜ |
| Resident gradients + clear policy | 🚧 GPU arithmetic; exports/staging between kernels |
| Resident update in declared storage semantics | 🚧 native GPU SGD/encoding; exports and mirror refresh remain |
| Host staging/export boundary | ✅ copies/exports only; numerical updates stay on GPU |
| CamSync on resident stores | ⬜ |
| Checkpoint/export for ENTITY flows | ⬜ |
| Migration before device/plan change | ⬜ |
| Memory budget accounting | ⬜ |
| Unified/shared memory only under explicit sync contract | ⬜ |
| Concurrency / device-loss honesty | ⬜ |

---

## Validation and performance (w2a)

Apply the [best-effort numerical parity target](#numerical-parity-target): aim
for 100%, accept the 60–100% spectrum range, and record actual error metrics and
bands. Evaluate performance separately from numerical agreement.

| Gate | Status |
|---|:---:|
| Pin Welvet + welvet.cpp revisions in fixtures | ⬜ |
| `Go_v_Metal` harness (parity + speed) | ✅ supported contract; see training evidence |
| Optional C++↔Metal cross-check on same Mac | ⬜ |
| Standalone construct→volume→train→save/reload | ⬜ |
| Numerical / forward / backward / train / volume / modes matrices | ✅ supported contract; see training evidence |
| Capability honesty (unsupported fails clearly) | ✅ supported contract; see training evidence |
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
| **M2** — Dense NormalBP | Multi-step native-dtype train parity; full tape residency separate | ✅ |
| **M3** — Dtype / quant breadth | ×34 verified; packed quant kernels open | 🚧 |
| **M4** — Layer breadth | All 23 supported baselines; broader variants open | 🚧 |
| **M5** — Full volumetric runtime | Grid/Line/Mesh and all 29 modes verified; resident graph/tapes open | 🚧 |
| **M6** — Systems + Model/IO + Lucy | Tween complete; other systems/model/IO remain | 🚧 |
| **M7** — Peak fused Metal | GPU-only numerical execution enforced; peak fusion remains | 🚧 |
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
src/systems/                 Tween/Chain/Split/Alt and proxy credit (other systems open)
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

1. Pick an open gate: packed quant kernels, broader variants, resident graph/tapes, systems or model/IO.
2. Implement Metal kernels + host wiring; preserve Go / welvet.cpp semantics.
3. Prove in w2a (`Go_v_Metal`): determinism + speed on named device.
4. Flip the mark in **this file**, sync `GAPS.md`, recompute earned points, bump `welvet_version` / CMake `VERSION`.
5. Repeat until the engine board is 100 and the volume trains end-to-end on Metal.

**Destination:** a complete standalone **3D volumetric** Welvet engine on Apple GPU — same model contract as Go Welvet — with resident state, native API, optional FFI/NAC, and w2a evidence for every ✅.
