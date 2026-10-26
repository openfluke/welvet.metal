# Training checklist — welvet.metal

One **TrainMode** at a time. Finish that mode’s table (or the rows you care about), bump version, move on.

Canonical enum: Go `layers/parallel/train_mode.go` (`TrainMode`).  
`inherit` (0) = resolve-only — not a ship target.

**Port mirror of** [`welvet.cpp/docs/TRAINING_CHECKLIST.md`](../../welvet.cpp/docs/TRAINING_CHECKLIST.md). Same cases / wires / column meanings; status here is **Metal** only. cpp `[x]` does **not** transfer credit.

---

## Metal-only mandate (non-negotiable)

**welvet.metal = GPU Metal compute only.** Host/CPU math belongs in **welvet.cpp** (and Go).  
Host numerical training loops are forbidden. Host scheduling, validation, allocation and storage staging are permitted; residency and performance work remain separate gates.

Current executable contract: [METAL_ONLY.md](METAL_ONLY.md).

| Path | Execution today |
|---|---|
| NormalBP layers | Dense, RMSNorm, LayerNorm, SwiGLU, Softmax, Embedding, MHA, CNN1/2/3, ConvT1/2/3, RNN, LSTM, Sequential, Residual, Mamba, Metacognition, GDN and Stack use Metal numerical kernels |
| MSE / MSEGrad | Metal kernels; scalar loss is exported in IEEE float64 representation |
| SGD and native storage updates | Metal arithmetic and native FormatNone encoding across all 34 dtypes; bias updates run on Metal too |
| GDN | Metal Float32 truncated-BPTT; state/conv history stop-gradient |
| Tween ModeTween | Metal block-average projection and local credit at lr×0.5; all 21 supported families / dtypes |
| TweenChain ModeTweenChain | Metal full-lr chain-rule BP in leaves/volumes; all 21 supported families / dtypes |
| Split / Alt and requested Step/Mesh variants | GPU 1/N scaling, refreshed global Alt phases, frozen Line per-hop credit, last-tick MeshTween and BPTT MeshTweenChain |
| Scheduling and data movement | Host validation, scheduling, allocation, copies and storage staging remain |
| Mesh gradient joins | Metal accumulation kernel; Dense mesh-clock ×34, disabled-cell and remote-link checks |
| KMeans / Parallel NormalBP | Open; no CPU numerical fallback |
| Performance | Correctness gates passed; full residency, dispatch reduction and throughput tuning remain open |

**Validated 2026-10-26:** all **17 NormalBP evidence sections / 128 case checks**
passed in `w2a` under `TestGoVMetalNormalBP`, including all dtype loops and leaf,
chain and volumetric fixtures. `TestGoVMetalOnly` also passed its primitive,
activation/derivative, native update, cache-coherence, rejected-host-API and 13
Float32 NormalBP/Line/Mesh smoke cases. Shared and static builds passed.
Tween ModeTween also passes 80 case groups covering supported leaf and volume
dtypes, plus GPU projection/nested-credit primitive checks. TweenChain adds 80
case groups covering supported leaf/volume dtypes and true chain-rule credit.
The eight requested Split/Alt/Line/Mesh sections add 576 case groups, plus hand-calculated credit and tiny-SGD primitive checks. These are working-tree
validations; no new release tag is claimed.

**Precision policy:** exact tensor bits where required by each case; finite loss
uses `abs(go-metal) <= 1e-12 * max(1, abs(go))`. NormalBP weight comparisons allow
FP32 subnormal flushing only when both values are below the smallest normal
FP32 value; normal values still require exact bits. MHA leaf ×34 retains its
existing ≥strict gradient floor. Softmax retains ≥loose (with stricter rows as
shown), and GDN retains ≥loose leaf / ≥industry volume. Stack's existing binary
211 post allowance is `max|Δ| <= 1e-18`, with its other checks retained.

Checkmarks below indicate validated correctness under these bands, not a speed
win or completion of GPU residency/performance work. Unsupported and unverified
modes remain unchecked.

---

### Columns (every table)

| Column | Means |
|---|---|
| **Inside node** | Layer trains correctly in a `1×1×1` cell (single-node / ordinary NN case). go_v_metal / exact_bit (or declared band). |
| **3D volumetric** | Same mode works when that layer sits in a multi-cell Grid (volume Forward/Backward + credit across cells). |
| **Fully signed off** | **All** supported FormatNone dtypes for that layer under this mode (not just f32). Softmax = dX only. GDN = f32-only in Go → sign off when f32 path is done. |

Tick `[x]` when validated. **Ver** records a release tag when one exists, otherwise a dated unreleased validation. **Fully signed off** requires every supported dtype, not only Float32 smoke coverage.

**Current engine:** run `./scripts/macos_build_w2a.sh --normalbp-all` to build
Metal and run both `TestGoVMetalNormalBP` and `TestGoVMetalOnly` in sibling `w2a`.
The combined runner defaults to one measured benchmark iteration per fixture;
set `WELVET_METAL_BENCH_ITERS=2000` for longer performance measurements.

---

## w2a Go↔Metal (Dense NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --dense-train`  

Wire: `WDB1`/`WDBO` (leaf), `WVN1`/`WVNO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: warm-up + `--bench=N` (in-process; no spawn) → **goµs / mtlµs / spd%Δ** (`(mtl−go)/go×100`, neg ⇒ Metal faster). Resident soft-double W (no remount each op); microbench spd%Δ is GPU-launch-bound vs Go host.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 ReLU (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔Metal (RMSNorm NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --rmsnorm-train`  
Wire: `WRB1`/`WRBO` (leaf), `WRV1`/`WRVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention as Dense (leaf mount-once + γ reset; no remount each iter).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγ/γ/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔Metal (LayerNorm NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --layernorm-train`  
Wire: `WLB1`/`WLBO` (leaf), `WLV1`/`WLVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + γ/β reset).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγβ/γβ/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔Metal (SwiGLU NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --swiglu-train`  
Wire: `WSB1`/`WSBO` (leaf), `WSV1`/`WSVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔Metal (Softmax NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --softmax-train`  
Wire: `WXB1`/`WXBO` (leaf), `WXV1`/`WXVO` (volume). Softmax = dX only (weightless).  
Timing: leaf mount-once (reuse layer; no W). Band: **≥loose** floor (Softmax hard to bit-match); leaf×34 / volume hit **strict (~80%)**; Kind×34 worst **loose**; Gumbel = smoke.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Standard Float32 (dX/loss) | [x] | strict + speed |
| 2 | Leaf all Kind Float32 (+ gumbel smoke) | [x] | ≥loose; gumbel non-det |
| 3 | Leaf Standard × all 34 dtypes | [x] | strict ×34 + speed |
| 4 | Leaf all Kind × all 34 dtypes | [x] | ≥loose (worst=loose) |
| 5 | Volume 1×1×1×1 Float32 | [x] | inside-node strict |
| 6 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer strict |
| 7 | Volume 1×1×1×1 × all 34 dtypes | [x] | strict ×34 |
| 8 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric strict |
| 9 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell strict ×34 |

## w2a Go↔Metal (Embedding NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --embedding-train`  
Wire: `WEB1`/`WEBO` (leaf), `WEV1`/`WEVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + W reset). dX zeros (discrete IDs); dW scatter.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + spectrum |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 |

## w2a Go↔Metal (MHA NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --mha-train`  
Wire: `WHB1`/`WHBO` (leaf), `WHV1`/`WHVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (resident Dense Q/K/V/O; microbench spd%Δ is launch-bound). ModeSelf baseline (no dropout/QKNorm/cross).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf DecoderCausal Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf GQA + Encoder Float32 | [x] | exact_bit |
| 3 | Leaf DecoderCausal × all 34 dtypes | [x] | W exact_bit; dX/dW ≥strict (uint2 dW ~1 ULP) |
| 4 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 5 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 6 | Volume 1×1×1×1 × all 34 dtypes | [x] | + spectrum |
| 7 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed; subnormal weight differences allowed |
| 8 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 |

## w2a Go↔Metal (CNN1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --cnn-train`  
Wire: `C1B1`/`C1BO`, `C2B1`/`C2BO`, `C3B1`/`C3BO` (leaf); `C*V1`/`C*VO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + proj W reset). Metal convolution gather forward + input/filter gradient kernels.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 CNN1/2/3 | [x] | exact_bit + speed |
| 4–6 | Leaf ×34 CNN1/2/3 | [x] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Metal (ConvT1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --convt-train`  
Wire: `T1B1`/`T1BO`, `T2B1`/`T2BO`, `T3B1`/`T3BO` (leaf); `T*V1`/`T*VO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + proj W reset). Scatter forward + act Jacobian + scatter adjoint (no bias).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 ConvT1/2/3 | [x] | exact_bit + speed |
| 4–6 | Leaf ×34 ConvT1/2/3 | [x] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Metal (RNN NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --rnn-train`  
Wire: `RNB1`/`RNBO` (leaf); `RNV1`/`RNVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. BPTT via Dense IH/HH + tanh′; loom `[dIH|dHH|dBias]`.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Metal (LSTM NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --lstm-train`  
Wire: `LSB1`/`LSBO` (leaf); `LSV1`/`LSVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. BPTT 4-gate Dense; loom `[i|f|g|o]` packs.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Metal (Sequential NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --sequential-train`  
Wire: `SQB1`/`SQBO` (leaf); `SQV1`/`SQVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. Reverse Dense Dim→Dim chain; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving Dim |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Metal (Residual NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --residual-train`  
Wire: `REB1`/`REBO` (leaf); `REV1`/`REVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. F reverse Dense + skip grad; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | y=F(x)+x |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Metal (Mamba NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --mamba-train`  
Wire: `MBB1`/`MBBO` (leaf); `MBV1`/`MBVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. Scan BPTT + In/Out Dense; loom In|Out|ALog|DSkip.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving D |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Metal (Metacognition NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --metacognition-train`  
Wire: `MCB1`/`MCBO` (leaf); `MCV1`/`MCVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. lastGate-scale + Observed Dense; loom [Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | default cool rules |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔Metal (GDN NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --gdn-train`  
Wire: `GDB1`/`GDBO` (leaf); `GDV1`/`GDVO` (volume). ≥loose leaf / ≥industry volume vs Go (Metal f32 scan; declared GPU precision floors).
Timing: same `--bench=N` / spd%Δ convention. Truncated-BPTT (state/conv stop-grad); f32-only (no ×34).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | ≥loose (dW floor) + speed |
| 2 | Volume 1×1×1×1 | [x] | ≥industry |
| 3 | Volume 1×1×1×2 chain | [x] | ≥industry |
| 4 | Volume 2×2×2×1 | [x] | ≥industry; scaled init |

## w2a Go↔Metal (Stack NestedStack-of-Dense NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --stack-train`  
Wire: `STKB`/`STKO` (leaf); `STKV`/`STVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. NestedStack of Dense Dim→Dim Linear; loom n_children×[Dim×Dim]. Avoids `STB1` (StepBP).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | NestedStack Dense |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | binary 211 post denorm ≤1e-18 OK |

---

## w2a Go↔Metal (StepBP Line + MeshBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --stepbp-train`  
Wires: `STB1`/`STBO` (Line FamilyBP); `MBP1`/`MBPO` (Dense mesh-clock FamilyBP).
`TestGoVMetalStepBPTrain` contains **84 case groups**, including the dtype loops below.
The runner also executes `TestGoVMetalOnly` and defaults to one measured iteration.
Every forward, backward, loss, gradient join and SGD calculation uses Metal kernels;
host code schedules ticks, stores tapes and stages buffers.

| Case group | Status | Evidence |
|---|:---:|---|
| Dense multi-hop Line Float32 | [x] | Per-tick tapes and FamilyBP on pop; exact weights, finite loss band |
| Dense Line ×34 | [x] | All supported FormatNone dtypes |
| RMSNorm / LayerNorm Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| Softmax Line Float32 + ×34 | [x] | Weightless loss check uses the finite loss band |
| SwiGLU / Sequential / Residual Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| Mamba / Metacognition Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| GDN Line Float32 | [x] | ≥strict weights; finite loss band; GPU float2 reductions/transcendentals |
| NestedStack Line Float32 + ×34 | [x] | Recursive flattening and child updates |
| Embedding / MHA Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| CNN1 / ConvT1 Line Float32 + ×34 | [x] | Explicit Line dispatch |
| CNN2/3 / ConvT2/3 Line ×34 | [x] | Explicit rank-4/rank-5 inputs, including Float32 |
| RNN / LSTM Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| Dense mesh-clock 111 / 211 / 222 Float32 | [x] | Double-buffer forward ticks, reverse-time credit and SGD |
| Dense mesh-clock 111 / 222 ×34 | [x] | Repeated ticks and native storage updates |
| Dense mesh-clock disabled cell + remote link | [x] | Disabled passthrough and remote gradient routing |
| Volume 222 Float32: all 21 supported families | [x] | Shared volume Forward+Step path used by Go ModeMeshBP |
| Volume 211 ×34: all 20 dtype-generic families | [x] | GDN is Float32-only; retains its volume ≥industry floor |
| Benchmark fixture reset | [x] | Line dtype sweeps and GDN compare benchmark results with fresh-run results; Dense mesh dtype/routing fixtures do likewise |

**Scope:** volume rows deliberately reuse the NormalBP volume fixtures because
Go's grid ModeMeshBP uses that same Forward+Step path. They do not establish
mesh-clock BPTT parity for non-Dense cells. The generic mesh runtime dispatches
supported GPU layers, but the mesh-clock wire and its parity sign-off cover Dense.
KMeans and Parallel remain deferred; no CPU fallback is supplied.

**Validated:** 2026-10-26, unreleased. Precision policy above applies. Benchmark
repetitions rebuild the initial child configuration and weights; timing includes
that reset, matching the Go fixture setup. These small correctness fixtures do
not establish a performance win.

## w2a Go↔Metal (Tween ModeTween) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tween-train`  
Wires: `TWB1`/`TWBO` (Stack ModeTween leaf), `TWV1`/`TWVO` (Dense volume),
and per-kind volume wires `TRV1`…`MBTV`, `MCTV`, `GDTV`, `SKTV`.
`TestGoVMetalTweenTrain` contains **80 case groups**, including **20 leaf ×34**
and **20 volume ×34** dtype sweeps. GDN is Float32-only.

All numerical projection, forward, backward, loss and SGD work runs on Metal.
The global gap is block-averaged onto each local output shape and applied at
`soft_lr = lr × 0.5`; Sequential/Residual expand their Dense leaves without
chain-rule credit. Host code schedules the walk and stages buffers/tapes.

| Case group | Status | Evidence |
|---|:---:|---|
| Dense multi-hop Float32 + 1-hop ×34 | [x] | Width-changing projected credit; native weight updates |
| RMSNorm / LayerNorm Float32 + ×34 | [x] | Exact weights; finite loss band |
| Softmax Float32 + ×34 | [x] | Weightless loss checks; GPU exp uses promoted evaluation before Float32 rounding |
| SwiGLU Float32 + ×34 | [x] | Exact weights; finite loss band |
| Sequential / Residual Float32 + ×34 | [x] | Leaf-projected credit, original forward inputs retained |
| Mamba / Metacognition Float32 + ×34 | [x] | Exact weights; finite loss band |
| GDN Float32 | [x] | ≥strict updated weights; finite loss band (one Float32-step weight difference remains) |
| NestedStack Float32 + ×34 | [x] | Flattened leaf walk; exact weights and finite loss band |
| Embedding / MHA Float32 + ×34 | [x] | Exact weights; finite loss band |
| CNN1/2/3 / ConvT1/2/3 ×34 | [x] | Explicit per-dimension leaf fixtures, including Float32 and rank-5 inputs |
| RNN / LSTM Float32 + ×34 | [x] | Exact weights; finite loss band |
| Dense volume 111 / 211 / 222 Float32 | [x] | Reverse-hop projected local updates |
| Volume 222 Float32: all 21 supported families | [x] | Per-kind local update dispatch; finite loss band |
| Volume 211 ×34: all 20 dtype-generic families | [x] | Native storage updates across all supported dtypes |
| GPU projection and nested-credit primitives | [x] | Downsample, upsample, cancellation, invalid shapes, and original-input local updates |
| Benchmark fixture reset | [x] | All leaf dtype sweeps and Dense volume dtype cases compare measured-run output against an independent run |

**Validated:** 2026-10-26, unreleased. Exact tensor bits are required where
specified; loss uses the finite float2 band above. GDN leaf weights require
≥strict, rather than exact bits. Volume weight subnormal flushing follows the
existing Metal policy. KMeans and Parallel remain deferred because their
NormalBP kernels are unavailable; no host numerical fallback is supplied.
This sign-off is ModeTween only. TweenChain has its own sign-off below;
Split/Alt and the requested Step/Mesh variants have separate sign-offs below; proxy/cache/sparse variants remain pending. Graph residency and performance tuning
remain open; benchmark timings include resetting the fixture.

## w2a Go↔Metal (TweenChain ModeTweenChain) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tweenchain-train`  
Wires: `TCB1`/`TCBO` (Stack FamilyBP at full lr), `TCV1`/`TCVO` (Dense volume),
and per-kind volume wires `TRC1`…`MBTC`, `MCTC`, `GDTC`, `SKTC`.
`TestGoVMetalTweenChainTrain` contains **80 case groups**, including **20 leaf ×34**
and **20 volume ×34** dtype sweeps. GDN is Float32-only.

Forward, backward, MSE/MSEGrad and native SGD use Metal kernels. TweenChain
propagates the true chain-rule input gradient at **full lr**, including through
Sequential, Residual and NestedStack. Volume credit is the same Forward+Step
operation as NormalBP; the named TweenChain entry point and every wire are
exercised here. Host code handles scheduling, allocation and tape/buffer staging.

| Case group | Status | Evidence |
|---|:---:|---|
| Dense multi-hop Float32 + 1-hop ×34 | [x] | Full-lr chain-rule credit; exact updated weights and finite loss band |
| RMSNorm / LayerNorm Float32 + ×34 | [x] | Exact weights; finite loss band |
| Softmax Float32 + ×34 | [x] | Weightless loss check uses the finite Metal loss band |
| SwiGLU Float32 + ×34 | [x] | Exact weights; finite loss band |
| Sequential / Residual Float32 + ×34 | [x] | Full chain rule through Dense children |
| Mamba / Metacognition Float32 + ×34 | [x] | Exact weights; finite loss band |
| GDN Float32 | [x] | ≥strict updated weights; finite loss band; Float32 rounding difference retained |
| NestedStack Float32 + ×34 | [x] | Recursive flattening with reverse chain-rule credit |
| Embedding / MHA Float32 + ×34 | [x] | Exact weights; finite loss band |
| CNN1/2/3 / ConvT1/2/3 ×34 | [x] | Explicit per-dimension fixtures, including Float32 and rank-5 inputs |
| RNN / LSTM Float32 + ×34 | [x] | Exact weights; finite loss band |
| Dense volume 111 / 211 / 222 Float32 | [x] | NormalBP-equivalent volume credit at full lr |
| Volume 222 Float32: all 21 supported families | [x] | Per-kind TweenChain protocol; true input-gradient comparisons |
| Volume 211 ×34: all 20 dtype-generic families | [x] | Native weight updates and chain-rule input gradients |
| GPU chain-rule/full-lr primitives | [x] | Hand-calculated Dense chain loss, weights and volume input gradient; invalid nil-loss/empty-stack rejection |
| Benchmark fixture reset | [x] | Leaf dtype sweeps and Dense volume dtype fixtures compare measured-run output with an independent run |

**Validated:** 2026-10-26, unreleased. Exact tensor bits are required where
specified; loss uses the finite float2 band above. GDN leaf updated weights
require ≥strict; global GDN Alt refreshed loss uses a finite relative `1e-5` band. GDN volume retains its NormalBP ≥industry floor, including
input gradients. Volume weight subnormal flushing follows the existing Metal
policy. KMeans/Parallel remain deferred; no CPU numerical fallback is supplied.
This sign-off covers ModeTweenChain. The requested Step/Mesh variants have separate sign-offs below; remaining proxy/cache/sparse variants retain pending gates. Graph residency and performance
tuning remain open; benchmark timings include resetting the fixture.

## w2a Go↔Metal (TweenSplit ModeTweenSplit) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tweensplit-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `TSB1`/`TSBO`; volume `TSP1`/`TSPO` and per-kind Split variants. `TestGoVMetalTweenSplitTrain`: **80 case groups**.

Single collect tape; global 1/N share at full lr. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 21 families; 20 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 21 supported families |
| Volume 211 ×34 | [x] | All 20 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Collect/phase primitives | [x] | Hand-calculated 1/N sharing and refreshed Alt loss/weights; input tape retained |

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. Global GDN Alt refreshed loss uses its finite `1e-5` relative band. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel remain unsupported. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (StepTweenSplit ModeStepTweenSplit) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptweensplit-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `STS1`/`STSO`; shared volume `TSP1` and per-kind Split variants. `TestGoVMetalStepTweenSplitTrain`: **80 case groups**.

Line per-hop Split at full lr; N counts trainable leaves under that hop. The popped global gap stays frozen. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 21 families; 20 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 21 supported families |
| Volume 211 ×34 | [x] | All 20 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [x] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. Global GDN Alt refreshed loss uses its finite `1e-5` relative band. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel remain unsupported. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (TweenAlt ModeTweenAlt) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tweenalt-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `TAB1`/`TABO`; volume `TAV1` and per-kind Alt variants. `TestGoVMetalTweenAltTrain`: **80 case groups**.

Split at full lr, refresh prediction and gap, then Tween at lr×0.5. AltTimes=1 is validated. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 21 families; 20 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 21 supported families |
| Volume 211 ×34 | [x] | All 20 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Collect/phase primitives | [x] | Hand-calculated 1/N sharing and refreshed Alt loss/weights; input tape retained |

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. Global GDN Alt refreshed loss uses its finite `1e-5` relative band. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel remain unsupported. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (StepTweenAlt ModeStepTweenAlt) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptweenalt-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `STA1`/`STAO`; shared volume `TAV1` and per-kind Alt variants. `TestGoVMetalStepTweenAltTrain`: **80 case groups**.

Line per-hop Split→Tween at full/half lr with a frozen popped gap. Its shared volume path refreshes the gap between phases. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 21 families; 20 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 21 supported families |
| Volume 211 ×34 | [x] | All 20 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [x] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. Global GDN Alt refreshed loss uses its finite `1e-5` relative band. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel remain unsupported. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (MeshTween mesh-clock) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --meshtween-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `MTW1`/`MTWO`; shared volume `TWV1` and per-kind Tween variants. `TestGoVMetalMeshTweenTrain`: **48 case groups**.

Last-tick captured mesh credit: projected local updates at lr×0.5; no BPTT. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense mesh-clock Float32 | [x] | 111 / 211 / 222; double-buffer ticks |
| Dense mesh-clock ×34 | [x] | 111 / 211 native dtypes; benchmark reset checked |
| Shared volume credit | [x] | All 21 supported families at 222 Float32; all 20 dtype-generic families at 211 ×34 |
| Credit primitive | [x] | Hand-calculated two-cell update distinguishes local Tween from BPTT/full-lr credit |

**Scope:** non-Dense mesh-clock parity remains open. Shared volume evidence exercises the Go ModeTween volume path; it does not establish mesh-clock parity for those cells.

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. Global GDN Alt refreshed loss uses its finite `1e-5` relative band. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel remain unsupported. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (MeshTweenChain mesh-clock) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --meshtweenchain-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `MTC1`/`MTCO`; shared volume `TCV1` and per-kind TweenChain variants. `TestGoVMetalMeshTweenChainTrain`: **48 case groups**.

Mesh-clock BPTT and SGD at full lr (FamilyBP, equivalent to MeshBP). Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense mesh-clock Float32 | [x] | 111 / 211 / 222; double-buffer ticks |
| Dense mesh-clock ×34 | [x] | 111 / 211 native dtypes; benchmark reset checked |
| Shared volume credit | [x] | All 21 supported families at 222 Float32; all 20 dtype-generic families at 211 ×34 |
| Credit primitive | [x] | Hand-calculated two-cell update distinguishes local Tween from BPTT/full-lr credit |

**Scope:** non-Dense mesh-clock parity remains open. Shared volume evidence exercises the Go ModeTweenChain volume path; it does not establish mesh-clock parity for those cells.

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. Global GDN Alt refreshed loss uses its finite `1e-5` relative band. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel remain unsupported. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (StepTween ModeStepTween) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptween-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `STW1`/`STWO`; shared volume `TWV1` and per-kind Tween variants. `TestGoVMetalStepTweenTrain`: **80 case groups**.

Line per-hop projected local credit at lr×0.5, applied when a training flight pops. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 21 families; 20 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 21 supported families |
| Volume 211 ×34 | [x] | All 20 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [x] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. Global GDN Alt refreshed loss uses its finite `1e-5` relative band. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel remain unsupported. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (StepTweenChain ModeStepTweenChain) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptweenchain-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `STC1`/`STCO`; shared volume `TCV1` and per-kind TweenChain variants. `TestGoVMetalStepTweenChainTrain`: **80 case groups**.

Line per-hop true chain-rule BP at full lr, applied when a training flight pops. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 21 families; 20 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 21 supported families |
| Volume 211 ×34 | [x] | All 20 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [x] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. Global GDN Alt refreshed loss uses its finite `1e-5` relative band. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel remain unsupported. Full graph residency and performance tuning remain open.

## 1 — NormalBP

Chain-rule Backward + ApplyGradSGD. Start here. Dense first.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** all 21 supported NormalBP layer families validated inside nodes and in volumes, including dtype coverage (GDN Float32 only), under the precision policy above. KMeans and Parallel remain open. StepBP Line and the requested MeshBP evidence are validated below; non-Dense mesh-clock parity remains separate.

---

## 2 — StepBP

Needs Step `Line` pipe + FamilyBP (NormalBP credit) on pop.  
Ship: thin Stack host + `runtime/line` TickLine/TrainLineMSE; wire `STB1`/`STBO`; w2a `--stepbp-train`.  
**3D volumetric** for StepBP = MeshBP (`runtime/step` mesh-clock FamilyBP + ModeMeshBP volume Step). All CNN/ConvT dimensions have explicit Line ×34 fixtures, including rank-5 inputs.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** all 21 supported Line families and the shared 3D volume path validated, including all supported dtypes (GDN Float32 only). KMeans/Parallel remain open. Actual mesh-clock BPTT evidence is Dense-only; see section 7.

---

## 3 — Tween

ModeTween: projected local credit @ `soft_lr = lr × 0.5` (Go `applyProjectedTween` — same gap on every leaf, no chain rule).  
Ship: `systems/tween` + `runtime/tween_train`; wires `TWB1`/`TWBO` + `TWV1`/`TWVO` + per-kind `TRV1`…`MBTV`; w2a `--tween-train`.  
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. 3D = reverse-hop volume 2×2×2×1 for all NormalBP-signed kinds.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** all 21 supported families validated on Metal in leaves and volumes, including all supported dtypes (GDN Float32 only). GDN leaf updated weights retain a ≥strict floor; other declared bands above apply. KMeans/Parallel remain open; other variant evidence is recorded separately.

---

## 4 — TweenChain

ModeTweenChain: named Tween, credit = BP (Go `familyTweenChain` ≡ `familyBP` @ full lr).  
Ship: `systems/tween_chain` + `runtime/tween_chain_train`; wires `TCB1`/`TCBO` + `TCV1`/`TCVO` + per-kind `TRC1`…`MBTC`; w2a `--tweenchain-train`.  
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. 3D = volume Step 2×2×2×1 for all NormalBP-signed kinds.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** all 21 supported families validated on Metal in leaves and volumes, including every supported dtype (GDN Float32 only). Full-rate chain-rule credit and volume input gradients are verified. GDN retains its declared precision floors; KMeans/Parallel remain open. Step/Mesh evidence is recorded separately below.

---

## 5 — StepTween

Needs Line + Tween.  
Ship: `runtime/line` Credit::FamilyTween on pop (`apply_projected` @ `lr×0.5`); wire `STW1`/`STWO`; w2a `--steptween-train`.  
**3D volumetric** for StepTween = MeshTween volume (= ModeTween reverse-hop projected local; existing `TWV1`/`TRV1`… wires). CNN2/3 and ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.

---

## 6 — StepTweenChain

Needs Line + TweenChain.  
Ship: `runtime/line` Credit::FamilyBP on pop (same credit as StepBP / TweenChain); wire `STC1`/`STCO`; w2a `--steptweenchain-train`.  
**3D volumetric** = MeshTweenChain volume (= ModeTweenChain volume Step wires `TCV1`/`TRC1`…).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.

---

## 7 — MeshBP

Grid tick FamilyBP: `runtime/step` Forward×ticks + Backward BPTT + ApplyGradSGD (`MBP1`/`MBPO` for Dense mesh-clock).  
Go ModeMeshBP on grid also = volume Forward+Step (cameral_train). Dense mesh-clock is validated at 111/211/222 Float32 and 111/222 ×34, plus disabled/remote routing. All supported families have shared volume-path evidence at 222 Float32 and 211 ×34 (GDN Float32 only). Non-Dense mesh-clock parity remains unverified.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| LayerNorm | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| SwiGLU | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| Softmax | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| Embedding | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| MHA | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| CNN1 | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| CNN2 | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| CNN3 | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| ConvT1 | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| ConvT2 | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| ConvT3 | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| RNN | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| LSTM | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| Sequential | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| Residual | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| Metacognition | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| GDN | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [x] | [ ] | volume path validated; mesh-clock parity pending |

## 8 — MeshTween

Needs Mesh + Tween.  
Ship: `runtime/step` `Credit::FamilyTween` — Forward×ticks → last-tick `apply_cell_projected` @ `lr×0.5` (`MTW1`/`MTWO`); w2a `--meshtween-train`.  
**3D volumetric** = ModeTween volume credit (`TWV1`/`TRV1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| LayerNorm | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| SwiGLU | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Softmax | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Embedding | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| MHA | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| CNN1 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| CNN2 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| CNN3 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| ConvT1 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| ConvT2 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| ConvT3 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| RNN | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| LSTM | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Sequential | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Residual | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Metacognition | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| GDN | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |

**Checkpoint:** Dense clock verified at 111/211/222 Float32 and 111/211 ×34; all supported shared volume families/dtypes verified. Non-Dense clock parity remains open.

---

## 9 — MeshTweenChain

Needs Mesh + TweenChain.  
Ship: `runtime/step` `Credit::FamilyBP` under mesh clock (`MTC1`/`MTCO` ≡ MeshBP); w2a `--meshtweenchain-train`.  
**3D volumetric** = ModeTweenChain volume credit (`TCV1`/`TRC1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| LayerNorm | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| SwiGLU | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Softmax | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Embedding | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| MHA | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| CNN1 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| CNN2 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| CNN3 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| ConvT1 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| ConvT2 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| ConvT3 | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| RNN | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| LSTM | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Sequential | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Residual | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Metacognition | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| GDN | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [x] | [ ] | shared volume ×34; mesh-clock parity pending |

**Checkpoint:** Dense clock verified at 111/211/222 Float32 and 111/211 ×34; all supported shared volume families/dtypes verified. Non-Dense clock parity remains open.

---

## 10 — TweenSplit

Ship: `systems::tween_split` + `runtime/tween_split_train` — collect trainable leaves, apply \(g_i=\frac1N P(g_y)\) @ full lr (`TSB1`/`TSP1`/`TRP1`…); w2a `--tweensplit-train`.  
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. Softmax weightless (N may be 0).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.

---

## 11 — StepTweenSplit

Needs Line + TweenSplit.  
Ship: `runtime/line` Credit::FamilyTweenSplit on pop (`apply_split_child` 1/N @ full lr); wire `STS1`/`STSO`; w2a `--steptweensplit-train`.  
**3D volumetric** = MeshTweenSplit volume (= ModeTweenSplit / TweenSplit volume wires `TSP1`/`TRP1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.

---

## 12 — TweenAlt

ModeTweenAlt: Split at full lr, refresh the prediction/gap, then Tween at lr×0.5.
Ship: `systems/tween_alt` + `runtime/tween_alt_train`; `TAB1`/`TABO` and `TAV1`/per-kind volume wires.
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. Default AltTimes=1.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.

---

## 13 — StepTweenAlt

Needs Line + TweenAlt.  
Ship: `runtime/line` Credit::FamilyTweenAlt on pop (`apply_alt_child` Split→Tween frozen gap); wire `STA1`/`STAO`; w2a `--steptweenalt-train`.  
**3D volumetric** = MeshTweenAlt volume (= ModeTweenAlt / TweenAlt volume wires `TAV1`/`TRA1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.
**3D volumetric** = ModeTweenSplit / TweenSplit volume credit (`TSP1`/`TRP1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.
**3D volumetric** = ModeTweenAlt / TweenAlt volume credit (`TAV1`/`TRA1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.
**3D volumetric** = ModeTweenSplitFastProxy / TweenSplitFastProxy volume credit (`FPV1`/`FRP1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.
**3D volumetric** = ModeTweenSplitSparse / TweenSplitSparse volume credit (`SPV1`/`SRP1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RMSNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LayerNorm | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| SwiGLU | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Softmax | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Embedding | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| MHA | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| CNN3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT1 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT2 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| ConvT3 | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| RNN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| LSTM | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Sequential | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Residual | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 21 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel remain open.

---

## 24 — StepTweenSplitHeadProxy

Needs Line + TweenSplitHeadProxy.  
Ship: `runtime/line` `Credit::FamilyTweenSplitHeadProxy` on pop (`apply_head_proxy_child`); wire `STH1`/`STHO`; w2a `--steptweensplitheadproxy-train`.  
**3D volumetric** = ModeTweenSplitHeadProxy / TweenSplitHeadProxy volume credit (`HPV1`/`HRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** — _(open)_

---

## 25 — StepTweenSplitLinear

Needs Line + TweenSplitLinear.  
Ship: `runtime/line` `Credit::FamilyTweenSplitLinear` on pop (`apply_linear_child`); wire `STL1`/`STLO`; w2a `--steptweensplitlinear-train`.  
**3D volumetric** = ModeTweenSplitLinear / TweenSplitLinear volume credit (`LLV1`/`LRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** — _(open)_

---

## 26 — StepTweenSplitFastProxy

Needs Line + TweenSplitFastProxy.  
Ship: `runtime/line` `Credit::FamilyTweenSplitFastProxy` on pop (`apply_fast_proxy_child`); wire `STF1`/`STFO`; w2a `--steptweensplitfastproxy-train`.  
**3D volumetric** = ModeTweenSplitFastProxy / TweenSplitFastProxy volume credit (`FPV1`/`FRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** — _(open)_

---

## 27 — StepTweenSplitLinearCache

Needs Line + TweenSplitLinearCache.  
Ship: `runtime/line` `Credit::FamilyTweenSplitLinearCache` on pop (`apply_linear_cache_child`, Dense hops accel=nil); wire `SLC1`/`SLCO`; w2a `--steptweensplitlinearcache-train`.  
**3D volumetric** = ModeTweenSplitLinearCache / TweenSplitLinearCache volume credit (`LCV1`/`CRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** — _(open)_

---

## 28 — StepTweenSplitHeadProxyAsync

Needs Line + TweenSplitHeadProxyAsync.  
Ship: `runtime/line` `Credit::FamilyTweenSplitHeadProxyAsync` on pop (`apply_head_proxy_async_child`, Dense hops accel=nil); wire `SHA1`/`SHAO`; w2a `--steptweensplitheadproxyasync-train`.  
**3D volumetric** = ModeTweenSplitHeadProxyAsync / TweenSplitHeadProxyAsync volume credit (`HAV1`/`ARPA`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** — _(open)_

---

## 29 — StepTweenSplitSparse

Needs Line + TweenSplitSparse.  
Ship: `runtime/line` `Credit::FamilyTweenSplitSparse` on pop (`apply_sparse_child`, Dense hops accel=nil → hidden idx=0); wire `SSP1`/`SSPO`; w2a `--steptweensplitsparse-train`.  
**3D volumetric** = ModeTweenSplitSparse / TweenSplitSparse volume credit (`SPV1`/`SRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** — _(open)_

---

## Version log

The older tags below preserve historical wire milestones, some of which used
host computation. They do not establish present GPU sign-off. The dated
unreleased validation and checked tables above describe current Metal evidence.

| Tag | Notes |
|---|---|
| Unreleased — 2026-10-26 | Metal-only NormalBP: all 17 evidence sections / 128 cases passed, covering 21 layer families and their supported dtypes. GPU native encoding/bias SGD, all Dense activations/derivatives, and primitive/cache/no-host-API checks passed. StepBP Line: all 21 supported families / dtypes; MeshBP: Dense mesh-clock ×34 and all supported volume families. Disabled/remote routing and benchmark reset checks passed. Non-Dense mesh-clock parity remains open. ModeTween: 80 case groups, supported leaf/volume dtypes, Metal projection/local credit and benchmark reset checks passed. ModeTweenChain: 80 groups, supported leaf/volume dtypes, full-lr chain-rule credit and true input gradients validated. Eight requested Split/Alt/Step/Mesh sections: 576 groups; tiny-SGD preservation and credit/phase primitives verified. |
| `0.19-m2-meshbp` | StepBP Line + MeshBP Go↔Metal: STB1 Line FamilyBP × kinds (+×34) + MBP1 Dense mesh-clock 111/211/222 + ModeMeshBP vol 222 for NormalBP-signed kinds (`runtime/line` + `runtime/step`; KMeans Line deferred) |
| `0.18-m2-stack-nbp` | Stack NestedStack-of-Dense NormalBP Go↔Metal: leaf Float32+×34 + volume 111/chain/×34/222/211×34 exact_bit (`STKB`/`STKV`; Dense Dim→Dim Linear, loom n_children×[Dim×Dim]; bind_stack) |
| `0.17-m2-gdn-nbp` | GDN NormalBP Go↔Metal: leaf ≥loose + volume 111/chain/222 ≥industry f32-only (`GDB1`/`GDV1`; truncated-BPTT; Metal≡cpp host f32 scan) |
| `0.16-m2-metacognition-nbp` | Metacognition NormalBP Go↔Metal: leaf Float32+×34 + volume 111/chain/×34/222/211×34 exact_bit (`MCB1`/`MCV1`; lastGate-scale + Observed Dense, loom [Dim×Dim]) |
| `0.15-m2-mamba-nbp` | Mamba NormalBP Go↔Metal: leaf Float32+×34 + volume 111/chain/×34/222/211×34 exact_bit (`MBB1`/`MBV1`; scan BPTT + In/Out Dense, loom In\|Out\|ALog\|DSkip) |
| `0.14-m2-residual-nbp` | Residual NormalBP Go↔Metal: leaf Float32+×34 + volume 111/chain/×34/222/211×34 exact_bit (`REB1`/`REV1`; F reverse Dense + skip grad, loom Depth×[Dim×Dim]) |
| `0.13-m2-sequential-nbp` | Sequential NormalBP Go↔Metal: leaf Float32+×34 + volume 111/chain/×34/222/211×34 exact_bit (`SQB1`/`SQV1`; reverse Dense Dim→Dim chain, loom Depth×[Dim×Dim]) |
| `0.12-m2-lstm-nbp` | LSTM NormalBP Go↔Metal: leaf Float32+×34 + volume 111/chain/×34/222/211×34 exact_bit (`LSB1`/`LSV1`; BPTT 4-gate Dense, loom i\|f\|g\|o packs) |
| `0.11-m2-rnn-nbp` | RNN NormalBP Go↔Metal: leaf Float32+×34 + volume 111/chain/×34/222/211×34 exact_bit (`RNB1`/`RNV1`; BPTT Dense IH/HH + tanh′, loom dIH\|dHH\|dBias) |
| `0.10-m2-convt-nbp` | ConvT1/2/3 NormalBP Go↔Metal: leaf Float32+×34 + volume 111/chain/×34/222/211×34 exact_bit (`T1B1`/`T2B1`/`T3B1` + `T*V1`; scatter+act′+adjoint, no bias) |
| `0.09-m2-cnn-nbp` | CNN1/2/3 NormalBP Go↔Metal: leaf Float32+×34 + volume 111/chain/×34/222/211×34 exact_bit (`C1B1`/`C2B1`/`C3B1` + `C*V1`; im2col→Dense bwd→col2im) |
| `0.08-m2-mha-nbp` | MHA NormalBP Go↔Metal: leaf DecoderCausal/GQA/Encoder + ×34 (W exact; dX/dW ≥strict) + volume 111/chain/×34/222/211×34 exact_bit (`WHB1`/`WHV1`; resident Q/K/V/O) |
| `0.07-m2-embedding-nbp` | Embedding NormalBP Go↔Metal: leaf Float32+×34 + volume 1×1×1×1 / chain / ×34 / 2×2×2×1 / 2×1×1×1×34 exact_bit (`WEB1`/`WEV1`; dX=0 scatter dW) |
| `0.06-m2-softmax-nbp` | Softmax NormalBP Go↔Metal: leaf Standard+Kind×34 + volume 111/chain/×34/222/211×34 ≥loose (~strict); Gumbel smoke (`WXB1`/`WXV1`) |
| `0.05-m2-swiglu-nbp` | SwiGLU NormalBP Go↔Metal: leaf Float32+×34 + volume 1×1×1×1 / chain / ×34 / 2×2×2×1 / 2×1×1×1×34 exact_bit (`WSB1`/`WSV1`) |
| `0.04-m2-layernorm-nbp` | LayerNorm NormalBP Go↔Metal: leaf Float32+×34 + volume 1×1×1×1 / chain / ×34 / 2×2×2×1 / 2×1×1×1×34 exact_bit (`WLB1`/`WLV1`) |
| `0.03-m2-rmsnorm-nbp` | RMSNorm NormalBP Go↔Metal: leaf Float32+×34 + volume 1×1×1×1 / chain / ×34 / 2×2×2×1 / 2×1×1×1×34 exact_bit (`WRB1`/`WRV1`) |
| `0.02-m2-dense-nbp` | Dense NormalBP Go↔Metal: leaf Float32+×34 + volume 1×1×1×1 / chain / ×34 / 2×2×2×1 / 2×1×1×1×34 exact_bit (`WDB1`/`WVN1`) |
| — | Checklist opened. Forward volume FormatNone×34 largely done; training port not started. |

