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

## w2a Go↔Metal (Freeze / Shadow / Adversarial / Memory / Cameral / CamSync) — sign-off evidence

2026-10-26, working-tree milestone **0.68**. Native C++ APIs, Metal numerical
kernels and w2a `CMB1`/`CMBO` bridge; no welvet.cpp dependency.

| Scope | Inside node | Placed volume | Native dtype sweep | Evidence |
|---|:---:|:---:|:---:|---|
| Freeze / Shadow | ✅ | ✅ | ✅ | All 23 families; 111/211 placements; exact frozen credit and input-gradient guards |
| Adversarial / Memory | ✅ | ✅ | ✅ | Branch ascent, sleep/wake gates; 111 native and 222 Float32 volumes |
| Shadow teacher credit | ✅ | ✅ | ✅ | First teacher, GPU consistency gap, frozen teacher and independent scalar oracle |
| Cameral BranchModes | ✅ | ✅ | ✅ | All enum choices; seven combines, rate scales, Stack child/nested overrides |
| CamSync / BlendStores | ✅ | ✅ | ✅ | Groups, biases, cross/nested endpoints, no-store cases, all four sync boundaries |
| Native replacement / coherence | ✅ | ✅ | ✅ | Fresh native encoding, primed forward probes, no-write byte preservation; Go cache regression |

Training: **465 groups** in `TestGoVMetalCameralTrain`. Synchronization:
**109 groups** in `TestGoVMetalCamSync`. The native sweep is 34 parameter dtypes
except GDN Float32. Shape comparisons and same-Metal reset/replay are exact.
These suites use documented combined absolute/relative industry-or-better
numerical gates; existing regression gates remain unchanged. Full contracts,
coverage boundaries and reference quirks are in [CAMERAL.md](CAMERAL.md).
In particular, explicit cross links can write frozen endpoints, Metacognition
can reset during forward, and GDN/Softmax have no blendable Go weights.Store.
Placed-volume sign-off does not assert mixed actors for every mesh-clock credit.

```sh
./scripts/macos_build_w2a.sh --cameral-all
```

The full `--training-all` command includes these suites and the prior numbered
mode regressions. Native tests stay in w2a, including scalar semantic guards
and `TestGoStoreReplaceInvalidatesWires`.

### Numerical execution inventory

| Path | Execution today |
|---|---|
| NormalBP layers | Dense, RMSNorm, LayerNorm, SwiGLU, Softmax, Embedding, MHA, CNN1/2/3, ConvT1/2/3, RNN, LSTM, Sequential, Residual, KMeans, Mamba, Metacognition, GDN, Parallel and Stack use Metal numerical kernels |
| MSE / MSEGrad | Metal kernels; scalar loss is exported in IEEE float64 representation |
| SGD and native storage updates | Metal arithmetic and native FormatNone encoding across all 34 dtypes; bias updates run on Metal too |
| GDN | Metal Float32 truncated-BPTT; state/conv history stop-gradient |
| Tween ModeTween | Metal block-average projection and local credit at lr×0.5; all 23 families / supported dtypes |
| TweenChain ModeTweenChain | Metal full-lr chain-rule BP in leaves/volumes; all 23 families / supported dtypes |
| Split / Alt and requested Step/Mesh variants | GPU 1/N scaling, refreshed global Alt phases, frozen Line per-hop credit, last-tick MeshTween and BPTT MeshTweenChain |
| Proxy / cache / sparse (sections 14–19) | Metal head/linear credit, GPU norm/rescaling, cached/asynchronous proxies and sparse rotation; shared volume parity uses the 60–100% project spectrum |
| Mesh Split / Alt / FastProxy / Sparse (20–23) | Final-tick local credit on Metal, 652 clock/shared-volume groups |
| Line HeadProxy (24) | Frozen per-hop gap with Metal head backward and hidden projected credit |
| Scheduling and data movement | Host validation, scheduling, allocation, copies and storage staging remain |
| Mesh gradient joins | Metal accumulation kernel; Dense mesh-clock ×34, disabled-cell and remote-link checks |
| KMeans / Parallel NormalBP | Metal soft assignments, all seven combines, reverse routing, gate gradients and native GPU SGD |
| Performance | Correctness gates passed; full residency, dispatch reduction and throughput tuning remain open |

**Validated 2026-10-26:** all **19 NormalBP evidence sections / 219 case groups**
passed in `w2a` under `TestGoVMetalNormalBP`, including all dtype loops and leaf,
chain and volumetric fixtures. `TestGoVMetalOnly` also passed its primitive,
activation/derivative, native update, cache-coherence, rejected-host-API and 13
Float32 NormalBP/Line/Mesh smoke cases. Shared and static builds passed.
Tween ModeTween also passes 130 case groups covering supported leaf and volume
dtypes, plus GPU projection/nested-credit primitive checks. TweenChain passes 130
case groups covering supported leaf/volume dtypes and true chain-rule credit.
The earlier eight-section Split/Alt/Line/Mesh validation covered 576 case groups, plus hand-calculated credit and tiny-SGD primitive checks. The StepTween/StepTweenChain extensions are covered by the newer combined run below. These are working-tree
validations; no new release tag is claimed.

**Sections 2–6 completed 2026-10-26:** KMeans and Parallel add 50 case groups
to each requested mode (250 new groups). The full combined runner passed
**654 requested-mode groups + 219 NormalBP groups + Metal-only checks** in
538.583 seconds. StepBP has 134 groups; Tween, TweenChain, StepTween and
StepTweenChain each have 130. Shared and static builds passed.

**Project parity target:** best-effort deterministic agreement with Go, aiming
for 100% and accepting 60–100% on w2a's spectrum (`industry` through `exact_bit`).
Long-horizon training need not reproduce an identical CPU trajectory. See
[the README policy](../README.md#numerical-parity-target). The fixture gates
below record existing evidence and may be stronger than that project target.

**Sections 11–24 completed 2026-10-26:** 130 groups each for StepTweenSplit,
TweenAlt and StepTweenAlt; six proxy leaf suites (65 each, plus 23 cache-refresh
groups), 65 Line HeadProxy groups, six proxy volume suites (69 each, plus 23
cache-refresh groups), and 652 mesh-clock/shared-volume groups. NormalBP,
TweenSplit, Metal-only and 16 Line Alt pop controls also passed. Verification
completed across the combined run and focused follow-up runs after the GDN
fixture-policy correction and the accepted 60–100% proxy volume policy.
Shared/static builds passed. Run `./scripts/macos_build_w2a.sh --training-11-24-all`
to execute the complete batch.

**Sections 25–29 completed 2026-10-26:** 352 new Line groups: 69 each for
Linear, FastProxy, Async and Sparse, and 76 for LinearCache. Includes all
23 families, supported native dtypes, pipeline warmup/pops, stateful Filter,
polymorphic branches, exact same-backend replay and seven 26-tick cache-refresh
cases. Shared volume evidence reuses the parent proxy paths above. Shared/static
builds and GPU-only guards passed. The 352-group Line/guard run passed in
603.623 seconds. All 29 numbered mode tables are checked.

**Fixture precision policy:** exact tensor bits where required by each case; finite loss
uses `abs(go-metal) <= 1e-12 * max(1, abs(go))`. NormalBP weight comparisons allow
FP32 subnormal flushing only when both values are below the smallest normal
FP32 value; normal values still require exact bits. MHA leaf ×34 retains its
existing ≥strict gradient floor. Softmax retains ≥loose (with stricter rows as
shown), and GDN retains ≥loose leaf / ≥industry volume. Stack's existing binary
211 post allowance is `max|Δ| <= 1e-18`, with its other checks retained.
KMeans input-gradient cancellation has a separate finite band: mismatching
`dX` / volume `gradIn` values must both have magnitude ≤`1e-12` and differ by
≤`1e-14`. This is a near-zero cancellation allowance, not FP32 subnormal
flushing. KMeans pre/post, weight gradients and updated weights retain their
exact gates; its scalar loss retains the common finite band.

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

**All 29 training modes:** run `./scripts/macos_build_w2a.sh --training-all`
to build and run the full training matrix plus Metal-only regressions.
Run `./scripts/macos_build_w2a.sh --training-25-29-all` for the final five Line
modes, shared proxy volumes and Metal-only guard.

**NormalBP subset:** run `./scripts/macos_build_w2a.sh --normalbp-all` to build
Metal and run both `TestGoVMetalNormalBP` and `TestGoVMetalOnly` in sibling `w2a`.
Run `./scripts/macos_build_w2a.sh --step-tween-all` for NormalBP plus all five
sections 2–6 and the Metal-only guard.
The combined runners default to one measured benchmark iteration per fixture;
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

## w2a Go↔Metal (KMeans NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --normalbp-all` (`KMeans` and KMeans
`NormalBPEdges` groups in `TestGoVMetalNormalBP`).
Wires: `KMB1`/`KMBO`, `KMV1`/`KMVO`; exported `welvet_kmeans_normalbp_f32` ABI.

| Evidence | Status | Scope |
|---|:---:|---|
| Probabilities and Features Float32 | [x] | Forward, stored pre, dX, dW, native SGD, finite MSE loss |
| Probabilities and Features ×34 | [x] | FormatNone native centers; both output modes |
| All eight activations / temperatures | [x] | Probabilities and Features; temperature 0→1, 0.25, 2.5; rank-3 inputs |
| Volume 111 and two-layer chain Float32 | [x] | Features, complete reverse credit and updated centers |
| Volume 111 / 211 ×34 | [x] | Output, input gradient, loss and updated native weights |
| Volume 222 Float32 | [x] | Multi-cell joins and reverse-hop credit |

**Validated:** 2026-10-26, unreleased. Metal computes distances, softmax,
reconstruction, activation, assignment/center/input gradients and SGD. Go's
Float32 activation/derivative materialization points are preserved. Dynamic GPU
scratch has no fixed cluster-count cap. The narrow input-gradient cancellation
band above is explicit; larger errors and non-finite values fail. Performance
and complete graph residency remain separate work.

---

## w2a Go↔Metal (Parallel NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --normalbp-all` (`Parallel`, `ParallelPoly`,
Parallel `NormalBPEdges`, and `ParallelVolumes` groups).
Wires: `PNB1`/`PNBO`, `PPB1`/`PPBO`, `PNV1`/`PNVO`, `NBL1`/`NBLO`.
Exported Dense-branch ABI: `welvet_parallel_normalbp_f32`.

| Evidence | Status | Scope |
|---|:---:|---|
| All seven combines Float32 | [x] | Concat, Add, Avg, Filter, Max, SparseK, Disagree |
| Direct gradients for all seven combines ×34 | [x] | Rank-3 pre/post, dX, dW, native SGD; Filter gate weights included |
| Polymorphic branches | [x] | All 23 branch kinds Float32; 15 dtype-generic branch fixtures ×34 |
| Nested Parallel / Stack | [x] | Recursive GPU forward/backward and packed updates |
| Routing edges | [x] | First-branch Max ties, index-ordered SparseK ties, SparseK=1, zero lr, unequal-width Concat, nested Filter gate |
| Volume 111 / chain / 211 ×34 | [x] | Add and Avg; loss, output, input gradients, native weights |
| Volume 222 Float32 | [x] | Add branches through multi-cell graph joins |
| ABI / host-API guard | [x] | Exported ABI gradients and SGD match C++ Metal API; legacy CPU matvec still rejects |

**Validated:** 2026-10-26, unreleased. Branch loops on the host only schedule
kernels and stage buffers. Metal performs every combine, routing operation,
softmax gate derivative and input-gradient reduction. Input gradients cast only
after the promoted branch sum; Avg backward uses the promoted reciprocal.
Integer-backed 53-bit arithmetic preserves cancellation and Go arm64 fused
multiply/add rounding in Disagree. Native FormatNone weights remain native.
Other TrainModes, full residency and throughput tuning retain separate gates.

---

## w2a Go↔Metal (KMeans + Parallel, sections 2–6) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --step-tween-all`.
Cases: [`kmeans_parallel_modes.go`](../../w2a/suites/go_v_metal/kmeans_parallel_modes.go).
Each mode adds the same 50 groups below, with fresh Go fixtures and independent
benchmark replay. The combined suite also reruns the 219 NormalBP groups and
`TestGoVMetalOnly`.

| New evidence per mode | Groups | Status | Scope |
|---|---:|:---:|---|
| KMeans Probabilities / Features | 4 | [x] | Float32 and ×34; SiLU, temperature 0.7, shape-changing KMeans→Dense chain |
| All seven Parallel combines | 14 | [x] | Float32 and ×34; three branches, Filter gate export, Concat→Dense width change |
| Polymorphic Parallel branches | 23 | [x] | Every branch kind Float32; nested Parallel/Stack, recurrent and convolutional branches |
| Stateful Filter collection | 1 | [x] | Metacognition rules and gate, capture-before-update semantics |
| KMeans / Parallel volume credit | 8 | [x] | 111 / two-layer chain / 211 ×34, plus 222 Float32 for each kind |

**Validated:** 2026-10-26, unreleased. StepBP/StepTween/StepTweenChain use five
Line ticks in the multi-hop fixtures. Tween and StepTween project the global
gap onto each trainable Parallel leaf and its gate at half lr; the collect tape
is complete before SGD starts. Filter performs its Go-compatible extra forward,
including stateful rule counters. TweenChain uses full-rate BP. Volume credit
uses the same shared Forward+Step paths as the existing mode fixtures; Parallel
volume Tween treats the cell as a local backward/update unit.

Leaf wires: `STB1`/`STBO`, `TWB1`/`TWBO`, `TCB1`/`TCBO`, `STW1`/`STWO`,
`STC1`/`STCO`. Volume wires: `KMV1`/`KMVO`, `KMTV`/`KMTO`, `KMTC`/`KMCO`,
`PNV1`/`PNVO`, `PNTV`/`PNTO`, `PNTC`/`PNCO`. Step variants reuse their named
parent volume credit as specified in each mode section.

Native weight checks remain exact or within the existing FP32 subnormal policy;
GDN branches retain the declared ≥strict weight floor, exact packed lengths,
and finite loss band. Benchmark outputs must match independent fresh outputs.
KMeans volume input gradients retain the narrow cancellation band declared
above. GPU-only probes also exercise KMeans/Parallel mesh SGD. Full mesh-clock
Go parity is validated separately in section 7. Full residency and throughput
remain separate work.

---

## w2a Go↔Metal (StepBP Line + MeshBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --stepbp-train`  
Wires: `STB1`/`STBO` (Line FamilyBP); `MBP1`/`MBPO` (Dense mesh-clock FamilyBP).
`TestGoVMetalStepBPTrain` contains **134 case groups**, including the dtype loops below.
The runner also executes `TestGoVMetalOnly` and defaults to one measured iteration.
Every forward, backward, loss, gradient join and SGD calculation uses Metal kernels;
host code schedules ticks, stores tapes and stages buffers.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

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
| Volume 222 Float32: all 23 supported families | [x] | Shared volume Forward+Step path used by Go ModeMeshBP |
| Volume 211 ×34: all 22 dtype-generic families | [x] | GDN is Float32-only; retains its volume ≥industry floor |
| Benchmark fixture reset | [x] | Line dtype sweeps and GDN compare benchmark results with fresh-run results; Dense mesh dtype/routing fixtures do likewise |

**Scope:** volume rows deliberately reuse the NormalBP volume fixtures because
Go's grid ModeMeshBP uses that same Forward+Step path. They do not establish
mesh-clock BPTT parity for non-Dense cells. The separate `MGB1` all-family
clock fixtures and section 7 provide that evidence.
KMeans and Parallel are signed off for Line/shared-volume credit above. No CPU fallback is supplied.

**Validated:** 2026-10-26, unreleased. Precision policy above applies. Benchmark
repetitions rebuild the initial child configuration and weights; timing includes
that reset, matching the Go fixture setup. These small correctness fixtures do
not establish a performance win.

## w2a Go↔Metal (Tween ModeTween) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tween-train`  
Wires: `TWB1`/`TWBO` (Stack ModeTween leaf), `TWV1`/`TWVO` (Dense volume),
and per-kind volume wires `TRV1`…`MBTV`, `MCTV`, `GDTV`, `SKTV`.
`TestGoVMetalTweenTrain` contains **130 case groups**, including **29 leaf ×34**
and **26 volume ×34** dtype sweeps. GDN is Float32-only.

All numerical projection, forward, backward, loss and SGD work runs on Metal.
The global gap is block-averaged onto each local output shape and applied at
`soft_lr = lr × 0.5`; Sequential/Residual expand their Dense leaves without
chain-rule credit. Host code schedules the walk and stages buffers/tapes.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

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
| Volume 222 Float32: all 23 supported families | [x] | Per-kind local update dispatch; finite loss band |
| Volume 211 ×34: all 22 dtype-generic families | [x] | Native storage updates across all supported dtypes |
| GPU projection and nested-credit primitives | [x] | Downsample, upsample, cancellation, invalid shapes, and original-input local updates |
| Benchmark fixture reset | [x] | All leaf dtype sweeps and Dense volume dtype cases compare measured-run output against an independent run |

**Validated:** 2026-10-26, unreleased. Exact tensor bits are required where
specified; loss uses the finite float2 band above. GDN leaf weights require
≥strict, rather than exact bits. Volume weight subnormal flushing follows the
existing Metal policy. KMeans and Parallel have independent ModeTween evidence above. No host numerical fallback is supplied.
This sign-off is ModeTween only. TweenChain has its own sign-off below;
Split/Alt and the proxy/cache/sparse and Step/Mesh variants have separate sign-offs below. Graph residency and performance tuning
remain open; benchmark timings include resetting the fixture.

## w2a Go↔Metal (TweenChain ModeTweenChain) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tweenchain-train`  
Wires: `TCB1`/`TCBO` (Stack FamilyBP at full lr), `TCV1`/`TCVO` (Dense volume),
and per-kind volume wires `TRC1`…`MBTC`, `MCTC`, `GDTC`, `SKTC`.
`TestGoVMetalTweenChainTrain` contains **130 case groups**, including **29 leaf ×34**
and **26 volume ×34** dtype sweeps. GDN is Float32-only.

Forward, backward, MSE/MSEGrad and native SGD use Metal kernels. TweenChain
propagates the true chain-rule input gradient at **full lr**, including through
Sequential, Residual and NestedStack. Volume credit is the same Forward+Step
operation as NormalBP; the named TweenChain entry point and every wire are
exercised here. Host code handles scheduling, allocation and tape/buffer staging.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

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
| Volume 222 Float32: all 23 supported families | [x] | Per-kind TweenChain protocol; true input-gradient comparisons |
| Volume 211 ×34: all 22 dtype-generic families | [x] | Native weight updates and chain-rule input gradients |
| GPU chain-rule/full-lr primitives | [x] | Hand-calculated Dense chain loss, weights and volume input gradient; invalid nil-loss/empty-stack rejection |
| Benchmark fixture reset | [x] | Leaf dtype sweeps and Dense volume dtype fixtures compare measured-run output with an independent run |

**Validated:** 2026-10-26, unreleased. Exact tensor bits and the finite
loss band apply where specified. GDN leaf weights retain ≥strict; its volume
retains ≥industry. The existing subnormal and KMeans cancellation policies
remain explicit above. KMeans/Parallel now have independent evidence for this
mode. Remaining variants retain their own gates. Residency and performance
tuning remain open; benchmark timings include fixture reset.

## w2a Go↔Metal (TweenSplit ModeTweenSplit) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tweensplit-train`.
The runner includes TweenSplit, ModeTween, TweenAlt and Metal-only regressions.
`TestGoVMetalTweenSplitTrain` contains **131 case groups**: the existing 80 plus
51 KMeans/Parallel groups. Wires: `TSB1`/`TSBO`; existing per-kind Split volumes
plus `KMSP`/`KMSO` and `PNSP`/`PNSO`.

Single collect tape; global 1/N share at full lr. All numerical forward, loss,
projection, 1/N scaling, local backward and native SGD work uses Metal kernels.
Host code collects tapes, schedules leaves and stages buffers. No chain-rule
input gradient is claimed for this mode.

| Evidence | Status | Scope |
|---|:---:|---|
| All 23 supported families | [x] | Native dtype sweeps for the 22 generic families; GDN Float32 only |
| Both KMeans outputs | [x] | Probabilities / Features, Float32 and ×34, including shape-changing KMeans→Dense |
| Seven Parallel combines | [x] | Float32 and ×34; Filter gate counts as a trainable leaf; Concat→Dense width changes |
| Polymorphic Parallel | [x] | All 23 branch kinds Float32, including nested Stack/Parallel and rank-4/rank-5 convolutional branches |
| Stateful Filter | [x] | Metacognition capture and gate update; complete tape before SGD |
| Mixed weightless/trainable leaves | [x] | Dense + Softmax Parallel followed by Dense ×34; Softmax excluded from N |
| Shared volume credit | [x] | All 23 families; KMeans/Parallel 111/chain/211 ×34 and 222 Float32 |
| Benchmark reset | [x] | Fresh loss/weights match measured runs; all new leaf and volume fixtures require exact Metal replay across at least two iterations |
| Collect primitives | [x] | Existing hand-calculated 1/N sharing and tape preservation guards |

Parallel leaf-mode Split recursively collects trainable branches, nested
containers, Sequential/Residual Dense leaves and the Filter gate. It reuses
Tween's collector, including Filter's extra forward and state transitions;
updates begin only after collection. Split loss is scored from the collected
output, whose native branch axes can differ from regular Parallel Forward.
The Go fixture discovers target shapes through Go's public `OpenSplitTape`.
Volume Split retains its existing grid contract: each placed KMeans/Parallel
cell is one local backward/SGD unit; Sequential/Residual expand Dense children.

Existing precision gates apply: finite loss band, exact native weights or the
documented FP32 subnormal policy. GDN leaf/Parallel branch weights retain
≥strict and GDN shared volumes retain ≥industry. No precision gates were
weakened. Other Split/Alt modes retain their separate sign-off sections. Full
residency and throughput tuning remain separate work.

**Validated:** 2026-10-26, unreleased. All 131 TweenSplit groups passed. The
combined run also passed 130 ModeTween groups, 80 TweenAlt groups and Metal-only
guards in 191.609 seconds. Shared/static builds passed.

## w2a Go↔Metal (StepTweenSplit ModeStepTweenSplit) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptweensplit-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `STS1`/`STSO`; shared volume `TSP1` and per-kind Split variants. `TestGoVMetalStepTweenSplitTrain`: **130 case groups**.

Line per-hop Split at full lr; N counts trainable leaves under that hop. The popped global gap stays frozen. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 23 families; 22 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 23 supported families |
| Volume 211 ×34 | [x] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [x] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. GDN loss retains its existing finite `1e-5` relative band, including multi-epoch cache-refresh tests; updated weights still require ≥strict. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel are verified: both KMeans outputs, seven Parallel combines, polymorphic branches, Filter state and shared volumes. The 16 Line Alt edge controls cover individual pipeline pops. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (TweenAlt ModeTweenAlt) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tweenalt-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `TAB1`/`TABO`; volume `TAV1` and per-kind Alt variants. `TestGoVMetalTweenAltTrain`: **130 case groups**.

Split at full lr, refresh prediction and gap, then Tween at lr×0.5. AltTimes=1 is validated. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 23 families; 22 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 23 supported families |
| Volume 211 ×34 | [x] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Collect/phase primitives | [x] | Hand-calculated 1/N sharing and refreshed Alt loss/weights; input tape retained |

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. GDN loss retains its existing finite `1e-5` relative band, including multi-epoch cache-refresh tests; updated weights still require ≥strict. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel are verified: both KMeans outputs, seven Parallel combines, polymorphic branches, Filter state and shared volumes. The 16 Line Alt edge controls cover individual pipeline pops. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (StepTweenAlt ModeStepTweenAlt) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptweenalt-train`  
Batch: `./scripts/macos_build_w2a.sh --tween-next-all` (all eight requested sections + Metal-only checks).

Wires: `STA1`/`STAO`; shared volume `TAV1` and per-kind Alt variants. `TestGoVMetalStepTweenAltTrain`: **130 case groups**.

Line per-hop Split→Tween at full/half lr with a frozen popped gap. Its shared volume path refreshes the gap between phases. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 23 families; 22 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 23 supported families |
| Volume 211 ×34 | [x] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [x] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Validated:** 2026-10-26, unreleased. Exact tensors and the finite Metal loss band apply where specified; GDN leaf weights require ≥strict. GDN loss retains its existing finite `1e-5` relative band, including multi-epoch cache-refresh tests; updated weights still require ≥strict. Volume subnormal weight differences retain the existing policy. Float32 SGD now preserves encoded subnormal steps across phases so normal-range results still pass the exact gate. KMeans/Parallel are verified: both KMeans outputs, seven Parallel combines, polymorphic branches, Filter state and shared volumes. The 16 Line Alt edge controls cover individual pipeline pops. Full graph residency and performance tuning remain open.

## w2a Go↔Metal (MeshTween mesh-clock) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --meshtween-train`.
The runner includes MeshTween, MeshBP, ModeTween and Metal-only regression gates.
`TestGoVMetalMeshTweenTrain` contains **159 case groups**: 48 existing Dense/shared
volume groups, eight KMeans/Parallel volume groups, and 103 all-family clock groups.
Wires: `MGW1`/`MGWO` for actual heterogeneous clock nodes, existing `MTW1`/`MTWO`
for Dense, and the existing per-kind ModeTween volume wires.

Last-tick captured mesh credit applies local projected updates at lr×0.5.
`step::train_mse(FamilyTween)` and the generic clock wire share
`step::apply_tween`; warmup/remote fixtures supply the captured state directly.
All numerical forward, gap projection, backward and native SGD work uses Metal.
Host code schedules ticks, stores tapes, rebuilds fixtures and stages buffers.

| Evidence | Status | Scope |
|---|:---:|---|
| All 23 actual node kinds | [x] | 111 / 211 ×34, plus 222 Float32; GDN Float32 only |
| Recurrent remote clock | [x] | 22 shape-preserving families; detached warmup then captured ticks |
| KMeans output modes | [x] | Probabilities and Features ×34 in three-node chains |
| Parallel combines | [x] | All seven ×34, including Filter gate updates and Concat width changes |
| Shared volume credit | [x] | All 23 families; supported dtype loops, including KMeans/Parallel 111/chain/211 ×34 and 222 Float32 |
| Zero learning rate | [x] | Dense/KMeans/Parallel preserve native weights ×34 |
| Captured tensors and updates | [x] | Every tick/node input, pre and post; final loss/output and all updated weights |
| Exact replay | [x] | Independent fresh Metal result matches at least two measured iterations exactly |
| Credit primitive | [x] | Existing hand-calculated two-cell update distinguishes local Tween from BPTT/full-lr credit |

**Scope:** this is last-tick local mesh credit; no temporal backward pass or
chain-rule input/accumulated parameter gradient is exported. Sequential and
Residual project onto their Dense leaves. Stack and Parallel nodes use their
concrete backward/SGD as local units, matching the existing grid ModeTween
volume contract. Gap copying continues only when the next input length matches;
otherwise the upstream gap is zero. Go references combine Go step capture,
concrete Go backward/SGD and its public local-Tween collector in w2a.
Clock precision policy: finite loss band, exact/native-subnormal weight
checks, GDN tensors ≥loose and weights ≥strict. Shared-volume checks retain
their existing gates, including GDN ≥industry. Full graph residency and
performance tuning remain separate work. MeshTweenChain retains its own gate.

**Validated:** 2026-10-26, unreleased. All 159 MeshTween groups passed; the
combined runner, including 103 MeshBP groups, 130 ModeTween groups and Metal-only
guards, passed in 299.198 seconds. Shared/static builds passed.

## w2a Go↔Metal (MeshTweenChain mesh-clock) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --meshtweenchain-train`.
The runner covers MeshTweenChain, MeshTween, ModeTweenChain and Metal-only guards.
`TestGoVMetalMeshTweenChainTrain` contains **160 case groups**: 48 existing
Dense/shared-volume groups, eight KMeans/Parallel volume groups, 103 all-family
clock groups, and one direct 23-family MeshBP equivalence group.

Wires: `MGC1`/`MGCO` binds actual heterogeneous clock nodes; existing Dense
`MTC1`/`MTCO` and per-kind TweenChain volume wires retain their coverage.
MeshTweenChain uses the existing `runtime/step` FamilyBP path: captured forward
ticks → reverse-time BPTT → accumulated parameter gradients → native SGD at full
lr. Every numerical operation uses Metal kernels; host code schedules ticks,
stores tapes, stages buffers and rebuilds replay fixtures. The wire alias adds
no separate numerical implementation or CPU fallback.

| Evidence | Status | Scope |
|---|:---:|---|
| All 23 actual node kinds | [x] | 111 / 211 ×34 for 22 dtype-generic families; GDN Float32 only; all 23 at 222 Float32 |
| Recurrent remote clock | [x] | 22 shape-preserving families; detached warmup then captured ticks |
| KMeans output modes | [x] | Probabilities / Features ×34 with real Dense adapter nodes |
| Parallel combines | [x] | All seven ×34, including Filter gate gradients and Concat shape changes |
| Captured tensors and gradients | [x] | Every tick/node input, pre and post; final output/loss, oldest input gradient and packed accumulated parameter gradients |
| Full-rate native SGD | [x] | Updated weights for every enabled node; Dense/KMeans/Parallel zero-lr preservation ×34 |
| Exact replay | [x] | Fresh Metal result equals at least two measured repetitions across all exported tensors, weights, tapes and loss |
| MeshBP equivalence | [x] | Independent MeshBP and MeshTweenChain wire calls agree exactly for all 23 families at 211 Float32 |
| Shared volume credit | [x] | All 23 families, including KMeans/Parallel 111/chain/211 ×34 and 222 Float32 |
| Credit primitive | [x] | Existing hand-calculated test distinguishes full-rate BPTT from last-tick half-rate MeshTween |

The existing precision policy applies: finite loss band; exact/subnormal tensor
and weight gates where specified; MHA gradients ≥strict, Softmax gradients ≥loose;
GDN clock tensors ≥loose and weights ≥strict. GDN shared-volume checks retain
≥industry. Only KMeans input gradients use the narrow cancellation allowance.
Parameter gradient axes are flattened for packed comparison; activation and
input-gradient shapes remain strict. Embedding uses one 111 tick and disabled
ingress plus RMSNorm adapter nodes in grids. SwiGLU uses the bounded recurrent
trajectory from section 7. These are correctness fixtures; full residency and
throughput tuning retain separate gates.

**Validated:** 2026-10-26, unreleased. All 160 MeshTweenChain groups passed.
The combined runner also passed 159 MeshTween groups, 130 ModeTweenChain groups
and Metal-only guards in 321.915 seconds. Shared/static builds passed.

## w2a Go↔Metal (StepTween ModeStepTween) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptween-train`  
Batch: `./scripts/macos_build_w2a.sh --step-tween-all` (sections 2–6 + NormalBP + Metal-only checks).

Wires: `STW1`/`STWO`; shared volume `TWV1` and per-kind Tween variants. `TestGoVMetalStepTweenTrain`: **130 case groups**.

Line per-hop projected local credit at lr×0.5, applied when a training flight pops. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 23 families; 22 dtype-generic families ×34, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 23 supported families |
| Volume 211 ×34 | [x] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [x] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Validated:** 2026-10-26, unreleased. Exact native weights and the finite
Metal loss band apply where specified; GDN branches retain ≥strict. The
existing subnormal policy and KMeans volume cancellation band remain explicit.
KMeans/Parallel evidence is included above. Full graph residency and
performance tuning remain open.

## w2a Go↔Metal (StepTweenChain ModeStepTweenChain) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptweenchain-train`  
Batch: `./scripts/macos_build_w2a.sh --step-tween-all` (sections 2–6 + NormalBP + Metal-only checks).

Wires: `STC1`/`STCO`; shared volume `TCV1` and per-kind TweenChain variants. `TestGoVMetalStepTweenChainTrain`: **130 case groups**.

Line per-hop true chain-rule BP at full lr, applied when a training flight pops. Numerical work uses Metal kernels; host code handles scheduling, tapes and staging.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 23 families; 22 dtype-generic families ×34, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 23 supported families |
| Volume 211 ×34 | [x] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [x] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Validated:** 2026-10-26, unreleased. Exact native weights and the finite
Metal loss band apply where specified; GDN branches retain ≥strict. The
existing subnormal policy and KMeans volume cancellation band remain explicit.
KMeans/Parallel evidence is included above. Full graph residency and
performance tuning remain open.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** all 23 NormalBP layer families validated inside nodes and in volumes, including dtype coverage (GDN Float32 only), under the precision policy above. KMeans and Parallel are included, with the declared KMeans cancellation band. The complete NormalBP and Metal-only regression passed 219 case groups in 171.771 seconds; shared and static builds passed. StepBP Line and all-family MeshBP clock evidence are validated below.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** all 23 supported Line families and the shared 3D volume path validated, including all supported dtypes (GDN Float32 only). KMeans/Parallel are included in the independent 50-group extension above. Actual mesh-clock BPTT evidence is Dense-only; see section 7.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** all 23 supported families validated on Metal in leaves and volumes, including all supported dtypes (GDN Float32 only). GDN leaf updated weights retain a ≥strict floor; other declared bands above apply. KMeans/Parallel are included in the independent 50-group extension above; other variants retain separate evidence.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** all 23 supported families validated on Metal in leaves and volumes, including every supported dtype (GDN Float32 only). Full-rate chain-rule credit and volume input gradients are verified. GDN retains its declared precision floors; KMeans/Parallel are included in the independent 50-group extension above. Step/Mesh evidence is recorded separately below.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel are included in the independent 50-group extension above.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel are included in the independent 50-group extension above.

---

## 7 — MeshBP

Grid tick FamilyBP: `runtime/step` Forward×ticks → reverse-time Backward BPTT →
ApplyGradSGD. `MGB1`/`MGBO` binds actual heterogeneous node kinds; existing
`MBP1`/`MBPO` Dense fixtures remain covered. Numerical forward, loss, backward,
gradient joins and native SGD use Metal kernels. Host code schedules ticks,
stores tapes, stages buffers and rebuilds benchmark fixtures.

Run: `./scripts/macos_build_w2a.sh --meshbp-train`.
The runner includes `TestGoVMetalMeshBPTrain` (**103 case groups**), the existing StepBP/volume suite,
NormalBP and the Metal-only guard. Go ModeMeshBP on grid also uses the shared
volume Forward+Step path validated at 222 Float32 and 211 ×34.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Validated:** 2026-10-26, unreleased. The 103 MeshBP case groups passed in
111.72 seconds with at least two measured replay iterations per fixture. The
combined runner passed in 372.500 seconds, including 134 StepBP groups, all
219 NormalBP groups, and the Metal-only guard. Shared and static builds passed.

**Checkpoint:** All 23 supported families have actual mesh-clock node and grid
BPTT evidence. All 22 dtype-generic families cover 111 and 211 ×34; GDN is
Float32-only. Every family also covers 222 Float32. Shape-changing Embedding
uses a single-tick 111 node and disabled ingress plus RMSNorm adapter nodes in
grids; its integer indices are not fed back as embedding outputs. Remote
self-links cover the 22 shape-preserving families with detached warmup followed
by captured ticks. SwiGLU uses a shorter bounded recurrent trajectory to avoid
vanishing intermediate subnormals outside the declared GPU precision contract.

Both KMeans outputs and all seven Parallel combines additionally cover ×34
three-node clock chains, including shape-changing Concat/probabilities→Dense
and Filter gate gradients/updates. Dense/KMeans/Parallel zero-lr cases preserve
native weights across ×34; `step::apply_sgd(0)` is a no-op.

Checks include every captured node input/pre/post, final output/loss, oldest
input gradient, packed accumulated parameter gradients, and post-SGD native
weights. Parameter gradients use a flat packed axis for comparison; activation
and input-gradient shapes remain strict. Existing precision policy applies:
GDN tensors ≥loose and updated weights ≥strict, MHA gradients ≥strict, Softmax
gradients ≥loose, other gates exact or the documented FP32 subnormal policy.
Only KMeans input gradients use its narrow near-zero cancellation allowance.
Fresh and benchmark Metal runs must match exactly across all exported tensors,
loss, weights and tapes. MeshBP measures at least two iterations to exercise
reset between repetitions. Replay timing includes rebuilding/staging; residency
and performance tuning retain separate gates. MeshTween/other modes retain
their own clock sign-off below.

## 8 — MeshTween

Needs Mesh + Tween.  
Ship: `runtime/step` `Credit::FamilyTween` — Forward×ticks → last-tick `apply_tween` / `apply_cell_projected` @ `lr×0.5` (`MGW1`/`MGWO`, existing Dense `MTW1`/`MTWO`); w2a `--meshtween-train`.

**3D volumetric** = ModeTween volume credit (`TWV1`/`TRV1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Validated:** 2026-10-26, unreleased. The combined MeshTween/MeshBP/ModeTween
and Metal-only runner passed in 299.198 seconds. Shared and static builds passed.

**Checkpoint:** All 23 supported families have actual clock and shared-volume
sign-off. The 159 MeshTween groups passed in 122.91 seconds. Clock fixtures
cover 111/211 ×34 for the 22 dtype-generic families and 222 Float32 for all 23;
GDN remains Float32-only. Every captured node input/pre/post, final output/loss
and updated weight is checked. Clock GDN tensors require ≥loose and weights
≥strict; other weights remain exact or use the existing subnormal policy.
Shared volume gates retain their existing floors.

The 103 generic clock groups include recurrent remote links with detached
warmup for 22 shape-preserving families, both KMeans outputs and seven Parallel
combines ×34, and Dense/KMeans/Parallel zero-lr native preservation ×34.
Embedding uses one 111 tick, and disabled ingress plus RMSNorm adapters in grids;
integer indices are not fed back from embedding outputs. SwiGLU uses the bounded
recurrent trajectory documented in section 7. Fresh and at least two measured
Metal replays match exactly across outputs, weights, loss and all captured tapes.

This mode applies local credit only on the last tick. Sequential/Residual
project onto Dense leaves; Stack/Parallel cells are concrete local backward/SGD
units, as in the shared grid ModeTween contract. No temporal BPTT, chain-rule
input gradient or accumulated parameter gradient is claimed. Full residency
and throughput tuning retain separate gates. MeshTweenChain remains section 9.

---

## 9 — MeshTweenChain

Needs Mesh + TweenChain.  
Ship: `runtime/step` `Credit::FamilyBP` under mesh clock (`MGC1`/`MGCO`, existing Dense `MTC1`/`MTCO` ≡ MeshBP); w2a `--meshtweenchain-train`.

**3D volumetric** = ModeTweenChain volume credit (`TCV1`/`TRC1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Validated:** 2026-10-26, unreleased. The combined MeshTweenChain/MeshTween/
ModeTweenChain/Metal-only runner passed in 321.915 seconds. Shared and static
builds passed.

**Checkpoint:** All 23 families have actual clock and shared-volume sign-off,
including KMeans and Parallel. All 160 MeshTweenChain groups passed in 134.60
seconds. The 22 dtype-generic families cover 111/211 ×34; GDN is Float32-only.
All 23 cover 222 Float32. Existing shared-volume gates retain their floors.

Clock evidence includes every tick/node input, pre and post; final loss/output;
oldest input gradients; packed parameter gradients accumulated across ticks;
and full-lr native SGD weights. Recurrent remote links cover 22 shape-preserving
families after detached warmup. Both KMeans output modes and all seven Parallel
combines additionally cover ×34 three-node chains. Dense/KMeans/Parallel zero-lr
cases preserve native weights ×34. Fresh and at least two measured Metal replays
match exactly. A direct 211 Float32 comparison verifies exact MeshBP/MeshTweenChain
wire equivalence for every supported family.

Embedding uses one 111 tick and disabled ingress plus RMSNorm adapters in grids;
its integer indices are not fed back from embedding outputs. SwiGLU uses the
bounded recurrent trajectory from section 7. Existing clock precision gates
apply: finite loss band, exact/subnormal gates, MHA gradients ≥strict, Softmax
gradients ≥loose, GDN tensors ≥loose and weights ≥strict; only KMeans input
gradients use its narrow cancellation allowance. Gradient packing normalizes
only parameter axes; activation/input-gradient shapes remain strict. The named
wire reuses Metal FamilyBP BPTT/SGD; it adds no numerical fallback. Residency
and performance tuning remain separate work.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 families are signed off in leaves and shared volumes,
including both KMeans outputs and all seven Parallel combines. Native ×34
sweeps cover the 22 dtype-generic families; GDN remains Float32-only. The 131
TweenSplit groups passed in 72.47 seconds under the existing precision gates.

Parallel leaf-mode Split counts all trainable branch leaves, nested Stack/Parallel
leaves and its Filter gate before applying projected 1/N gaps at full lr. Softmax
is excluded from N, including the mixed Dense/Softmax ×34 fixture. Capture and
stateful Filter's extra forward finish before any update. Split loss and targets
use the collected output's native axes, matching Go `OpenSplitTape`.
Placed KMeans/Parallel volume cells retain the existing local-unit grid contract.

All 51 new groups passed, including 111/chain/211 ×34 and 222 Float32 volumes,
all 23 polymorphic Parallel branch kinds, stateful Filter, native SGD and exact
fresh-versus-measured replay with at least two iterations. Numerical work uses
Metal kernels; Go references stay in w2a. The combined TweenSplit/ModeTween/
TweenAlt/Metal-only runner passed in 191.609 seconds; shared/static builds passed.
Other Split/Alt mode sign-offs remain separate.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel are covered across native dtypes, nested/polymorphic branches, both outputs and seven combines.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel are covered across native dtypes, nested/polymorphic branches, both outputs and seven combines.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 supported families verified in leaves and shared volumes, including supported dtypes (GDN Float32 only). GDN precision bands above apply; KMeans/Parallel are covered across native dtypes, nested/polymorphic branches, both outputs and seven combines.

---

## 14 — TweenSplitHeadProxy

Head backward/SGD uses full lr; its input gradient is projected to hidden leaves with 1/(N−1) sharing. Dense hidden updates use GradWOnly.

Wires: `GPB1`/`GPBO` (variant 1); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVMetalProxyTrain` and `TestGoVMetalProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Direct Go↔Metal parity and repeated replay passed for all 23 families (34 native dtypes where supported; GDN Float32). Leaf fixtures retain their declared precision gates. Proxy volume loss, captured tensors and native weights require industry-or-better (60–100%) agreement, with actual deviations reported; observed cache/async rounding differences were strict (80%). Exact same-backend replay is checked separately. Numerical projection, credit, norms and SGD run in Metal; host code schedules work and stages tapes. Full graph residency and performance tuning remain open.

---

## 15 — TweenSplitLinear

Reverse linear-credit traversal updates each leaf at 1/N and propagates through the updated weights. Nested Stack, Residual and all Parallel combine routes retain their Go ordering.

Wires: `GPB1`/`GPBO` (variant 2); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVMetalProxyTrain` and `TestGoVMetalProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Direct Go↔Metal parity and repeated replay passed for all 23 families (34 native dtypes where supported; GDN Float32). Leaf fixtures retain their declared precision gates. Proxy volume loss, captured tensors and native weights require industry-or-better (60–100%) agreement, with actual deviations reported; observed cache/async rounding differences were strict (80%). Exact same-backend replay is checked separately. Numerical projection, credit, norms and SGD run in Metal; host code schedules work and stages tapes. Full graph residency and performance tuning remain open.

---

## 16 — TweenSplitFastProxy

Compute a linear head proxy before updating the head, then share it across hidden leaves at 1/(N−1). Dense local updates use GradWOnly.

Wires: `GPB1`/`GPBO` (variant 3); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVMetalProxyTrain` and `TestGoVMetalProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Direct Go↔Metal parity and repeated replay passed for all 23 families (34 native dtypes where supported; GDN Float32). Leaf fixtures retain their declared precision gates. Proxy volume loss, captured tensors and native weights require industry-or-better (60–100%) agreement, with actual deviations reported; observed cache/async rounding differences were strict (80%). Exact same-backend replay is checked separately. Numerical projection, credit, norms and SGD run in Metal; host code schedules work and stages tapes. Full graph residency and performance tuning remain open.

---

## 17 — TweenSplitLinearCache

Linear credit refreshes every 20 epochs. Intervening updates reuse cached projections rescaled by live/cached gap norms; GPU kernels compute norms and rescaling. The tests cross the refresh boundary with 22 epochs.

Wires: `GPB1`/`GPBO` (variant 4); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVMetalProxyTrain` and `TestGoVMetalProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Direct Go↔Metal parity and repeated replay passed for all 23 families (34 native dtypes where supported; GDN Float32). Leaf fixtures retain their declared precision gates. Proxy volume loss, captured tensors and native weights require industry-or-better (60–100%) agreement, with actual deviations reported; observed cache/async rounding differences were strict (80%). Exact same-backend replay is checked separately. Numerical projection, credit, norms and SGD run in Metal; host code schedules work and stages tapes. Full graph residency and performance tuning remain open.

---

## 18 — TweenSplitHeadProxyAsync

Apply the previous proxy to hidden leaves, update the head, then capture the next proxy through updated head weights. The first epoch uses the fresh proxy; four-epoch tests exercise stale reuse.

Wires: `GPB1`/`GPBO` (variant 5); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVMetalProxyTrain` and `TestGoVMetalProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Direct Go↔Metal parity and repeated replay passed for all 23 families (34 native dtypes where supported; GDN Float32). Leaf fixtures retain their declared precision gates. Proxy volume loss, captured tensors and native weights require industry-or-better (60–100%) agreement, with actual deviations reported; observed cache/async rounding differences were strict (80%). Exact same-backend replay is checked separately. Numerical projection, credit, norms and SGD run in Metal; host code schedules work and stages tapes. Full graph residency and performance tuning remain open.

---

## 19 — TweenSplitSparse

Update the head and one rotating hidden leaf per epoch, using the pre-update linear head proxy. Four-epoch tests exercise rotation and state reset.

Wires: `GPB1`/`GPBO` (variant 6); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVMetalProxyTrain` and `TestGoVMetalProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Direct Go↔Metal parity and repeated replay passed for all 23 families (34 native dtypes where supported; GDN Float32). Leaf fixtures retain their declared precision gates. Proxy volume loss, captured tensors and native weights require industry-or-better (60–100%) agreement, with actual deviations reported; observed cache/async rounding differences were strict (80%). Exact same-backend replay is checked separately. Numerical projection, credit, norms and SGD run in Metal; host code schedules work and stages tapes. Full graph residency and performance tuning remain open.

---

## 20 — MeshTweenSplit

Mesh-clock Split applies full-lr 1/N local credit to each enabled cell using the final captured tick. The gap remains frozen and passes to earlier cells only when input lengths agree.

Wires: `MGS1`/`MGSO`. `TestGoVMetalMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Passed 2026-10-26: all 23 actual node kinds, supported native dtypes (GDN Float32), shared volumes and exact repeated replay. Mesh Split/Alt each have 154 groups; FastProxy/Sparse each have 172. All four passed together in 646.21s. Numerical work runs in Metal; graph residency and performance tuning remain open.

---

## 21 — MeshTweenAlt

Mesh-clock Alt applies Split then half-lr Tween to the final captured local tape with the frozen popped gap. Shared volume Alt separately refreshes prediction/gap between phases.

Wires: `MGA1`/`MGAO`. `TestGoVMetalMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Passed 2026-10-26: all 23 actual node kinds, supported native dtypes (GDN Float32), shared volumes and exact repeated replay. Mesh Split/Alt each have 154 groups; FastProxy/Sparse each have 172. All four passed together in 646.21s. Numerical work runs in Metal; graph residency and performance tuning remain open.

---

## 22 — MeshTweenSplitFastProxy

Mesh-clock FastProxy applies per-cell pre-update head proxy credit on the final captured tick. Shared volume FastProxy uses the global collected volume tape.

Wires: `MGF1`/`MGFO`. `TestGoVMetalMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Passed 2026-10-26: all 23 actual node kinds, supported native dtypes (GDN Float32), shared volumes and exact repeated replay. Mesh Split/Alt each have 154 groups; FastProxy/Sparse each have 172. All four passed together in 646.21s. Numerical work runs in Metal; graph residency and performance tuning remain open.

---

## 23 — MeshTweenSplitSparse

Mesh-clock Sparse updates the head and one hidden leaf per eligible local unit. Stack/Parallel nodes own rotation state; shared volume Sparse owns grid state.

Wires: `MGQ1`/`MGQO`. `TestGoVMetalMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** Passed 2026-10-26: all 23 actual node kinds, supported native dtypes (GDN Float32), shared volumes and exact repeated replay. Mesh Split/Alt each have 154 groups; FastProxy/Sparse each have 172. All four passed together in 646.21s. Numerical work runs in Metal; graph residency and performance tuning remain open.

---

## 24 — StepTweenSplitHeadProxy

Line HeadProxy applies full-lr head backward/SGD on pop, then projects head input credit across hidden leaves. The popped global gap stays frozen. Shared volume coverage uses the section 14 HeadProxy path.

Wire: `STH1`/`STHO`. `TestGoVMetalProxyTrain/StepTweenSplitHeadProxy` covers 65 leaf/Line groups, including native CNN2/3 and ConvT2/3. Section 14 supplies its shared volume evidence. GDN is Float32 only.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 families passed Line HeadProxy and shared volume parity, including supported dtype sweeps, pipeline warmup/pops and replay. Numerical training runs in Metal; host code schedules and stages tapes. Line variants 25–29 are signed off below.

---

## 25 — StepTweenSplitLinear

Line per-hop credit on pop uses the shared Metal `systems::tween_split::apply_variant_child` implementation (variant 2). The popped gap stays frozen. Reverse local linear credit uses updated weights and 1/N sharing.

Wire: `STL1`/`STLO`. Tests: `TestGoVMetalStepProxyTrain/StepTweenSplitLinear`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 15 uses `PGB1`/`PGVO` variant 2 and `TestGoVMetalProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/macos_build_w2a.sh --training-25-29-all`.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 families and supported FormatNone dtypes passed (GDN Float32 only). Line loss/weights use the accepted 60–100% determinism spectrum; shapes/counts and same-backend replay retain their checks. Four Filter fixtures cover warmup and individual pipeline pops. Shared volume evidence comes from the corresponding parent proxy suite. Numerical training remains Metal-only; residency and performance tuning are separate work.

---

## 26 — StepTweenSplitFastProxy

Line per-hop credit on pop uses the shared Metal `systems::tween_split::apply_variant_child` implementation (variant 3). The popped gap stays frozen. The local head proxy is captured before the head update, then shared across hidden leaves.

Wire: `STF1`/`STFO`. Tests: `TestGoVMetalStepProxyTrain/StepTweenSplitFastProxy`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 16 uses `PGB1`/`PGVO` variant 3 and `TestGoVMetalProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/macos_build_w2a.sh --training-25-29-all`.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 families and supported FormatNone dtypes passed (GDN Float32 only). Line loss/weights use the accepted 60–100% determinism spectrum; shapes/counts and same-backend replay retain their checks. Four Filter fixtures cover warmup and individual pipeline pops. Shared volume evidence comes from the corresponding parent proxy suite. Numerical training remains Metal-only; residency and performance tuning are separate work.

---

## 27 — StepTweenSplitLinearCache

Line per-hop credit on pop uses the shared Metal `systems::tween_split::apply_variant_child` implementation (variant 4). The popped gap stays frozen. Eligible Parallel hops retain cached projections and GPU-produced norms between pops. Dense hops have no accelerator state. Seven 26-tick fixtures cross the 20-pop refresh boundary for every Parallel combine.

Wire: `SLC1`/`SLCO`. Tests: `TestGoVMetalStepProxyTrain/StepTweenSplitLinearCache`; 76 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 17 uses `PGB1`/`PGVO` variant 4 and `TestGoVMetalProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/macos_build_w2a.sh --training-25-29-all`.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 families and supported FormatNone dtypes passed (GDN Float32 only). Line loss/weights use the accepted 60–100% determinism spectrum; shapes/counts and same-backend replay retain their checks. Four Filter fixtures cover warmup and individual pipeline pops. Shared volume evidence comes from the corresponding parent proxy suite. Numerical training remains Metal-only; residency and performance tuning are separate work.

---

## 28 — StepTweenSplitHeadProxyAsync

Line per-hop credit on pop uses the shared Metal `systems::tween_split::apply_variant_child` implementation (variant 5). The popped gap stays frozen. Eligible Parallel hops retain their previous hidden proxy between pops and capture the next proxy after updating the head. Dense hops have no accelerator state.

Wire: `SHA1`/`SHAO`. Tests: `TestGoVMetalStepProxyTrain/StepTweenSplitHeadProxyAsync`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 18 uses `PGB1`/`PGVO` variant 5 and `TestGoVMetalProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/macos_build_w2a.sh --training-25-29-all`.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 families and supported FormatNone dtypes passed (GDN Float32 only). Line loss/weights use the accepted 60–100% determinism spectrum; shapes/counts and same-backend replay retain their checks. Four Filter fixtures cover warmup and individual pipeline pops. Shared volume evidence comes from the corresponding parent proxy suite. Numerical training remains Metal-only; residency and performance tuning are separate work.

---

## 29 — StepTweenSplitSparse

Line per-hop credit on pop uses the shared Metal `systems::tween_split::apply_variant_child` implementation (variant 6). The popped gap stays frozen. Eligible Parallel hops rotate the selected hidden leaf between pops. Dense hops have no accelerator state; the no-state hidden selection follows Go.

Wire: `SSP1`/`SSPO`. Tests: `TestGoVMetalStepProxyTrain/StepTweenSplitSparse`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 19 uses `PGB1`/`PGVO` variant 6 and `TestGoVMetalProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/macos_build_w2a.sh --training-25-29-all`.

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
| KMeans | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Mamba | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Metacognition | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| GDN | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Parallel | [x] | [x] | [x] | 2026-10-26 (unreleased) |
| Stack | [x] | [x] | [x] | 2026-10-26 (unreleased) |

**Checkpoint:** All 23 families and supported FormatNone dtypes passed (GDN Float32 only). Line loss/weights use the accepted 60–100% determinism spectrum; shapes/counts and same-backend replay retain their checks. Four Filter fixtures cover warmup and individual pipeline pops. Shared volume evidence comes from the corresponding parent proxy suite. Numerical training remains Metal-only; residency and performance tuning are separate work.

---

## Version log

The older tags below preserve historical wire milestones, some of which used
host computation. They do not establish present GPU sign-off. The dated
unreleased validation and checked tables above describe current Metal evidence.

| Tag | Notes |
|---|---|
| 0.68 — Cameral / CamSync | Freeze/Shadow/Adversarial/Memory, mixed branches, recursive Stack overrides and GPU store blending. 465 training and 109 sync groups, supported native dtypes, exact reset replay and independent semantic guards. Individual subgate version audit aligns accounting with the CPU sibling; unsupported model/quant/residency work stays open. |
| Unreleased — sections 25–29 | Finished all five Line proxy variants using shared Metal credit, GPU norm/cache rescaling and native SGD. 352 new all-family/native-dtype groups, pipeline-pop controls and all-combine cache-refresh fixtures passed; shared/static builds and GPU-only guard passed. Shared volumes reuse sections 15–19. CPU↔GPU parity accepts the 60–100% project spectrum. Full runner: `--training-all`. |
| Unreleased — sections 11–24 | All 23 families signed off in StepTweenSplit, TweenAlt, StepTweenAlt, six proxy/cache/sparse modes, four mesh-clock modes and Line HeadProxy. Restored missing section headings 14–23. Direct native-dtype, stateful replay, cache-refresh and pipeline-pop evidence is in the combined `--training-11-24-all` runner; shared/static builds passed. |
| Unreleased — TweenSplit section 10 | All 23 families: 131 groups, including 51 new KMeans/Parallel groups. Both output modes, seven combines, nested/polymorphic capture, Filter gate/state, weightless-leaf exclusion, native SGD and exact repeated leaf/volume replay. Combined TweenSplit/ModeTween/TweenAlt/Metal-only run passed in 191.609s; shared/static builds passed. |
| Unreleased — MeshTweenChain section 9 | All 23 actual node kinds: 160 clock/shared-volume groups, supported dtypes, recurrent links, both KMeans outputs, seven Parallel combines, full-rate BPTT/SGD and exact replay. MeshBP equivalence checked directly for every family. Combined MeshTweenChain/MeshTween/ModeTweenChain/Metal-only run passed in 321.915s; shared/static builds passed. |
| Unreleased — MeshTween section 8 | All 23 actual node kinds, both KMeans outputs and all Parallel combines: 159 clock/shared-volume groups; final-tick half-lr local credit, recurrent remote links, zero-lr native preservation and exact replay. MeshTween/MeshBP/ModeTween/Metal-only runner passed in 299.198s; shared/static builds passed. |
| Unreleased — MeshBP section 7 | All 23 actual node kinds: 103 clock groups, 111/211 supported dtypes and 222 Float32; recurrent remote links, both KMeans outputs, all Parallel combines, zero-lr native preservation and exact two-iteration replay. Combined MeshBP/StepBP/NormalBP/Metal-only runner passed in 372.500s; shared/static builds passed. |
| Unreleased — 2026-10-26 | Metal-only NormalBP: all 17 evidence sections / 128 cases passed, covering 21 layer families and their supported dtypes. GPU native encoding/bias SGD, all Dense activations/derivatives, and primitive/cache/no-host-API checks passed. StepBP Line: all 21 supported families / dtypes; MeshBP: Dense mesh-clock ×34 and all supported volume families. Disabled/remote routing and benchmark reset checks passed. All-family MeshBP clock parity is now recorded in section 7; other mesh modes retain separate gates. ModeTween: 80 case groups, supported leaf/volume dtypes, Metal projection/local credit and benchmark reset checks passed. ModeTweenChain: 80 groups, supported leaf/volume dtypes, full-lr chain-rule credit and true input gradients validated. Eight requested Split/Alt/Step/Mesh sections: 576 groups; tiny-SGD preservation and credit/phase primitives verified. |
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



## Mixed Go / C++ / Metal routing — sign-off evidence

- [x] JSON `metal` resolves to a real Metal backend; no silent Go fallback.
- [x] Explicit per-child placements survive Stack/Parallel inheritance.
- [x] Incoming global gradients, dInput/dWeights and native SGD cross segment boundaries.
- [x] Dense proxy primitives use Metal when their layer is assigned Metal.
- [x] 13-layer Go 0–3 → C++ 4–8 → Metal 9–12 forward/backward/update comparison.
- [x] Grid (0,0,0), (1,1,1), (2,2,2) placement and existing remote-link semantics.
- [x] Routed backward/SGD for all 23 baseline families over supported native dtypes (GDN Float32).
- [x] Persistent homogeneous MSE mode delegation and mixed Go/C++/Metal local training modes.
- [x] Routed weights return automatically for Go execution/saving; explicit sessions keep explicit SyncToGo.
- [x] Router lifecycle, missing-library errors and pending C++ parameter handoff.

Run `./scripts/macos_build_w2a.sh --go-routing`; see [GO_INTEGRATION.md](GO_INTEGRATION.md).
