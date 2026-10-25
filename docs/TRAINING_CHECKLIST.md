# Training checklist — welvet.metal

One **TrainMode** at a time. Finish that mode’s table (or the rows you care about), bump version, move on.

Canonical enum: Go `layers/parallel/train_mode.go` (`TrainMode`).  
`inherit` (0) = resolve-only — not a ship target.

**Port mirror of** [`welvet.cpp/docs/TRAINING_CHECKLIST.md`](../../welvet.cpp/docs/TRAINING_CHECKLIST.md). Same cases / wires / column meanings; status here is **Metal** only. cpp `[ ]` does **not** transfer credit.

---

## Metal-only mandate (non-negotiable)

**welvet.metal = GPU Metal compute only.** Host/CPU math belongs in **welvet.cpp** (and Go).  
Any host loop left in this tree is **tech debt** until the ✓/✗ table below is all ✓.

Current executable contract: [METAL_ONLY.md](METAL_ONLY.md).

| Path | Execution today |
|---|---|
| Dense forward / backward | Metal kernels; FormatNone staging supports all 34 dtypes |
| MSE / MSEGrad | Metal kernels; loss tolerance described in the execution contract |
| SGD | Metal kernel, **Float32 FormatNone only** |
| Sequential / NestedStack | Host scheduling and copies; supported Dense compute runs on Metal |
| Mesh gradient joins | Metal accumulation kernel |
| All other layer forward / backward / updates | `Unsupported`; CPU implementation retired |
| Non-Float32 SGD / float64 bias SGD | `Unsupported`; no CPU update fallback |

The tables below retain historical wire/case coverage for the port backlog.
All former sign-offs are reset: parity against CPU implementations was not proof
of GPU training. Re-sign each row only after its Metal kernel and w2a evidence land.

---

### Columns (every table)

| Column | Means |
|---|---|
| **Inside node** | Layer trains correctly in a `1×1×1` cell (single-node / ordinary NN case). go_v_metal / exact_bit (or declared band). |
| **3D volumetric** | Same mode works when that layer sits in a multi-cell Grid (volume Forward/Backward + credit across cells). |
| **Fully signed off** | **All** supported FormatNone dtypes for that layer under this mode (not just f32). Softmax = dX only. GDN = f32-only in Go → sign off when f32 path is done. |

Tick `[ ]` when done. Put the **release tag** in **Ver** when that row (or the whole mode table) is signed.

**Current engine:** use the GPU-only gate in w2a:
`go test ./tests/go_v_metal -run '^TestGoVMetalOnly$' -count=1 -v`.
The historical broad suites deliberately remain available; unported rows now fail
with an explicit unsupported status rather than executing on the CPU.

---

## w2a Go↔Metal (Dense NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --dense-train`  

Wire: `WDB1`/`WDBO` (leaf), `WVN1`/`WVNO` (volume). Exact_bit vs Go.  
Timing: warm-up + `--bench=N` (in-process; no spawn) → **goµs / mtlµs / spd%Δ** (`(mtl−go)/go×100`, neg ⇒ Metal faster). Resident soft-double W (no remount each op); microbench spd%Δ is GPU-launch-bound vs Go host.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 ReLU (dX/dW/W/loss) | [ ] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [ ] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [ ] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [ ] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [ ] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [ ] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [ ] | multi-cell ×34 + speed table |

## w2a Go↔Metal (RMSNorm NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --rmsnorm-train`  
Wire: `WRB1`/`WRBO` (leaf), `WRV1`/`WRVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention as Dense (leaf mount-once + γ reset; no remount each iter).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγ/γ/loss) | [ ] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [ ] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [ ] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [ ] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [ ] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [ ] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [ ] | multi-cell ×34 + speed table |

## w2a Go↔Metal (LayerNorm NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --layernorm-train`  
Wire: `WLB1`/`WLBO` (leaf), `WLV1`/`WLVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + γ/β reset).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγβ/γβ/loss) | [ ] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [ ] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [ ] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [ ] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [ ] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [ ] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [ ] | multi-cell ×34 + speed table |

## w2a Go↔Metal (SwiGLU NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --swiglu-train`  
Wire: `WSB1`/`WSBO` (leaf), `WSV1`/`WSVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [ ] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [ ] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [ ] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [ ] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [ ] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [ ] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [ ] | multi-cell ×34 + speed table |

## w2a Go↔Metal (Softmax NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --softmax-train`  
Wire: `WXB1`/`WXBO` (leaf), `WXV1`/`WXVO` (volume). Softmax = dX only (weightless).  
Timing: leaf mount-once (reuse layer; no W). Band: **≥loose** floor (Softmax hard to bit-match); leaf×34 / volume hit **strict (~80%)**; Kind×34 worst **loose**; Gumbel = smoke.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Standard Float32 (dX/loss) | [ ] | strict + speed |
| 2 | Leaf all Kind Float32 (+ gumbel smoke) | [ ] | ≥loose; gumbel non-det |
| 3 | Leaf Standard × all 34 dtypes | [ ] | strict ×34 + speed |
| 4 | Leaf all Kind × all 34 dtypes | [ ] | ≥loose (worst=loose) |
| 5 | Volume 1×1×1×1 Float32 | [ ] | inside-node strict |
| 6 | Volume 1×1×1×2 chain Float32 | [ ] | multi-layer strict |
| 7 | Volume 1×1×1×1 × all 34 dtypes | [ ] | strict ×34 |
| 8 | Volume 2×2×2×1 Float32 | [ ] | 3D volumetric strict |
| 9 | Volume 2×1×1×1 × all 34 dtypes | [ ] | multi-cell strict ×34 |

## w2a Go↔Metal (Embedding NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --embedding-train`  
Wire: `WEB1`/`WEBO` (leaf), `WEV1`/`WEVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + W reset). dX zeros (discrete IDs); dW scatter.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [ ] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [ ] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [ ] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [ ] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [ ] | + spectrum |
| 6 | Volume 2×2×2×1 Float32 | [ ] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [ ] | multi-cell ×34 |

## w2a Go↔Metal (MHA NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --mha-train`  
Wire: `WHB1`/`WHBO` (leaf), `WHV1`/`WHVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention (resident Dense Q/K/V/O; microbench spd%Δ is launch-bound). ModeSelf baseline (no dropout/QKNorm/cross).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf DecoderCausal Float32 (dX/dW/W/loss) | [ ] | exact_bit + speed |
| 2 | Leaf GQA + Encoder Float32 | [ ] | exact_bit |
| 3 | Leaf DecoderCausal × all 34 dtypes | [ ] | W exact_bit; dX/dW ≥strict (uint2 dW ~1 ULP) |
| 4 | Volume 1×1×1×1 Float32 | [ ] | inside-node + speed |
| 5 | Volume 1×1×1×2 chain Float32 | [ ] | multi-layer cell + speed |
| 6 | Volume 1×1×1×1 × all 34 dtypes | [ ] | + spectrum |
| 7 | Volume 2×2×2×1 Float32 | [ ] | 3D volumetric + speed |
| 8 | Volume 2×1×1×1 × all 34 dtypes | [ ] | multi-cell ×34 |

## w2a Go↔Metal (CNN1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --cnn-train`  
Wire: `C1B1`/`C1BO`, `C2B1`/`C2BO`, `C3B1`/`C3BO` (leaf); `C*V1`/`C*VO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + proj W reset). im2col → Dense bwd → col2im.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 CNN1/2/3 | [ ] | exact_bit + speed |
| 4–6 | Leaf ×34 CNN1/2/3 | [ ] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [ ] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [ ] | scaled init for 8-hop |

## w2a Go↔Metal (ConvT1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --convt-train`  
Wire: `T1B1`/`T1BO`, `T2B1`/`T2BO`, `T3B1`/`T3BO` (leaf); `T*V1`/`T*VO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + proj W reset). Scatter forward + act Jacobian + scatter adjoint (no bias).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 ConvT1/2/3 | [ ] | exact_bit + speed |
| 4–6 | Leaf ×34 ConvT1/2/3 | [ ] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [ ] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [ ] | scaled init for 8-hop |

## w2a Go↔Metal (RNN NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --rnn-train`  
Wire: `RNB1`/`RNBO` (leaf); `RNV1`/`RNVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. BPTT via Dense IH/HH + tanh′; loom `[dIH|dHH|dBias]`.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [ ] | exact_bit + speed |
| 2 | Leaf ×34 | [ ] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [ ] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [ ] | scaled init for 8-hop |

## w2a Go↔Metal (LSTM NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --lstm-train`  
Wire: `LSB1`/`LSBO` (leaf); `LSV1`/`LSVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. BPTT 4-gate Dense; loom `[i|f|g|o]` packs.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [ ] | exact_bit + speed |
| 2 | Leaf ×34 | [ ] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [ ] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [ ] | scaled init for 8-hop |

## w2a Go↔Metal (Sequential NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --sequential-train`  
Wire: `SQB1`/`SQBO` (leaf); `SQV1`/`SQVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. Reverse Dense Dim→Dim chain; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [ ] | exact_bit + speed |
| 2 | Leaf ×34 | [ ] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [ ] | shape-preserving Dim |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [ ] | scaled init for 8-hop |

## w2a Go↔Metal (Residual NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --residual-train`  
Wire: `REB1`/`REBO` (leaf); `REV1`/`REVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. F reverse Dense + skip grad; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [ ] | exact_bit + speed |
| 2 | Leaf ×34 | [ ] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [ ] | y=F(x)+x |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [ ] | scaled init for 8-hop |

## w2a Go↔Metal (Mamba NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --mamba-train`  
Wire: `MBB1`/`MBBO` (leaf); `MBV1`/`MBVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. Scan BPTT + In/Out Dense; loom In|Out|ALog|DSkip.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [ ] | exact_bit + speed |
| 2 | Leaf ×34 | [ ] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [ ] | shape-preserving D |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [ ] | scaled init for 8-hop |

## w2a Go↔Metal (Metacognition NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --metacognition-train`  
Wire: `MCB1`/`MCBO` (leaf); `MCV1`/`MCVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. lastGate-scale + Observed Dense; loom [Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [ ] | exact_bit + speed |
| 2 | Leaf ×34 | [ ] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [ ] | default cool rules |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [ ] | scaled init for 8-hop |

## w2a Go↔Metal (GDN NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --gdn-train`  
Wire: `GDB1`/`GDBO` (leaf); `GDV1`/`GDVO` (volume). ≥loose leaf / ≥industry volume vs Go (host f32 scan FP; Metal≡cpp).  
Timing: same `--bench=N` / spd%Δ convention. Truncated-BPTT (state/conv stop-grad); f32-only (no ×34).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [ ] | ≥loose (dW floor) + speed |
| 2 | Volume 1×1×1×1 | [ ] | ≥industry |
| 3 | Volume 1×1×1×2 chain | [ ] | ≥industry |
| 4 | Volume 2×2×2×1 | [ ] | ≥industry; scaled init |

## w2a Go↔Metal (Stack NestedStack-of-Dense NormalBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --stack-train`  
Wire: `STKB`/`STKO` (leaf); `STKV`/`STVO` (volume). Exact_bit vs Go.  
Timing: same `--bench=N` / spd%Δ convention. NestedStack of Dense Dim→Dim Linear; loom n_children×[Dim×Dim]. Avoids `STB1` (StepBP).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [ ] | exact_bit + speed |
| 2 | Leaf ×34 | [ ] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [ ] | NestedStack Dense |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [ ] | binary 211 post denorm ≤1e-18 OK |

---

## w2a Go↔Metal (StepBP Line + MeshBP) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --stepbp-train`  
Wires: `STB1`/`STBO` (Line StepBP); `MBP1`/`MBPO` (mesh-clock MeshBP FamilyBP). Exact_bit vs Go (GDN Line ≥strict scan FP).  
Each Line ×34 kind prints **dtype / band / exact%W / goµs / mtlµs / spd%Δ**.  
3D: Dense mesh-clock 1×1×1×1 / 2×1×1×1 / 2×2×2×1 + ModeMeshBP volume 2×2×2×1 for all NormalBP-signed kinds (Go cameral ModeMeshBP = volume Forward+Step). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path. KMeans Line deferred (NBP open).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Dense multi-hop Float32 (W/loss) | [ ] | warm-up ticks until first pop |
| 2 | Dense 1-hop ×34 | [ ] | spectrum + speed table |
| 3–4 | RMSNorm f32 + ×34 | [ ] | full spectrum + speed |
| 5–6 | LayerNorm f32 + ×34 | [ ] | full spectrum + speed |
| 7–8 | Softmax f32 + ×34 | [ ] | full spectrum + speed |
| 9–10 | SwiGLU f32 + ×34 | [ ] | full spectrum + speed |
| 11–12 | Sequential f32 + ×34 | [ ] | full spectrum + speed |
| 13–14 | Residual f32 + ×34 | [ ] | full spectrum + speed |
| 15–16 | Mamba f32 + ×34 | [ ] | full spectrum + speed |
| 17–18 | Embedding f32 + ×34 | [ ] | full spectrum + speed |
| 19–20 | MHA f32 + ×34 | [ ] | full spectrum + speed |
| 21–22 | CNN1 f32 + ×34 | [ ] | full spectrum + speed (CNN2/3 share) |
| 23–24 | ConvT1 f32 + ×34 | [ ] | full spectrum + speed (ConvT2/3 share) |
| 25–26 | RNN f32 + ×34 | [ ] | full spectrum + speed |
| 27–28 | LSTM f32 + ×34 | [ ] | full spectrum + speed |
| 29–31 | Dense MeshBP mesh-clock 111/211/222 | [ ] | MBP1 exact_bit + speed |
| 32–49 | MeshBP volume 2×2×2×1 all signed kinds | [ ] | ModeMeshBP = volume Step |

## w2a Go↔Metal (Tween ModeTween) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tween-train`  
Wires: `TWB1`/`TWBO` (Stack ModeTween leaf); `TWV1`/`TWVO` + `TRV1`…`MBTV` (volume reverse-hop projected local @ lr×0.5). Exact_bit vs Go.  
Each leaf ×34 kind prints **dtype / band / exact%W / goµs / mtlµs / spd%Δ**.  
3D: Dense 111/211/222 + ModeTween volume 2×2×2×1 for all NormalBP-signed kinds. CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Dense multi-hop Float32 (W/loss) | [ ] | ModeTween soft_lr=lr×0.5 |
| 2 | Dense 1-hop ×34 | [ ] | spectrum + speed table |
| 3–4 | RMSNorm f32 + ×34 | [ ] | full spectrum + speed |
| 5–6 | LayerNorm f32 + ×34 | [ ] | full spectrum + speed |
| 7–8 | Softmax f32 + ×34 | [ ] | full spectrum + speed |
| 9–10 | SwiGLU f32 + ×34 | [ ] | full spectrum + speed |
| 11–12 | Sequential f32 + ×34 | [ ] | leaf-projected (no chain) |
| 13–14 | Residual f32 + ×34 | [ ] | leaf-projected (no chain) |
| 15–16 | Mamba f32 + ×34 | [ ] | full spectrum + speed |
| 17–18 | Embedding f32 + ×34 | [ ] | full spectrum + speed |
| 19–20 | MHA f32 + ×34 | [ ] | full spectrum + speed |
| 21–22 | CNN1 f32 + ×34 | [ ] | full spectrum + speed (CNN2/3 share) |
| 23–24 | ConvT1 f32 + ×34 | [ ] | full spectrum + speed (ConvT2/3 share) |
| 25–26 | RNN f32 + ×34 | [ ] | full spectrum + speed |
| 27–28 | LSTM f32 + ×34 | [ ] | full spectrum + speed |
| 29–31 | Dense Tween volume 111/211/222 | [ ] | TWV1 exact_bit + speed |
| 32–48 | ModeTween volume 2×2×2×1 all signed kinds | [ ] | TRV1…MBTV / C1TV…; reverse-hop @ lr×0.5 |

## w2a Go↔Metal (TweenChain ModeTweenChain) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tweenchain-train`  
Wires: `TCB1`/`TCBO` (Stack ModeTweenChain leaf = FamilyBP @ full lr); `TCV1`/`TCVO` + `TRC1`…`MBTC` (volume Step = NormalBP credit). Exact_bit vs Go.  
Each leaf ×34 kind prints **dtype / band / exact%W / goµs / mtlµs / spd%Δ**.  
3D: Dense 111/211/222 + ModeTweenChain volume 2×2×2×1 for all NormalBP-signed kinds. CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Dense multi-hop Float32 (W/loss) | [ ] | ModeTweenChain = BP @ full lr |
| 2 | Dense 1-hop ×34 | [ ] | spectrum + speed table |
| 3–4 | RMSNorm f32 + ×34 | [ ] | full spectrum + speed |
| 5–6 | LayerNorm f32 + ×34 | [ ] | full spectrum + speed |
| 7–8 | Softmax f32 + ×34 | [ ] | full spectrum + speed |
| 9–10 | SwiGLU f32 + ×34 | [ ] | full spectrum + speed |
| 11–12 | Sequential f32 + ×34 | [ ] | full chain FamilyBP |
| 13–14 | Residual f32 + ×34 | [ ] | full chain FamilyBP |
| 15–16 | Mamba f32 + ×34 | [ ] | full spectrum + speed |
| 17–18 | Embedding f32 + ×34 | [ ] | full spectrum + speed |
| 19–20 | MHA f32 + ×34 | [ ] | full spectrum + speed |
| 21–22 | CNN1 f32 + ×34 | [ ] | full spectrum + speed (CNN2/3 share) |
| 23–24 | ConvT1 f32 + ×34 | [ ] | full spectrum + speed (ConvT2/3 share) |
| 25–26 | RNN f32 + ×34 | [ ] | full spectrum + speed |
| 27–28 | LSTM f32 + ×34 | [ ] | full spectrum + speed |
| 29–31 | Dense TweenChain volume 111/211/222 | [ ] | TCV1 exact_bit + speed |
| 32–48 | ModeTweenChain volume 2×2×2×1 all signed kinds | [ ] | TRC1…MBTC / C1TC…; volume Step |

## w2a Go↔Metal (TweenSplit ModeTweenSplit) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tweensplit-train`  
Wires: `TSB1`/`TSBO` (Stack 1/N @ full lr); `TSP1`/`TSPO` + `TRP1`…`MBTP` (volume leaf collect 1/N). Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Leaf kinds ×34 | [ ] | FamilyTweenSplit OpenSplitTape twin |
| 29–48 | Volume 2×2×2×1 all signed kinds | [ ] | TSP1/TRP1… collect leaves 1/N |

## w2a Go↔Metal (StepTweenSplit ModeStepTweenSplit) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptweensplit-train`  
Wires: `STS1`/`STSO` (Line FamilyTweenSplit 1/N @ full lr on pop); 3D via TweenSplit volume wires `TSP1`/`TRP1`… (MeshTweenSplit volume credit). Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Line leaf kinds ×34 | [ ] | FamilyTweenSplit per hop |
| 29–48 | MeshTweenSplit volume 2×2×2×1 | [ ] | = ModeTweenSplit volume wires |

## w2a Go↔Metal (TweenAlt ModeTweenAlt) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --tweenalt-train`  
Wires: `TAB1`/`TABO` (Split→Tween AltTimes=1); `TAV1`/`TRA1`…`MBTA` volume. Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Leaf kinds ×34 | [ ] | Split @ full lr then Tween @ lr×0.5 |
| 29–48 | Volume 2×2×2×1 all signed kinds | [ ] | re-forward between phases |

## w2a Go↔Metal (StepTweenAlt ModeStepTweenAlt) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptweenalt-train`  
Wires: `STA1`/`STAO` (Line FamilyTweenAlt Split→Tween on pop); 3D via TweenAlt volume wires `TAV1`/`TRA1`… (MeshTweenAlt volume credit). Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Line leaf kinds ×34 | [ ] | trainTweenAltOp frozen gap per hop |
| 29–48 | MeshTweenAlt volume 2×2×2×1 | [ ] | = ModeTweenAlt volume wires |

## w2a Go↔Metal (MeshTween mesh-clock) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --meshtween-train`  
Wires: `MTW1`/`MTWO` (mesh-clock FamilyTween @ lr×0.5); 3D via ModeTween volume `TWV1`/`TRV1`…. Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Dense MeshTween mesh-clock 111/211/222 | [ ] | MTW1 exact_bit + speed |
| 4–23 | MeshTween volume 2×2×2×1 all signed kinds | [ ] | = ModeTween volume wires |

## w2a Go↔Metal (MeshTweenChain mesh-clock) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --meshtweenchain-train`  
Wires: `MTC1`/`MTCO` (mesh-clock FamilyBP ≡ MeshBP); 3D via ModeTweenChain volume `TCV1`/`TRC1`…. Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Dense MeshTweenChain mesh-clock 111/211/222 | [ ] | MTC1 exact_bit + speed |
| 4–23 | MeshTweenChain volume 2×2×2×1 all signed kinds | [ ] | = ModeTweenChain volume wires |

## w2a Go↔Metal (StepTween ModeStepTween) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptween-train`  
Wires: `STW1`/`STWO` (Line FamilyTween on pop @ lr×0.5); 3D via ModeTween volume wires `TWV1`/`TRV1`… (MeshTween volume credit). Exact_bit vs Go.  
Each leaf ×34 kind prints **dtype / band / exact%W / goµs / mtlµs / spd%Δ**.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Dense multi-hop Float32 (W/loss) | [ ] | Line StepTween soft_lr=lr×0.5 |
| 2 | Dense 1-hop ×34 | [ ] | spectrum + speed table |
| 3–28 | Other signed kinds f32 + ×34 | [ ] | Seq/Res leaf-projected |
| 29–48 | MeshTween volume 2×2×2×1 all signed kinds | [ ] | = ModeTween volume wires |

## w2a Go↔Metal (StepTweenChain ModeStepTweenChain) — sign-off evidence

Run: `./scripts/macos_build_w2a.sh --steptweenchain-train`  
Wires: `STC1`/`STCO` (Line FamilyBP on pop @ full lr); 3D via ModeTweenChain volume wires `TCV1`/`TRC1`…. Exact_bit vs Go.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–28 | Line StepTweenChain leaf kinds ×34 | [ ] | FamilyTweenChain ≡ FamilyBP |
| 29–48 | MeshTweenChain volume 2×2×2×1 | [ ] | = ModeTweenChain volume wires |

## 1 — NormalBP

Chain-rule Backward + ApplyGradSGD. Start here. Dense first.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | `0.02-m2-dense-nbp` |
| RMSNorm | [ ] | [ ] | [ ] | `0.03-m2-rmsnorm-nbp` |
| LayerNorm | [ ] | [ ] | [ ] | `0.04-m2-layernorm-nbp` |
| SwiGLU | [ ] | [ ] | [ ] | `0.05-m2-swiglu-nbp` |
| Softmax | [ ] | [ ] | [ ] | `0.06-m2-softmax-nbp` |
| Embedding | [ ] | [ ] | [ ] | `0.07-m2-embedding-nbp` |
| MHA | [ ] | [ ] | [ ] | `0.08-m2-mha-nbp` |
| CNN1 | [ ] | [ ] | [ ] | `0.09-m2-cnn-nbp` |
| CNN2 | [ ] | [ ] | [ ] | `0.09-m2-cnn-nbp` |
| CNN3 | [ ] | [ ] | [ ] | `0.09-m2-cnn-nbp` |
| ConvT1 | [ ] | [ ] | [ ] | `0.10-m2-convt-nbp` |
| ConvT2 | [ ] | [ ] | [ ] | `0.10-m2-convt-nbp` |
| ConvT3 | [ ] | [ ] | [ ] | `0.10-m2-convt-nbp` |
| RNN | [ ] | [ ] | [ ] | `0.11-m2-rnn-nbp` |
| LSTM | [ ] | [ ] | [ ] | `0.12-m2-lstm-nbp` |
| Sequential | [ ] | [ ] | [ ] | `0.13-m2-sequential-nbp` |
| Residual | [ ] | [ ] | [ ] | `0.14-m2-residual-nbp` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | `0.15-m2-mamba-nbp` |
| Metacognition | [ ] | [ ] | [ ] | `0.16-m2-metacognition-nbp` |
| GDN | [ ] | [ ] | [ ] | `0.17-m2-gdn-nbp` |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | `0.18-m2-stack-nbp` |

**Checkpoint:** `0.18-m2-stack-nbp` — Dense…Stack NormalBP (GDN f32 ≥loose/≥industry; Stack NestedStack exact_bit STKB/STKV). StepBP/MeshBP signed at `0.19-m2-meshbp`; remaining NormalBP: KMeans / Parallel.

---

## 2 — StepBP

Needs Step `Line` pipe + FamilyBP (NormalBP credit) on pop.  
Ship: thin Stack host + `runtime/line` TickLine/TrainLineMSE; wire `STB1`/`STBO`; w2a `--stepbp-train`.  
**3D volumetric** for StepBP = MeshBP (`runtime/step` mesh-clock FamilyBP + ModeMeshBP volume Step). CNN2/3 and ConvT2/3 share CNN1/ConvT1 FamilyBP dispatch.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| RMSNorm | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| LayerNorm | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| SwiGLU | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| Softmax | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| Embedding | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| MHA | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| CNN1 | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| CNN2 | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| CNN3 | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| ConvT1 | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| ConvT2 | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| ConvT3 | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| RNN | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| LSTM | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| Sequential | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| Residual | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| Metacognition | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| GDN | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | `0.19-m2-meshbp` |

**Checkpoint:** `0.19-m2-meshbp` — StepBP Line (STB1) + MeshBP mesh-clock (MBP1) + ModeMeshBP volume 222 for NormalBP-signed kinds; next Tween / KMeans / Parallel.

---

## 3 — Tween

ModeTween: projected local credit @ `soft_lr = lr × 0.5` (Go `applyProjectedTween` — same gap on every leaf, no chain rule).  
Ship: `systems/tween` + `runtime/tween_train`; wires `TWB1`/`TWBO` + `TWV1`/`TWVO` + per-kind `TRV1`…`MBTV`; w2a `--tween-train`.  
CNN2/3 and ConvT2/3 share CNN1/ConvT1 dispatch. 3D = reverse-hop volume 2×2×2×1 for all NormalBP-signed kinds.

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

## 4 — TweenChain

ModeTweenChain: named Tween, credit = BP (Go `familyTweenChain` ≡ `familyBP` @ full lr).  
Ship: `systems/tween_chain` + `runtime/tween_chain_train`; wires `TCB1`/`TCBO` + `TCV1`/`TCVO` + per-kind `TRC1`…`MBTC`; w2a `--tweenchain-train`.  
CNN2/3 and ConvT2/3 share CNN1/ConvT1 dispatch. 3D = volume Step 2×2×2×1 for all NormalBP-signed kinds.

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

## 5 — StepTween

Needs Line + Tween.  
Ship: `runtime/line` Credit::FamilyTween on pop (`apply_projected` @ `lr×0.5`); wire `STW1`/`STWO`; w2a `--steptween-train`.  
**3D volumetric** for StepTween = MeshTween volume (= ModeTween reverse-hop projected local; existing `TWV1`/`TRV1`… wires). CNN2/3 and ConvT2/3 share CNN1/ConvT1 path.

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

## 6 — StepTweenChain

Needs Line + TweenChain.  
Ship: `runtime/line` Credit::FamilyBP on pop (same credit as StepBP / TweenChain); wire `STC1`/`STCO`; w2a `--steptweenchain-train`.  
**3D volumetric** = MeshTweenChain volume (= ModeTweenChain volume Step wires `TCV1`/`TRC1`…).

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

## 7 — MeshBP

Grid tick FamilyBP: `runtime/step` Forward×ticks + Backward BPTT + ApplyGradSGD (`MBP1`/`MBPO` for Dense mesh-clock).  
Go ModeMeshBP on grid also = volume Forward+Step (cameral_train) — signed 2×2×2×1 for all NormalBP-signed kinds via `--stepbp-train`.

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


## 8 — MeshTween

Needs Mesh + Tween.  
Ship: `runtime/step` `Credit::FamilyTween` — Forward×ticks → last-tick `apply_cell_projected` @ `lr×0.5` (`MTW1`/`MTWO`); w2a `--meshtween-train`.  
**3D volumetric** = ModeTween volume credit (`TWV1`/`TRV1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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

## 9 — MeshTweenChain

Needs Mesh + TweenChain.  
Ship: `runtime/step` `Credit::FamilyBP` under mesh clock (`MTC1`/`MTCO` ≡ MeshBP); w2a `--meshtweenchain-train`.  
**3D volumetric** = ModeTweenChain volume credit (`TCV1`/`TRC1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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

## 10 — TweenSplit

Ship: `systems::tween_split` + `runtime/tween_split_train` — collect trainable leaves, apply \(g_i=\frac1N P(g_y)\) @ full lr (`TSB1`/`TSP1`/`TRP1`…); w2a `--tweensplit-train`.  
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path. Softmax weightless (N may be 0).

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

## 11 — StepTweenSplit

Needs Line + TweenSplit.  
Ship: `runtime/line` Credit::FamilyTweenSplit on pop (`apply_split_child` 1/N @ full lr); wire `STS1`/`STSO`; w2a `--steptweensplit-train`.  
**3D volumetric** = MeshTweenSplit volume (= ModeTweenSplit / TweenSplit volume wires `TSP1`/`TRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path. Default AltTimes=1.

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

## 13 — StepTweenAlt

Needs Line + TweenAlt.  
Ship: `runtime/line` Credit::FamilyTweenAlt on pop (`apply_alt_child` Split→Tween frozen gap); wire `STA1`/`STAO`; w2a `--steptweenalt-train`.  
**3D volumetric** = MeshTweenAlt volume (= ModeTweenAlt / TweenAlt volume wires `TAV1`/`TRA1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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
CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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
**3D volumetric** = ModeTweenSplit / TweenSplit volume credit (`TSP1`/`TRP1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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
**3D volumetric** = ModeTweenAlt / TweenAlt volume credit (`TAV1`/`TRA1`…). CNN2/3 & ConvT2/3 share CNN1/ConvT1 path.

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

| Tag | Notes |
|---|---|
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

