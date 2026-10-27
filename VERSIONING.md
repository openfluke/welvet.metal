# welvet.metal versioning

**One scoreboard:** same ten categories / 100 points as Go Welvet.  
**Formula:** `version = 0.{round-half-up(earned)}` until **100 → 1.0.0**.

| Tag | Meaning |
|-----|---------|
| **0.00** | Bedrock — Metal host scaffold |
| **0.01** | Dense FormatNone×34 Metal forward exact_bit vs Go (`go_v_metal`) |
| **0.48** | 2026-10-26 board audit: 48/100, rounded to 48. All 29 numbered training modes and supported layer baselines; broader engine gates remain. |
| **0.68** | 2026-10-26: Freeze/Shadow/Adversarial/Memory, mixed Cameral branches, Stack overrides and GPU CamSync. Individual subgate audit replaces coarse 50% category accounting. |
| **0.69** | 2026-10-27: `stack::Kind::View` / `Flatten` (wire 24/25) — zero-weight reshape for Go `parallel.View` / `Flatten` so creature Mamba/CNN/MHA mids Mount. |
| **0.70** | 2026-10-27: Mesh* TrainMSE on sparse host grids — if step clock yields `empty mesh output`, fall back to `volume_dense` + Mesh credit family (creature 3×3×3 center cell). |
| **0.x** | `round-half-up(earned)` from the engine table in [README.md](README.md) |
| **1.0.0** | Engine board **100/100** — full Go parity on Apple GPU (FormatNone path + Model/IO + Systems + CamSync + peak fused, or explicit gate rewrite) |

| In (moves `0.x`) | Out |
|------------------|-----|
| Layers, FormatNone×34, quants, train modes, Grid/Line/Mesh, Tween*, systems, model/IO — **on Metal** | ANE until separately gated |
| Evidence via w2a `Go_v_Metal` | welvet.cpp / Go scores (not inherited) |
| Determinism + speed on named Apple GPU | Packaging polish alone |

**Sibling map:** welvet (Go ref) · welvet.cpp (CPU sibling) · **welvet.metal** (this) · welvet.qnn · w2a.

The 0.00/0.01 entries are historical milestone labels. The 0.48 audit applied
50% to whole partial categories. The 0.68 audit resolves individual subgates,
matching the granularity of welvet.cpp's current 0.67 accounting. It includes
both newly completed behavior and a correction to the earlier coarse score.
No CPU sibling credit or CPU implementation is inherited.

The ten earned values are **11 + 12 + 13 + 6 + 7 + 8 + 1 + 0 + 8 + 2 = 68**.
README names every subgate. Open budgets are foundation 4, Dense packed quant
3, broader Transformer variants 1, systems 4, model/IO 8 and peak residency/
fusion 12: **32 points**. Declared baseline contracts, including GDN Float32,
remain explicit; a category check does not claim arbitrary model variants.
Peak GPU kernels and rejected CPU fallbacks earn 2, rather than the former
blanket 7; residency is not deducted again from runtime scheduling.

This is an engine completion score, separate from the 60–100% numerical parity
target. CMake and runtime `welvet_version()` agree. Updating the working tree
does not publish a release or create a Git tag. Generic descriptor negotiation,
Go NAC integration and distribution remain separate unfinished work.
