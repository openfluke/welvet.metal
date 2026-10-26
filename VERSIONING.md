# welvet.metal versioning

**One scoreboard:** same ten categories / 100 points as Go Welvet.  
**Formula:** `version = 0.{round-half-up(earned)}` until **100 → 1.0.0**.

| Tag | Meaning |
|-----|---------|
| **0.00** | Bedrock — Metal host scaffold |
| **0.01** | Dense FormatNone×34 Metal forward exact_bit vs Go (`go_v_metal`) |
| **0.48** | 2026-10-26 board audit: 48/100, rounded to 48. All 29 numbered training modes and supported layer baselines; broader engine gates remain. |
| **0.x** | `round-half-up(earned)` from the engine table in [README.md](README.md) |
| **1.0.0** | Engine board **100/100** — full Go parity on Apple GPU (FormatNone path + Model/IO + Systems + CamSync + peak fused, or explicit gate rewrite) |

| In (moves `0.x`) | Out |
|------------------|-----|
| Layers, FormatNone×34, quants, train modes, Grid/Line/Mesh, Tween*, systems, model/IO — **on Metal** | ANE until separately gated |
| Evidence via w2a `Go_v_Metal` | welvet.cpp / Go scores (not inherited) |
| Determinism + speed on named Apple GPU | Packaging polish alone |

**Sibling map:** welvet (Go ref) · welvet.cpp (CPU sibling) · **welvet.metal** (this) · welvet.qnn · w2a.

The 0.00/0.01 entries are historical milestone labels. The 0.48 audit applies
the documented 50% rule to partial categories and the five one-point systems
subgates individually. See the README for all ten scores. This is an engine
completion score, separate from the 60–100% numerical parity target. The source
version, CMake project version and runtime `welvet_version()` agree; no Git
release tag is created by updating these files.
