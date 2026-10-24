# welvet.metal versioning

**One scoreboard:** same ten categories / 100 points as Go Welvet.  
**Formula:** `version = 0.{round(earned)}` until **100 → 1.0.0**.

| Tag | Meaning |
|-----|---------|
| **0.00** | Bedrock — Metal host scaffold; **0 / 100** on the engine board |
| **0.x** | `round(earned)` from the engine table in [README.md](README.md) |
| **1.0.0** | Engine board **100/100** — full Go parity on Apple GPU (FormatNone path + Model/IO + Systems + CamSync + peak fused, or explicit gate rewrite) |

| In (moves `0.x`) | Out |
|------------------|-----|
| Layers, FormatNone×34, quants, train modes, Grid/Line/Mesh, Tween*, systems, model/IO — **on Metal** | ANE until separately gated |
| Evidence via w2a `Go_v_Metal` | welvet.cpp / Go scores (not inherited) |
| Determinism + speed on named Apple GPU | Packaging polish alone |

**Sibling map:** welvet (Go ref) · welvet.cpp (CPU sibling) · **welvet.metal** (this) · welvet.qnn · w2a.
