# welvet.metal gap census

**Authoritative checklist:** [README.md](README.md) — full Go Welvet inventory as Metal rebuild gates (3D volumetric engine). Tick there.

**Version `0.00`** — engine board **0 / 100**. This file is the machine-oriented dump (`welvet_dense_fwd --gaps`).

```bash
./scripts/build_macos.sh
./build/example_metal_smoke
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
```

## Done (M0 host — not engine-board credit)

| ID | Detail |
|---|---|
| Metal device | `MTLDevice` probe, list, runtime identity |
| Shader library | Runtime compile `shaders/welvet_dense.metal` |
| Shared buffers | Alloc / upload / download |
| Pipeline smoke | `welvet_smoke_saxpy` PSO |
| C ABI | version / init / gaps counters |
| macOS build | `scripts/build_macos.sh` |

## Open

Everything else in [README.md](README.md): foundation, Dense, 23 layers, 34 dtypes, 20 quants, 29 train modes, Grid/Line/Mesh, systems, model/IO, lucy, CamSync, peak fused, w2a `Go_v_Metal`, iOS, ANE.
