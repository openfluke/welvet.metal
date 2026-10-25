# welvet.metal gap census

**Authoritative checklist:** [README.md](README.md) — full Go Welvet inventory as Metal rebuild gates (3D volumetric engine). Tick there.

**Version `0.01`** — Dense FormatNone×34 Metal forward exact_bit vs Go. This file is the machine-oriented dump (`welvet_dense_fwd --gaps`).

```bash
./scripts/build_macos.sh
./scripts/macos_build_w2a.sh          # go_v_metal Dense forward
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
```

## Done

| ID | Detail |
|---|---|
| M0 host | Device, library, buffers, smoke PSO, C ABI |
| FormatNone×34 | encode/decode (Go staging) |
| Dense Metal F | soft-double GEMV + bias/act; **no host MatVec**; `fastMathEnabled=NO` |
| w2a `go_v_metal` | exact_bit×34 vs Go; 3-way speed Go/C++/Metal |

**Note:** Apple Metal has **no hardware float64**. Soft-double (float2) matches Go’s f64 MatVec when fast-math is off. Tiny shapes are dispatch-bound (~400µs) — C++ wins those; larger shapes are next for Metal speed.

## GPU-only correction (2026-10-25)

CPU-backed layer entries are now unsupported. `--gaps` reports them as pending,
even when their old host implementations passed Go parity. Dense Float32 SGD,
MSE/MSEGrad, and mesh gradient accumulation now execute Metal kernels. Dense
forward/backward retain FormatNone staging for all 34 dtypes; other update dtypes
are unsupported. See [METAL_ONLY.md](docs/METAL_ONLY.md).

## Open

NormalBP, remaining layers, quants, volumes, train modes, systems, model/IO, … — see [README.md](README.md).
