# Cameral training and store synchronization

The native APIs are in `welvet/runtime/cameral_train.hpp` and
`welvet/cam_sync.hpp`. They use this backend's Metal kernels and native stores;
they do not load or call welvet.cpp. `Controls` lives on Parallel, Stack and
nested children. The current interface uses Float32 tensors with authoritative
FormatNone parameter storage across 34 dtypes; GDN remains Float32-only.

## Training behavior

- Freeze skips optimizer updates while preserving the input gradient. Shadow
  also freezes credit and can supply the teacher prediction in a mixed Parallel.
- The first Shadow branch supplies consistency credit to trainable peers:
  `2 * coefficient * (student - teacher) / element_count`, added on the GPU to
  their ordinary output gradient. A zero coefficient selects Go's default of 1.
- Adversarial branches negate their learning rate. Memory branches update when
  the last loss meets the surprise threshold; a nonpositive threshold opens the
  gate. Root Parallel training records its loss before evaluating this gate.
- Per-branch rates multiply the outer rate; zero skips that branch. Inherit
  resolves to the parent mode. All 34 enum values are covered, including the
  resolve-only Inherit entry, Freeze/Shadow and BP-gated actors.
- Stack child overrides and nested Parallel controls follow Go's credit walker.
  Frozen children still propagate input gradients to an upstream Dense stem.
  Mixed BP branches return their input gradient using the updated weights,
  matching Go's update order.
- Mixed Filter credit follows Go's approximate gate update (`mean(gy) * 0.1`),
  with the parent Tween exception. The remaining combine semantics retain
  their existing GPU implementations.

Two reference details matter: a standalone Adversarial/Memory mode outside the
Parallel actor context resolves to ordinary BP; ascent and surprise gating are
branch behavior. Metacognition's forward rules may reset the observed store
even when its optimizer credit is frozen. Freeze does not suppress forward
state transitions. Tests distinguish these from optimizer changes.

`train_mse` and `train_stack_mse` reproduce Go's public Parallel/Stack training
entry points. Stack Step modes retain Line state between calls. The volume
entry point composes the actual Go per-cell backward, SGD and Parallel/Stack
credit contract for BP-gated modifiers, using the volume loss for Memory.
It covers placed cells and multi-cell volumes; it does not establish arbitrary
mixed Cameral actor behavior for every mesh-clock credit. Existing ordinary
mesh-clock modes retain their separate training evidence.

## CamSync and BlendStores

GPU kernels compute store means, blend weights/biases and fully re-encode the
declared native dtype. Store replacement recomputes native scale/minimum; it
does not reuse an SGD scale. Primed Dense and native wire caches are invalidated
before the next forward. w2a also tests the corresponding Go `SetFromF32` cache
invalidation fix in the source reference. The Go nested Stack walker also uses
its recursive mixed-mode predicate, so inherited Stack modes do not bypass
deeper explicit branch controls. The added native-dtype inheritance fixtures
exercise this correction in both backends.

`blend_stores` snapshots all participants before writing. Every participant
contributes to the mean. A participant alpha at or below zero prevents its
write; positive alpha is clamped to one. At configuration level, a nonpositive
default alpha means Go's default of one. Matching nonempty biases blend too.
Mismatching bias counts leave biases alone.

Parallel synchronization supports default/all groups, explicit and overlapping
groups, per-branch alpha and unequal store counts. Invalid group indices are
ignored and a mismatching matrix shape skips that store slot. Freeze/Shadow
branches are excluded from implicit groups. Stack synchronization supports
explicit cross-layer/branch/store endpoint pairs and nested endpoints. Explicit
cross links can write a frozen endpoint, as in Go; group exclusion does not
override an explicitly requested cross link. Bad endpoints return an error.

The scheduling API exposes Manual, AfterSample, AfterStep and AfterPulse.
Automatic sample/step hooks follow Go, including Stack warmup boundaries.
Pulse is an explicit `maybe_sync(..., AfterPulse)` call. Root Parallel Split/Alt
returns before NoteLoss/automatic synchronization, following Go's early-return
behavior. Full persistent GPU graph residency remains open: host scheduling,
buffer staging and copies surround the GPU arithmetic.

Softmax and GDN have no blendable Go `weights.Store`. Mamba synchronization
visits In/Out projections; its ALog/DSkip fields are outside this Go store
contract. Recursive Stack/Parallel collection includes nested gates and all
other declared trainable store families. A no-store case is an intentional
no-op, rather than evidence of blending an unsupported representation.

## Verification

`TestGoVMetalCameralTrain` has 465 groups; `TestGoVMetalCamSync` has 109. They
sweep 34 native dtypes where supported, GDN Float32, all 23 layer families,
seven combines, nested overrides, per-branch rates, singleton sequence shapes,
111/211 placements and 222 Float32 volumes. Each fixture rebuilds its original
state and requires identical Metal loss, weights and output on replay. Store
replacement/cache tests and independent scalar assertions check ascent,
Memory sleep/wake, teacher/student credit, frozen input gradients, alpha-zero
native byte preservation and bias/cache coherence.

These new suites use standard combined absolute/relative tolerances near zero:
`max(|Go - Metal| / (1 + max(|Go|, |Metal|)))`. At most `1e-5` is strict (80),
at most `1e-3` is industry (60), and bit identity is exact (100). Loss uses the
same industry floor. Nonfinite results fail. Shapes and applicable frozen
store checks are exact. Reports include actual bit-match rates and raw maximum
absolute differences; the band score is not the percentage of correct elements.
Existing training suites retain their own established gates.

From the Metal checkout, rebuild and run the complete new evidence:

```sh
./scripts/macos_build_w2a.sh --cameral-all
```

`--training-all` includes these suites and all numbered training regressions.
The bridge is `CMB1`/`CMBO`; validation sources and engine guards live in w2a.
