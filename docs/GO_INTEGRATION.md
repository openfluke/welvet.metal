# Go integration: mounted models and mixed backend routing

The native artifact is `libwelvet_metal.dylib`. Go Welvet's `nativemetal`
package loads it through a versioned C ABI. No Go pointers are retained by C++.
The backend owns copies of model parameters, configuration and its training
state until `Close`. It never calls welvet.cpp or silently trains on the CPU.
The working-tree engine version remains 0.68; packaging is outside the engine
completion score. The model-session ABI starts at version 1.

## Build and install

From the Metal checkout, with the Go sibling at `../welvet`:

```sh
./scripts/macos_install_go.sh
```

This builds the standalone wire executable and shared library, targets macOS
13, and installs the dylib, C header and SHA-256 manifest into Go's
`native/metal/`. Override `WELVET_GO_ROOT` for another Go checkout. Building the
backend requires CMake and Apple's Clang/SDK/frameworks. Go applications using
the prebuilt dylib need Go/cgo and a compatible Apple toolchain; they do not
rebuild the C++ engine or require welvet.cpp. A cgo-disabled adapter build is
currently blocked by the existing Go WebGPU dependency; it is not a supported
build of this integration.

The verified hardware is Apple M5. A macOS 13 deployment target records the
library's minimum OS target; it is not evidence of testing on every older
macOS release or Apple GPU.

## Explicit weight ownership

In a Go application:

```go
import "github.com/openfluke/welvet/nativemetal"

session, err := nativemetal.Mount(model) // layer, Stack, Parallel or Grid
if err != nil { return err }
defer session.Close()

loss, err := session.TrainMSE(input, target, parallel.ModeNormalBP, 0.001)
if err != nil { return err }
_ = loss

output, err := session.Forward(input)
if err != nil { return err }
_ = output

// Go stores still contain their original weights until this explicit call.
if err := session.SyncToGo(); err != nil { return err }
return entity.WriteCameralFile("trained.entity", model, parallel.ModeNormalBP)
```

The save example assumes `model` is a `*parallel.Stack` supported by Go's
Cameral ENTITY writer. Other model shapes use their existing Go save APIs.
`SyncToGo` restores native bytes, scale, Float32 payloads and Float64 biases.
Mamba's ALog/DSkip and GDN's projection/parameter arrays are included. Derived
Go weight caches are replaced along with the stores. Export preparation and
metadata validation finish before Go weights are committed.

`Weights()` returns an owned snapshot without changing Go. `SyncCams(when)`
applies CamSync inside the native session and is separate from `SyncToGo()`.
`Close()` is idempotent and discards any unpulled updates; it never saves or
implicitly changes Go weights. Mount reads the current Go Store values; pull any pending updates from another
backend into Go first. The native session continues from its current
weights after a pull. Mounting the same Go model again creates a new independent
session from the Go values at that point.

Do not replace the source model's topology/stores, or mutate/read/save its
weights concurrently with `SyncToGo`. Calls on one native Session serialize;
separate sessions own independent parameter copies. A failure during a native
training operation is reported; training is not a transactional rollback API.

## Build and deployment selection

Build a Metal-only Go application with:

```sh
go build -tags welvet_metal ./cmd/your-app
```

The tag disables the independent `nativecpp` cgo binding while retaining Go's
explicit CPU implementations. Without it, the existing Go CPU-native binding
retains its original build behavior and may link the CPU sibling. Enable `nativemetal.EnableRouting()` to connect `Exec.Backend` and the existing
JSON configuration to persistent Metal mounts. Explicit `Mount` sessions remain
available independently. Automatic hardware choice/performance planning remains
separate from explicit JSON placement.

Library resolution order is an explicit `Options{LibraryPath: ...}`, then
`WELVET_METAL_LIBRARY`, then `libwelvet_metal.dylib` beside the executable, then
the installed artifact in the Go checkout. For distribution, place the dylib
beside the executable or supply an explicit path/environment variable. A
compiled source-checkout path is only a development fallback, not a deployment
strategy. Loading an incompatible library/platform returns an error.

A complete runnable example is in Go's `cmd/welvet-metal-example`:

```sh
cd ../welvet
go run -tags welvet_metal ./cmd/welvet-metal-example -save trained.entity
```

## Supported mounted contracts

- All 23 baseline layer families and their FormatNone parameter dtypes, including
  nested Stack/Parallel, seven combines and Cameral controls. GDN is Float32-only.
  The public tensor bridge is Float32; native parameter bytes remain authoritative.
- All numbered modes plus Inherit/Freeze/Shadow/Adversarial/Memory on Stack and
  Parallel. Stack Step modes preserve Line flights between calls. Proxy/cache
  state remains native across repeated calls.
- Grids preserve coordinates, disabled ingress cells and remote links. Shared
  volume BP/Tween/Split/Alt/proxy rules and all seven mesh-clock credits are
  available. Pass `ticks` to `TrainMSE` for the mesh unroll; each call starts a
  fresh mesh clock while retaining model weights/proxy state. Stack-only Step
  modes on Grid return Unsupported.
- Frozen/teacher/memory behavior and CamSync retain the contracts in
  [CAMERAL.md](CAMERAL.md), including explicit cross-store exceptions.

Unsupported configurations fail explicitly. Rotation/DNA regularization/Dream replay,
packed quant stores, tied/shared
mutable stores, heterogeneous Sequential/Residual children (use Stack),
nonuniform Dense activations inside those two legacy containers, AltTimes > 1,
live KV caches and advanced MHA policies beyond this descriptor, Softmax mask/grid/hierarchy/Gumbel
configuration, custom/live Metacognition rules, and live GDN recurrent state
have separate mount gates. GDN must be reset and use default epsilon and
FormatNone projections. Native session state is not a portable checkpoint:
`SyncToGo` returns parameters, not Line flights, optimizer/proxy state, KV caches
or rule counters. Close/remount does not preserve those transient states.

Descriptor v1 permits up to 4096 nodes/cells, depth 64, a 256 MiB mount message,
and the existing per-payload limits. Full persistent GPU graph/tape residency,
large-model packaging/performance and automatic hardware selection remain open.

## ABI ownership and wire layout

The C header exposes `welvet_model_abi_version`, mount, forward, train, parameter
export, CamSync, release and blob-free. Opaque model handles belong to the
native library. Returned blobs must be freed with `welvet_metal_blob_free` from
that library. Release/operations on the same handle must not race in a raw C
caller; the Go wrapper enforces this with a mutex. C++ exceptions become status
codes; `nativemetal.NativeError` preserves native codes.

All descriptor scalars are little endian. `WMN1` contains root kind (Stack=0,
Parallel=1, Grid=2), count, four grid dimensions, recursive child descriptors,
root/per-child Cameral controls and full native Store snapshots; Grid appends
six metadata integers per cell (enabled, remote flag, remote coordinates).
The child grammar is shared with the existing native wire tools rather than a
second numerical implementation. Input tensors are rank, dimensions and raw
Float32 values; output tensors add their element count before values.

`NWS1` exports stores in the recursive Go `dna.CollectStores` order: dtype,
rows, cols, scale, counted Float32 payload, counted native bytes and counted
Float64 bias. Additional Mamba/GDN records contain kind and packed parameters.
All input buffers are copied during calls; shared Apple memory does not create
an alias to a Go slice. Native full-model saving/loading remains separate from
Go ENTITY saving after explicit weight return.

## Verification

In w2a, the integration matrix checks 144 mounted training groups and 161 mesh
groups: all 23 families, supported native dtypes, three placements, all 34 mode
enum choices on Stack/Parallel, seven mixed combines and all seven mesh credits.
Lifecycle tests exercise ENTITY save/reload, Go cache invalidation after pull,
closed handles, invalid models/inputs, concurrent callers and CamSync boundaries.
The Metal-only tag and a linked-library audit verify CPU sibling independence.
Existing GPU-only guards also exercise malformed public ABI descriptors.

Numerical comparisons use the documented 60–100% spectrum. Native mount/export
byte preservation, delayed Go updates, metadata and saved Float32 example
replay retain exact checks. Run all of it from the Metal checkout:

```sh
./scripts/macos_build_w2a.sh --go-integration
```

## Mixed Go / C++ / Metal execution

```go
router, err := nativemetal.EnableRouting() // optionally Options{LibraryPath: ...}
if err != nil { return err }
defer router.Close()

cfg, err := nativecpp.ParseExecConfigJSON(jsonBytes)
if err != nil { return err }
if err = placement.Stack(model, cfg, "main", "0,0,0,0"); err != nil { return err }
// Existing public execution/training APIs now dispatch the assigned operations.
loss, err := parallel.TrainStackMSE(model, input, target, parallel.ModeNormalBP, 0.001)
```

`placement` is `github.com/openfluke/welvet/runtime/placement`. For a Stack,
JSON `layers` keys are zero-based immediate child indices. A nested Stack can
be assigned as a segment. For example, default `go_cpu`, child indices 4–8
assigned `native_cpu`, and 9–12 assigned `metal` execute in that order under
Go's graph scheduler. Explicit child assignments survive parent propagation.
For a Grid, `placement.Grid(grid, cfg, "main")` uses sector keys `"z,y,x,0"`
and layer keys `"L"`. Existing remote links control activation sources;
backend assignments control execution. Routing does not create connections.
The existing Go backward remote-link convention is preserved: gradients enter
local tape inputs and are not newly accumulated onto remote targets.

Each assigned Metal operation uses a persistent native session. Forward pre/post,
incoming output gradients, input gradients, weight gradients, SGD, Dense proxy
primitives, and homogeneous Stack/Parallel credit/MSE training cross the C ABI.
Float32 activation/gradient tensors and the existing mounted-model/dtype limits
apply; unsupported models, missing libraries or missing segment symbols fail
explicitly. Selecting Metal without an enabled router fails rather than running
Go CPU. Routing statistics expose completed native calls.

Go orchestrates mixed graph scheduling, losses and cross-backend tensor transfer.
The assigned Metal forward/backward/SGD/proxy kernels run on Apple GPU. This is
not a fully resident heterogeneous graph: activations and gradients cross host
buffers. Whole Stack/Parallel MSE delegation retains native Line/proxy state;
mixed-mode scheduling follows the existing Go implementation.

Routed native updates automatically replace the corresponding Go parameters,
so subsequent Go execution, backend changes and Go ENTITY saving see current
values. Pending C++ parameter updates are pulled before a routed Metal mount.
Changes made directly in Go are detected and remount the affected operation;
remounting resets that operation's native transient state. Backend changes
transfer parameters; live Line/proxy/KV/rule state is not migrated between
different engines. Do not mutate, replace
or concurrently execute/save the same Go graph while a routed call uses it.
Close the router to release every cached mount. Explicit `Mount` sessions still
require explicit `SyncToGo`; their ownership contract is unchanged.

Use the **normal build** for simultaneous C++ and Metal. The `welvet_metal` tag
intentionally disables C++ and is appropriate for Go/Metal-only programs.
Linux/Windows ordinary Go builds do not load Apple libraries. `EnableRouting`
is opt-in and requires Apple silicon macOS/cgo.

A runnable 13-layer example, with optional JSON configuration and Go ENTITY save:

```sh
cd ../welvet
go run ./cmd/welvet-metal-routing-example -save /tmp/welvet-mixed.entity
```

Full verification including the existing session/mesh/ownership tests:

```sh
cd /Users/openfluke/gs/welvet.metal
./scripts/macos_build_w2a.sh --go-routing
```

The mixed suite requires an available C++ artifact and does not accept silently
skipping that backend. It covers a 13-layer Go→C++→Metal chain, grid coordinates
and remote links, backward/SGD for all 23 layer families over supported native
dtypes, persistent mode delegation, mixed local training modes, pinned overrides,
C++ pending weight handoff, missing-library/closed-router errors and explicit
session ownership. All tests reside in w2a.

Sign-off on Apple M5: the full `--go-routing` command passed in 109.612 seconds;
the expanded routing checks (including nested Sequential/Residual overrides and
direct credit training) passed under the Go race detector in 7.527 seconds.
Existing Go core/nativecpp/layer/runtime regressions passed. The standalone mixed
example trained and wrote a Go ENTITY file. A repository-wide `go build ./...`
is independently blocked by the pre-existing mosstts dumper importing an
external `go-sentencepiece/internal/model` package; routing packages and the
mixed example build successfully.
