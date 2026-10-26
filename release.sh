#!/usr/bin/env bash
# Build, validate, package and publish this independent Apple Metal backend.
# Default: commit local source changes, push the current branch + version tag,
# then create a GitHub Release with macOS arm64 shared/static binaries.
# The script never force-pushes or moves an existing tag.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DRY_RUN=0
PLAN=0
SKIP_BUILD=0
SKIP_TESTS=0
TEST_GO=0
DRAFT=0
AUTO_COMMIT=1
BUILD_ROOT="$ROOT/build-release"
DIST="$ROOT/dist"
MIN_MACOS="13.0"

usage() {
  cat <<'HELP'
Usage: ./release.sh [options]
  --dry-run       Build, test and package; do not commit, tag, push or publish
  --plan          Show version and release actions only; no builds or mutations
  --skip-build    Reuse build-release/shared and build-release/static
  --skip-tests    Skip native GPU smoke checks (recorded in release notes)
  --test-go       Also run sibling w2a mounted-model/mesh/save-reload checks
  --no-commit     Require a clean source tree instead of committing local changes
  --draft         Create a draft GitHub Release after pushing branch/tag
  -h, --help      Show this help

Default: release v<current CMake/README/runtime version> to the origin GitHub repo.
Requires macOS Apple silicon, CMake, Apple Clang/SDK, Python 3 and git.
Publishing additionally requires an authenticated GitHub CLI (gh).
Generated builds and dist/ are ignored. No sibling repository is modified.
HELP
}
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --plan) PLAN=1; DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --test-go) TEST_GO=1 ;;
    --no-commit) AUTO_COMMIT=0 ;;
    --draft) DRAFT=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
fail() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Required command missing: $1"; }
need git
need python3
VERSION="$(python3 - "$ROOT" <<'PY'
from pathlib import Path
import re,sys
r=Path(sys.argv[1])
cm=re.search(r'project\(welvet_metal\s+VERSION\s+([0-9.]+)',(r/'CMakeLists.txt').read_text())
rm=re.search(r'\*\*Version\*\*\s*\|\s*\*\*`([0-9.]+)`',(r/'README.md').read_text())
abi=re.search(r'welvet_version\(void\)\s*\{.*?return\s+"([0-9.]+)"',(r/'src/interop/c_abi.cpp').read_text(),re.S)
if not all((cm,rm,abi)) or len({m[1] for m in (cm,rm,abi)})!=1:
    raise SystemExit('CMake, README and runtime source versions must agree')
print(cm[1])
PY
)"
SCORE="$(python3 - "$ROOT/README.md" <<'PY_SCORE'
from pathlib import Path
import re,sys
m=re.search(r'\*\*Engine board\*\*\s*\|\s*\*\*(\d+(?:\.\d+)?)\s*/\s*100',Path(sys.argv[1]).read_text())
if not m: raise SystemExit('README engine board not found')
print(m[1])
PY_SCORE
)"
TAG="v$VERSION"
BRANCH="$(git symbolic-ref --quiet --short HEAD)" || fail "Check out a branch before releasing"
REPO="$(git remote get-url origin | python3 -c '
import re,sys
url=sys.stdin.read().strip()
m=re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+/[^/]+?)(?:\.git)?/?",url)
if not m: raise SystemExit("origin must point to a GitHub repository")
print(m[1])
')"
[[ "$REPO" == */welvet.metal ]] || fail "origin must name a welvet.metal repository"
if [[ "$DRY_RUN" -eq 0 ]]; then
  need gh
  gh auth status --hostname github.com >/dev/null 2>&1 || fail "Run gh auth login first"
  if [[ "$AUTO_COMMIT" -eq 0 && -n "$(git status --porcelain)" ]]; then
    fail "Source tree is dirty; commit changes or omit --no-commit"
  fi
  if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
    [[ "$(git rev-parse "$TAG^{commit}")" == "$(git rev-parse HEAD)" ]] || fail "$TAG already names another commit; tags are never moved"
    [[ -z "$(git status --porcelain)" ]] || fail "$TAG already exists; do not change the tagged source"
  fi
fi
printf 'welvet.metal %s\nGitHub: %s\nBranch: %s\nAsset: dist/welvet-metal-%s-macos-arm64.tar.gz\n' "$TAG" "$REPO" "$BRANCH" "$TAG"
if [[ "$PLAN" -eq 1 ]]; then
  echo "Plan: build shared + static -> relocate/install -> GPU forward/backward/SGD checks -> package + SHA256"
  [[ "$TEST_GO" -eq 0 ]] || echo "Plan: test the staged dylib through sibling Go/w2a"
  echo "Publish: commit local source (unless --no-commit), push current branch and immutable tag, gh release create"
  echo "Plan only: no files, git state or remote state changed"
  exit 0
fi
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || fail "Build/package requires Apple silicon macOS"
for command in cmake xcrun clang++ tar shasum otool; do need "$command"; done
JOBS="$(sysctl -n hw.ncpu)"
mkdir -p "$BUILD_ROOT" "$DIST"
STAGE="$(mktemp -d "$DIST/.stage.XXXXXX")"
# Clean up only the temporary directory created by this invocation.
cleanup() { python3 - "$STAGE" <<'PY'
from pathlib import Path
import shutil,sys
p=Path(sys.argv[1])
if p.name.startswith('.stage.'):
    shutil.rmtree(p,ignore_errors=True)
PY
}
trap cleanup EXIT
NAME="welvet-metal-$TAG-macos-arm64"
PACKAGE="$STAGE/$NAME"
mkdir -p "$PACKAGE"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  for kind in shared static; do
    shared=ON
    tools=ON
    [[ "$kind" == shared ]] || { shared=OFF; tools=OFF; }
    cmake -S "$ROOT" -B "$BUILD_ROOT/$kind" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN_MACOS" \
      -DCMAKE_INSTALL_NAME_DIR=@rpath '-DCMAKE_INSTALL_RPATH=@loader_path/../lib' \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF -DWELVET_ENABLE_ANE=OFF \
      -DWELVET_BUILD_SHARED="$shared" -DWELVET_BUILD_TOOLS="$tools" \
      -DWELVET_BUILD_EXAMPLES=OFF
    cmake --build "$BUILD_ROOT/$kind" --parallel "$JOBS"
  done
fi
for artifact in shared/libwelvet_metal.dylib shared/welvet_dense_fwd static/libwelvet_metal.a; do
  [[ -f "$BUILD_ROOT/$artifact" ]] || fail "Missing $BUILD_ROOT/$artifact; run without --skip-build"
done
cmake --install "$BUILD_ROOT/shared" --prefix "$PACKAGE"
cmake --install "$BUILD_ROOT/static" --prefix "$PACKAGE"
[[ "$("$PACKAGE/bin/welvet_dense_fwd" --version)" == "$VERSION" ]] || fail "Packaged CLI version mismatch"
# A published executable must not link this checkout's absolute build paths.
if otool -L "$PACKAGE/bin/welvet_dense_fwd" | python3 -c '
import sys
paths=[line.strip().split(" (")[0] for line in sys.stdin.read().splitlines()[1:]]
sys.exit(0 if any("libwelvet_metal" in p and p!="@rpath/libwelvet_metal.dylib" for p in paths) else 1)
'; then
  fail "Packaged tool is not relocatable"
fi

NATIVE_CHECK="skipped (--skip-tests)"
if [[ "$SKIP_TESTS" -eq 0 ]]; then
  python3 "$ROOT/scripts/release_smoke.py" "$PACKAGE/lib/libwelvet_metal.dylib" "$VERSION"
  NATIVE_CHECK="passed: packaged C ABI + Metal identity + forward/backward/SGD"
fi
GO_CHECK="not requested"
if [[ "$TEST_GO" -eq 1 ]]; then
  need go
  W2A="${WELVET_W2A_ROOT:-$ROOT/../w2a}"
  [[ -f "$W2A/go.mod" ]] || fail "Sibling w2a missing; set WELVET_W2A_ROOT or omit --test-go"
  (
    cd "$W2A"
    WELVET_METAL_LIBRARY="$PACKAGE/lib/libwelvet_metal.dylib" \
      go test -tags welvet_metal ./tests/go_v_metal -count=1 -timeout 25m \
      -run '^(TestGoNativeMetalIntegration|TestGoNativeMetalMesh|TestGoNativeMetalSaveReload)$'
  )
  GO_CHECK="passed: Go mounted-model/native-dtype/mesh/ENTITY roundtrips"
fi
# The user invokes this publishing branch; --dry-run never mutates git.
if [[ "$DRY_RUN" -eq 0 && -n "$(git status --porcelain)" ]]; then
  [[ "$AUTO_COMMIT" -eq 1 ]] || fail "Dirty source tree"
  git add -A
  git commit -m "Release welvet.metal $TAG"
fi
COMMIT="$(git rev-parse HEAD)"
DIRTY=false
[[ -z "$(git status --porcelain)" ]] || DIRTY=true
cp "$ROOT/README.md" "$ROOT/VERSIONING.md" "$ROOT/LICENSE" "$PACKAGE/"
mkdir -p "$PACKAGE/docs"
cp "$ROOT/docs/GO_INTEGRATION.md" "$ROOT/docs/METAL_ONLY.md" "$PACKAGE/docs/"
python3 - "$PACKAGE" "$VERSION" "$COMMIT" "$DIRTY" "$MIN_MACOS" "$NATIVE_CHECK" "$GO_CHECK" <<'PY'
import json,subprocess,sys
from pathlib import Path
p,v,commit,dirty,minimum,native,go=sys.argv[1:]
manifest=dict(engine='welvet.metal',version=v,platform='macos-arm64',minimum_macos=minimum,
              model_abi=1,segment_extension=1,source_commit=commit,source_dirty_at_build=dirty=='true',
              native_validation=native,go_validation=go,
              sdk_version=subprocess.check_output(['xcrun','--show-sdk-version'],text=True).strip(),
              compiler=subprocess.check_output(['clang++','--version'],text=True).splitlines()[0])
Path(p,'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
ARCHIVE="$DIST/$NAME.tar.gz"
COPYFILE_DISABLE=1 tar -czf "$ARCHIVE" -C "$STAGE" "$NAME"
(cd "$DIST" && shasum -a 256 "$NAME.tar.gz" > "$NAME.sha256")
NOTES="$DIST/release-$TAG.md"
cat > "$NOTES" <<NOTES
# welvet.metal $TAG

Independent Apple Metal backend for Welvet. Engine board: $SCORE/100;
see the tagged README and VERSIONING.md for the current subgate accounting.

- Supported numerical forward, backward and training run on Apple GPU through custom Metal kernels.
- 23 baseline layer families, 29 numbered training modes, 34 native parameter encodings where supported (GDN Float32).
- Persistent Go mount/forward/training/weight-return API; explicit JSON and mixed Go/C++/Metal routing in the Go binding.
- Float32 activation interface; best-effort numerical compatibility targets the industry60–exact100 bands.
- Separate from welvet.cpp; this package has no CPU-sibling runtime dependency.

## macOS arm64 artifact

Deployment target: macOS $MIN_MACOS. Includes shared and static libraries, C/C++ headers,
embedded shader sources, the standalone wire CLI, license, usage docs and a provenance manifest.
A deployment target does not imply validation on every older OS or Apple GPU.
Standalone native model/checkpoint IO, automatic hardware planning and ANE remain separate gates.

Validation: $NATIVE_CHECK. Go/w2a: $GO_CHECK.
Source commit: $COMMIT (dirty working-tree package: $DIRTY).

Verify the archive with: shasum -a 256 -c $NAME.sha256
NOTES
printf '\nPrepared:\n  %s\n  %s\n  %s\n' "$ARCHIVE" "$DIST/$NAME.sha256" "$NOTES"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete: no commit, tag, push, authentication check or GitHub release performed"
  exit 0
fi
# Push explicit refs. Existing tags can only be reused when they already name HEAD.
if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
  [[ "$(git rev-parse "$TAG^{commit}")" == "$COMMIT" ]] || fail "$TAG is not the release commit"
else
  git tag -a "$TAG" -m "welvet.metal $TAG"
fi
git push origin "HEAD:refs/heads/$BRANCH"
git push origin "refs/tags/$TAG"
release_args=("$TAG" "$ARCHIVE" "$DIST/$NAME.sha256" --repo "$REPO" --verify-tag
  --title "welvet.metal $TAG" --notes-file "$NOTES")
[[ "$DRAFT" -eq 0 ]] || release_args+=(--draft)
gh release create "${release_args[@]}"
echo "Release: https://github.com/$REPO/releases/tag/$TAG"
