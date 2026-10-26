#!/usr/bin/env bash
# Build and install this independent backend into a local Go Welvet checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_GO_ROOT="${WELVET_GO_ROOT:-$ROOT/../welvet}"
if [[ ! -f "$WELVET_GO_ROOT/go.mod" ]]; then
  echo "Go Welvet checkout not found: $WELVET_GO_ROOT" >&2
  exit 1
fi
"$ROOT/scripts/build_macos.sh" -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0
DEST="$WELVET_GO_ROOT/native/metal/darwin_arm64"
mkdir -p "$DEST" "$WELVET_GO_ROOT/native/metal/include"
cp "$ROOT/build/libwelvet_metal.dylib" "$DEST/libwelvet_metal.dylib"
cp "$ROOT/include/welvet_c.h" "$WELVET_GO_ROOT/native/metal/include/welvet_metal.h"
VERSION="$("$ROOT/build/welvet_dense_fwd" --version)"
{
  echo "engine=welvet.metal"
  echo "version=$VERSION"
  echo "model_abi=1"
  echo "platform=darwin_arm64"
  echo "minimum_macos=13.0"
  echo "source_checkout=$ROOT"
  shasum -a 256 "$DEST/libwelvet_metal.dylib"
} > "$WELVET_GO_ROOT/native/metal/MANIFEST.txt"
echo "Installed $DEST/libwelvet_metal.dylib"
echo "Go build: go build -tags welvet_metal ./..."
