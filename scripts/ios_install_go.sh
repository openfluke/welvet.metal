#!/usr/bin/env bash
# Build welvet.metal for iPhoneOS arm64 and install into Go Welvet native/metal/ios_arm64.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_GO_ROOT="${WELVET_GO_ROOT:-$ROOT/../welvet}"
if [[ ! -f "$WELVET_GO_ROOT/go.mod" ]]; then
  echo "Go Welvet checkout not found: $WELVET_GO_ROOT" >&2
  exit 1
fi

"$ROOT/scripts/build_ios.sh"

DEST="$WELVET_GO_ROOT/native/metal/ios_arm64"
INC="$WELVET_GO_ROOT/native/metal/include"
mkdir -p "$DEST" "$INC"
cp -f "$ROOT/build-ios-arm64/libwelvet_metal.a" "$DEST/libwelvet_metal.a"
cp -f "$ROOT/include/welvet_c.h" "$INC/welvet_metal.h"

VERSION="0.71"
if [[ -x "$ROOT/build/welvet_dense_fwd" ]]; then
  VERSION="$("$ROOT/build/welvet_dense_fwd" --version 2>/dev/null || echo 0.71)"
fi

{
  echo "engine=welvet.metal"
  echo "version=$VERSION"
  echo "artifact=libwelvet_metal.a"
  echo "platform=ios_arm64"
  echo "minimum_ios=${IOS_DEPLOYMENT_TARGET:-15.0}"
  echo "source_checkout=$ROOT"
  shasum -a 256 "$DEST/libwelvet_metal.a"
} > "$DEST/MANIFEST.txt"

# Keep top-level MANIFEST pointing at darwin (primary Go host); ios noted in ios_arm64/.
if [[ -f "$WELVET_GO_ROOT/native/metal/MANIFEST.txt" ]]; then
  if ! grep -q 'ios_arm64=present' "$WELVET_GO_ROOT/native/metal/MANIFEST.txt"; then
    echo "ios_arm64=present" >> "$WELVET_GO_ROOT/native/metal/MANIFEST.txt"
  fi
fi

cat > "$DEST/README.md" <<EOF
# ios_arm64

Shipped static \`libwelvet_metal.a\` for iPhoneOS arm64 (deployment ${IOS_DEPLOYMENT_TARGET:-15.0}+).
Link into the app binary with \`-force_load\` plus \`-framework Metal -framework Foundation\`;
no shared dylib on device.

\`\`\`bash
cd welvet.metal
./scripts/ios_install_go.sh
\`\`\`

Go \`nativemetal\` dlopen path is macOS-only today; this artifact is for Xcode / gomobile app link.
EOF

echo "Installed $DEST/libwelvet_metal.a"
