#!/usr/bin/env bash
# iPhoneOS arm64 static build for welvet.metal (libwelvet_metal.a).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build-ios-arm64}"
DEPLOY="${IOS_DEPLOYMENT_TARGET:-15.0}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "welvet.metal build_ios.sh requires macOS + Xcode" >&2
  exit 1
fi

SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CXX="$(xcrun --sdk iphoneos --find clang++)"

cmake -S "$ROOT" -B "$BUILD" -G "Unix Makefiles" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOY" \
  -DCMAKE_OSX_SYSROOT="$SDK" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_OBJCXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="-isysroot $SDK -arch arm64 -miphoneos-version-min=$DEPLOY" \
  -DCMAKE_OBJCXX_FLAGS="-isysroot $SDK -arch arm64 -miphoneos-version-min=$DEPLOY -fobjc-arc" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWELVET_BUILD_SHARED=OFF \
  -DWELVET_BUILD_TOOLS=OFF \
  -DWELVET_BUILD_EXAMPLES=OFF \
  "$@"

cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "built: $BUILD/libwelvet_metal.a"
lipo -info "$BUILD/libwelvet_metal.a" 2>/dev/null || true
