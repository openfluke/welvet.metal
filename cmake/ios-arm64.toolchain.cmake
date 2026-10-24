# Placeholder iOS device toolchain — artifact packaging is gap? (see GAPS.md).
set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")
# Configure with: cmake -S . -B build-ios -DCMAKE_TOOLCHAIN_FILE=cmake/ios-arm64.toolchain.cmake
