# iPhoneOS arm64 toolchain for welvet.metal (static libwelvet_metal.a).
# Prefer scripts/build_ios.sh which sets the same flags explicitly.
set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "" FORCE)

execute_process(
  COMMAND xcrun --sdk iphoneos --show-sdk-path
  OUTPUT_VARIABLE _IOS_SDK
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_OSX_SYSROOT "${_IOS_SDK}" CACHE PATH "" FORCE)

execute_process(
  COMMAND xcrun --sdk iphoneos --find clang++
  OUTPUT_VARIABLE _IOS_CXX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_CXX_COMPILER "${_IOS_CXX}" CACHE FILEPATH "" FORCE)
set(CMAKE_OBJCXX_COMPILER "${_IOS_CXX}" CACHE FILEPATH "" FORCE)

set(CMAKE_CXX_FLAGS_INIT "-arch arm64 -miphoneos-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
set(CMAKE_OBJCXX_FLAGS_INIT "-arch arm64 -miphoneos-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET} -fobjc-arc")

# Device static artifact — no shared / tools / examples.
set(WELVET_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(WELVET_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(WELVET_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
