cmake_minimum_required(VERSION 3.20)
project(welvet_metal VERSION 0.01 LANGUAGES CXX OBJCXX)

# Apple-only Welvet engine: C++ host + Metal compute shaders.
# macOS Apple silicon is the primary host; iOS device is a later artifact.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WELVET_BUILD_SHARED "Build shared library (C ABI / NAC)" ON)
option(WELVET_BUILD_TOOLS "Build CLI tools for w2a go_v_metal" ON)
option(WELVET_BUILD_EXAMPLES "Build standalone Metal smoke examples" ON)
option(WELVET_ENABLE_ANE "gap? Apple Neural Engine / Core ML path" OFF)

if(NOT APPLE)
  message(FATAL_ERROR "welvet.metal requires an Apple host toolchain (macOS/iOS).")
endif()

if(IOS)
  set(WELVET_PLATFORM_IOS ON)
else()
  set(WELVET_PLATFORM_MACOS ON)
endif()

include(GNUInstallDirs)

set(WELVET_CORE_SOURCES
  src/core/status.cpp
  src/core/dtype.cpp
  src/core/tensor.cpp
  src/core/float_codecs.cpp
  src/metal/device.mm
  src/metal/buffer.mm
  src/metal/pipeline.mm
  src/metal/library.mm
  src/metal/matvec.mm
  src/metal/training.mm
  src/metal/ops.mm
  src/weights/store.cpp
  src/weights/native_codec.cpp
  src/layers/dense/dense.cpp
  src/layers/dense/session.cpp
  src/layers/mha/mha.cpp
  src/layers/swiglu/swiglu.cpp
  src/layers/rmsnorm/rmsnorm.cpp
  src/layers/layernorm/layernorm.cpp
  src/layers/softmax/softmax.cpp
  src/layers/embedding/embedding.cpp
  src/layers/sequential/sequential.cpp
  src/layers/residual/residual.cpp
  src/layers/cnn1/cnn1.cpp
  src/layers/cnn2/cnn2.cpp
  src/layers/cnn3/cnn3.cpp
  src/layers/convt1/convt1.cpp
  src/layers/convt2/convt2.cpp
  src/layers/convt3/convt3.cpp
  src/layers/rnn/rnn.cpp
  src/layers/lstm/lstm.cpp
  src/layers/gdn/gdn.cpp
  src/layers/stack/stack.cpp
  src/layers/mamba/mamba.cpp
  src/layers/kmeans/kmeans.cpp
  src/layers/parallel/parallel.cpp
  src/layers/metacognition/metacognition.cpp
  src/architecture/grid.cpp
  src/runtime/forward.cpp
  src/runtime/backward.cpp
  src/runtime/training.cpp
  src/runtime/line.cpp
  src/runtime/step.cpp
  src/runtime/tween_chain_train.cpp
  src/systems/tween_chain.cpp
  src/runtime/tween_train.cpp
  src/systems/tween.cpp
  src/gaps/registry.cpp
  src/interop/c_abi.cpp
)

add_library(welvet_metal_obj OBJECT ${WELVET_CORE_SOURCES})
set_target_properties(welvet_metal_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  # .mm sources need ARC for Metal/Foundation object lifetimes.
  OBJCXX_STANDARD 20
)
target_include_directories(welvet_metal_obj
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(welvet_metal_obj
  PRIVATE
    WELVET_METAL=1
    $<$<BOOL:${WELVET_PLATFORM_MACOS}>:WELVET_OS_MACOS=1>
    $<$<BOOL:${WELVET_PLATFORM_IOS}>:WELVET_OS_IOS=1>
    $<$<BOOL:${WELVET_ENABLE_ANE}>:WELVET_ENABLE_ANE=1>
)
target_compile_options(welvet_metal_obj PRIVATE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)
target_link_libraries(welvet_metal_obj
  PUBLIC
    "-framework Metal"
    "-framework Foundation"
)

# Embed source so static/shared Metal artifacts work without the checkout.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/welvet_dense.metal")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/shaders/welvet_dense.metal" WELVET_METAL_SOURCE)
file(GLOB WELVET_EXTRA_SHADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ops_*.metal")
foreach(SHADER IN LISTS WELVET_EXTRA_SHADERS)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SHADER}")
  file(READ "${SHADER}" SHADER_SOURCE)
  string(APPEND WELVET_METAL_SOURCE "\n${SHADER_SOURCE}")
endforeach()
configure_file(cmake/embedded_shaders.hpp.in generated/embedded_shaders.hpp @ONLY)
target_include_directories(welvet_metal_obj PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

if(WELVET_BUILD_SHARED)
  add_library(welvet SHARED $<TARGET_OBJECTS:welvet_metal_obj>)
else()
  add_library(welvet STATIC $<TARGET_OBJECTS:welvet_metal_obj>)
endif()
add_library(welvet::welvet ALIAS welvet)
target_include_directories(welvet
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(welvet
  PUBLIC
    "-framework Metal"
    "-framework Foundation"
)
set_target_properties(welvet PROPERTIES
  OUTPUT_NAME welvet_metal
  POSITION_INDEPENDENT_CODE ON
)

if(WELVET_BUILD_TOOLS)
  add_executable(welvet_dense_fwd tools/dense_fwd/main.cpp tools/dense_fwd/volume_fwd.cpp
    tools/dense_fwd/stepbp_wires.cpp)
  target_link_libraries(welvet_dense_fwd PRIVATE welvet)
endif()

if(WELVET_BUILD_EXAMPLES)
  add_executable(example_metal_smoke examples/metal_smoke/main.cpp)
  target_link_libraries(example_metal_smoke PRIVATE welvet)
endif()

install(TARGETS welvet
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY shaders/ DESTINATION ${CMAKE_INSTALL_DATADIR}/welvet/shaders)
if(WELVET_BUILD_TOOLS)
  install(TARGETS welvet_dense_fwd RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

message(STATUS "welvet.metal platform: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "ANE stub=${WELVET_ENABLE_ANE}; Metal shaders embedded")
