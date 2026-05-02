cmake_minimum_required(VERSION 3.20)
project(ane-lm VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS deployment target
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target")

# Use new Accelerate LAPACK API
add_compile_definitions(ACCELERATE_NEW_LAPACK)

# Fetch dependencies
include(FetchContent)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(json tokenizers_cpp)

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# Source files
set(VENDOR_SOURCES
    vendor/jinja/value.cpp
    vendor/jinja/string.cpp
    vendor/jinja/lexer.cpp
    vendor/jinja/parser.cpp
    vendor/jinja/runtime.cpp
    vendor/jinja/caps.cpp
)

set(CORE_SOURCES
    common.cpp
    core/safetensors.cpp
    core/model_loader.cpp
    core/tokenizer.cpp
    core/cpu_ops.cpp
    core/metal_ops.mm
    core/sampling.cpp
    core/ane_runtime.cpp
    utils.cpp
    generate.cpp
)

set(MODEL_SOURCES
    models/llm/qwen3_5.cpp
    models/llm/qwen3.cpp
    models/llm/gemma3.cpp
)

set(SERVER_SOURCES
    server/http_server.cpp
)

set(MAIN_SOURCE
    main.cpp
)

set(CLIB_SOURCE
    lib/ane_lm_c.cpp
)

# Common compile options
set(COMMON_LINK_LIBS
    nlohmann_json::nlohmann_json
    tokenizers_cpp
    "-framework Foundation"
    "-framework IOSurface"
    "-framework Accelerate"
    "-framework Metal"
    "-framework QuartzCore"
    "-lobjc"
)

set(COMMON_COMPILE_OPTS
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
)

# ============ Main CLI executable ============
add_executable(ane-lm
    ${VENDOR_SOURCES}
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${SERVER_SOURCES}
    ${MAIN_SOURCE}
)
target_link_libraries(ane-lm PRIVATE ${COMMON_LINK_LIBS})
target_compile_options(ane-lm PRIVATE ${COMMON_COMPILE_OPTS})

# ============ Shared library (C public API) ============
add_library(ane-lm-lib SHARED
    ${VENDOR_SOURCES}
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${CLIB_SOURCE}
)
target_link_libraries(ane-lm-lib PRIVATE ${COMMON_LINK_LIBS})
target_compile_options(ane-lm-lib PRIVATE ${COMMON_COMPILE_OPTS})
set_target_properties(ane-lm-lib PROPERTIES
    OUTPUT_NAME "ane_lm"
    PUBLIC_HEADER "include/ane_lm/ane_lm.h;include/ane_lm/common.h"
    MACOSX_RPATH ON
)

# Install targets
install(TARGETS ane-lm ane-lm-lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/ane_lm
)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
