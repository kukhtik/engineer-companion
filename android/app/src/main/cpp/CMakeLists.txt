cmake_minimum_required(VERSION 3.22)
project("engineercompanion" CXX)

include(FetchContent)

# ── llama.cpp via FetchContent ────────────────────────────────────────────────
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
# CPU-only; no Vulkan this phase
set(GGML_VULKAN          OFF CACHE BOOL "" FORCE)
set(GGML_METAL           OFF CACHE BOOL "" FORCE)
set(GGML_CUDA            OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        b4600
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(llama)

# ── JNI wrapper ───────────────────────────────────────────────────────────────
add_library(engineer_companion_jni SHARED native-lib.cpp)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(engineer_companion_jni
    llama
    ggml
    ${log-lib}
    ${android-lib}
)

target_include_directories(engineer_companion_jni PRIVATE
    ${llama_SOURCE_DIR}/include
    ${llama_SOURCE_DIR}/ggml/include
)

target_compile_options(engineer_companion_jni PRIVATE -std=c++17)
