cmake_minimum_required(VERSION 3.22)
project("engineercompanion")

# llama.cpp sources (pulled via CMake FetchContent or prebuilt)
# Prebuilt .so files go to src/main/jniLibs/arm64-v8a/
# This CMake builds the JNI wrapper that links against them.

add_library(engineer_companion_jni SHARED native-lib.cpp)

# Find system log lib
find_library(log-lib log)

target_link_libraries(engineer_companion_jni
    ${log-lib}
    # libllama.so and libggml.so are prebuilt and placed in jniLibs
    # They are linked automatically by Android build system
)

target_include_directories(engineer_companion_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a/include
)
