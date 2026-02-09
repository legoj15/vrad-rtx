cmake_minimum_required(VERSION 3.15)
project(vrad_simd_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vrad_simd_bench vrad_simd_bench.cpp)

if(MSVC)
    # /O2 = full optimization, /fp:fast = fast floating point (matches VRAD build)
    target_compile_options(vrad_simd_bench PRIVATE /O2 /fp:fast /EHsc /W3)
    # AVX2 as the max arch flag — individual functions use target attributes for finer control
    # We set /arch:AVX2 as the baseline compile flag since we runtime-detect anyway
    target_compile_options(vrad_simd_bench PRIVATE /arch:AVX2)
else()
    # GCC/Clang
    target_compile_options(vrad_simd_bench PRIVATE -O2 -mavx2 -mfma -Wall)
endif()
