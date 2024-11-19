# 设置最低版本要求
cmake_minimum_required(VERSION 3.14)

# 设置工程名称
project(Dac2SyclTest)

# 指定用于 C++ 代码的编译器
# set(CMAKE_C_COMPILER "/usr/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

add_subdirectory(src)

# 指定用于 DPC++ 代码的编译器
# set(CMAKE_C_COMPILER "/data/qinian/share/intel/oneapi/compiler/2024.0/bin/icpx")
# set(CMAKE_CXX_COMPILER "/data/qinian/share/intel/oneapi/compiler/2024.0/bin/icpx")

# 添加第二个子文件夹（DPC++ 代码）
add_subdirectory(sycl)
