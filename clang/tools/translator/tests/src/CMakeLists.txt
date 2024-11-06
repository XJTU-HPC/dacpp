# cmake 最低版本要求
cmake_minimum_required(VERSION 3.0)

# 项目信息
project(dac2sycl)

# 设置使用的编译器为Clang
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

# 添加源文件
add_executable(blockMatMulTest blockMatMulTest.cpp)
add_executable(blockMatMulTest2 blockMatMulTest2.cpp)
add_executable(matmulTest matmulTest.cpp)
add_executable(packingTest DataReconstructorTest.cpp)
add_executable(packingTest2 DataReconstructorTest2.cpp)
add_executable(packingTest3 DataReconstructorTest3.cpp)
add_executable(packingTest4 DataReconstructorTest4.cpp)
add_executable(vecMulScalarTest vecMulScalarTest.cpp)


set_target_properties(blockMatMulTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(blockMatMulTest2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(matmulTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(packingTest4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(vecMulScalarTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)

# 链接库到可执行文件
target_link_libraries(blockMatMulTest DacppLib RewriterLib)
target_link_libraries(blockMatMulTest2 DacppLib RewriterLib)
target_link_libraries(matmulTest DacppLib RewriterLib)
target_link_libraries(packingTest DacppLib RewriterLib)
target_link_libraries(packingTest2 DacppLib RewriterLib)
target_link_libraries(packingTest3 DacppLib RewriterLib)
target_link_libraries(packingTest4 DacppLib RewriterLib)
target_link_libraries(vecMulScalarTest DacppLib RewriterLib)

# # 设置使用的编译器为icpx
# set(CMAKE_C_COMPILER "/data/qinian/share/intel/oneapi/compiler/2024.0/bin/icpx")
# set(CMAKE_CXX_COMPILER "/data/qinian/share/intel/oneapi/compiler/2024.0/bin/icpx")

# add_executable(syclBlockMatMul1 sycl/blockMatMul1.sycl.cpp)
# # 添加编译选项
# target_compile_options(syclBlockMatMul1 PRIVATE
#     -fsycl
#     -fsycl-targets=nvptx64-nvidia-cuda
# )

# set_target_properties(syclBlockMatMul1 PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# target_link_libraries(syclBlockMatMul1 DacppLib RewriterLib)