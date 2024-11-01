# cmake 最低版本要求
cmake_minimum_required(VERSION 3.0)

# 项目信息
project(dac2sycl)

# 设置使用的编译器为Clang
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

# 添加源文件
add_executable(blockMatMulTest src/blockMatMulTest.cpp)
add_executable(blockMatMulTest2 src/blockMatMulTest2.cpp)
add_executable(matmulTest src/matmulTest.cpp)
add_executable(packingTest src/DataReconstructorTest.cpp)
add_executable(packingTest2 src/DataReconstructorTest2.cpp)
add_executable(packingTest3 src/DataReconstructorTest3.cpp)
add_executable(vecMulScalarTest src/vecMulScalarTest.cpp)

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
target_link_libraries(vecMulScalarTest DacppLib RewriterLib)