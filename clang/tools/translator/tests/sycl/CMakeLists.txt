# cmake 最低版本要求
cmake_minimum_required(VERSION 3.0)

# 项目信息
project(SyclTest)

set(CMAKE_CXX_STANDARD 17)

# 设置使用的编译器为 icpx
set(CMAKE_C_COMPILER "/data/qinian/share/intel/oneapi2025/compiler/2025.0/bin/icpx")
set(CMAKE_CXX_COMPILER "/data/qinian/share/intel/oneapi2025/compiler/2025.0/bin/icpx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64")

# 添加 SYCL 头文件路径
set(SYCL_PATH "/data/qinian/share/intel/oneapi2025/compiler/latest/include")
set(OpenCL_LIBRARY "/data/qinian/share/intel/oneapi2025/compiler/latest/lib/libOpenCL.so")

include_directories(${SYCL_PATH})

# 添加源文件

add_executable(FOuLadac FOuLa_sycl.cpp)
add_executable(stencildac stencildac_sycl.cpp)
add_executable(stencilStandardSycl stencilStandardSycl.cpp)
add_executable(waveEquation.dac waveEquation.dac_sycl.cpp)
add_executable(waveEquationStandardSycl waveEquationStandardSycl.cpp)
add_executable(jacobi.dac jacobi.dac_sycl.cpp)
add_executable(jacobiStandardSycl jacobiStanardSycl.cpp)
add_executable(liuliangdac liuliangdac_sycl.cpp)
add_executable(liuliangStandardSycl liuliangStandardSycl.cpp)
add_executable(decay_chain_dac decay_chain_dac_sycl.cpp)
add_executable(decay_chain_StandardSycl decay_chain_StandardSycl.cpp)
add_executable(matMuldac matMuldac_sycl.cpp)
add_executable(matMulStandardSycl matMulStandardSycl.cpp)




add_executable(syclBlockMatMul0 blockMatMul0.sycl.cpp)
add_executable(syclBlockMatMul1 blockMatMul1.sycl.cpp)
add_executable(syclBlockMatMul2 blockMatMul2.sycl.cpp)
add_executable(syclBlockMatMul3 blockMatMul3.sycl.cpp)
add_executable(syclVectorMul0 vectorMul0.sycl.cpp)
add_executable(outputSyclBlockMatrixMultiplication blockMatrixMultiplication_sycl.cpp)

set_target_properties(FOuLadac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(stencildac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(stencilStandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(waveEquation.dac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(waveEquationStandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)

set_target_properties(jacobi.dac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(jacobiStandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(liuliangdac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(liuliangStandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(decay_chain_dac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(decay_chain_StandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(matMuldac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(matMulStandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)


set_target_properties(syclBlockMatMul0 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(syclBlockMatMul1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(syclBlockMatMul2 PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(syclBlockMatMul3 PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(syclVectorMul0 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(outputSyclBlockMatrixMultiplication PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)

target_link_libraries(FOuLadac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(stencildac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(stencilStandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(waveEquation.dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(waveEquationStandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(jacobi.dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(jacobiStandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(liuliangdac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(liuliangStandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(decay_chain_dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(decay_chain_StandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(matMuldac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(matMulStandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)


target_link_libraries(syclBlockMatMul1 PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(syclBlockMatMul0 PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(syclBlockMatMul2 PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(syclBlockMatMul3 PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(syclVectorMul0 PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(outputSyclBlockMatrixMultiplication PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
