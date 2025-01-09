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
# add_executable(matMul_dac matMul.dac_sycl.cpp)
add_executable(FOuLa_sycl FOuLa_sycl.cpp)
# add_executable(matMul_StandardSycl matMulStandardSycl.cpp)
add_executable(waveEquation_dac waveEquation.dac_sycl.cpp)
add_executable(waveEquation_StandardSycl waveEquationStandardSycl.cpp)
add_executable(stencil_dac stencil.dac_sycl.cpp)
add_executable(stencil_StandardSycl stencilStandardSycl.cpp)
add_executable(jacobi_dac jacobi.dac_sycl.cpp)
add_executable(jacobi_StandardSycl jacobiStandardSycl.cpp)
add_executable(liuliang_dac liuliangdac_sycl.cpp)
add_executable(liuliang_StandardSycl liuliangStandardSycl.cpp)
add_executable(decay_chain_dac_sycl liuliangStandardSycl.cpp)
add_executable(decay_chain_StandardSycl liuliangStandardSycl.cpp)


set_target_properties(FOuLa_sycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
# set_target_properties(matMul_dac PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
# )
# set_target_properties(matMul_StandardSycl PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
# )
set_target_properties(waveEquation_dac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
set_target_properties(waveEquation_StandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
set_target_properties(stencil_dac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
set_target_properties(stencil_StandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
set_target_properties(jacobi_dac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
set_target_properties(jacobi_StandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
set_target_properties(liuliang_dac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
set_target_properties(liuliang_StandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
set_target_properties(decay_chain_dac_sycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)
set_target_properties(decay_chain_StandardSycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests/examples
)

# # 链接库到可执行文件
target_link_libraries(FOuLa_sycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
# target_link_libraries(matMul_dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)

# target_link_libraries(matMul_StandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(waveEquation_dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(waveEquation_StandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(stencil_dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(stencil_StandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(jacobi_dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(jacobi_StandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(liuliang_dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(liuliang_StandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(decay_chain_dac_sycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
target_link_libraries(decay_chain_StandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)