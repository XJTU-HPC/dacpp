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
# add_executable(FOuLadac FOuLa_sycl.cpp)
# set_target_properties(FOuLadac PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# target_link_libraries(FOuLadac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)

add_executable(blockMatMul3.sycl.cpp blockMatMul3.sycl.cpp)
set_target_properties(blockMatMul3.sycl.cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
target_link_libraries(blockMatMul3.sycl.cpp PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)

# add_executable(matMul matMul_sycl.cpp)
# add_executable(syclBlockMatMul0 blockMatMul0.sycl.cpp)
# add_executable(syclBlockMatMul1 blockMatMul1.sycl.cpp)
# add_executable(syclBlockMatMul2 blockMatMul2.sycl.cpp)
# add_executable(syclVectorMul0 vectorMul0.sycl.cpp)
# add_executable(outputSyclBlockMatrixMultiplication blockMatrixMultiplication_sycl.cpp)
# add_executable(liuliangdac liuliangdac_sycl.cpp)
# add_executable(stencil.dac stencil.dac_sycl.cpp)
# add_executable(stencilStandardSycl stencilStandardSycl.cpp)
# add_executable(waveEquation.dac waveEquation.dac_sycl.cpp)
# add_executable(waveEquationStandardSycl waveEquationStandardSycl.cpp)

# set_target_properties(matMul PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )





# set_target_properties(syclBlockMatMul0 PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# set_target_properties(syclBlockMatMul1 PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# set_target_properties(syclBlockMatMul2 PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# set_target_properties(syclVectorMul0 PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# set_target_properties(outputSyclBlockMatrixMultiplication PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )

# # set_target_properties(liuliangdac PROPERTIES
# #     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# # )
# set_target_properties(stencil.dac PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# set_target_properties(stencilStandardSycl PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# set_target_properties(waveEquation.dac PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )
# set_target_properties(waveEquationStandardSycl PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
# )

# 链接库到可执行文件
# target_link_libraries(matMul PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)





# target_link_libraries(syclBlockMatMul1 PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
# target_link_libraries(syclBlockMatMul0 PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
# target_link_libraries(syclBlockMatMul2 PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
# target_link_libraries(syclVectorMul0 PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
# target_link_libraries(outputSyclBlockMatrixMultiplication PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)

# # target_link_libraries(liuliangdac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
# target_link_libraries(stencil.dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
# target_link_libraries(stencilStandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
# target_link_libraries(waveEquation.dac PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
# target_link_libraries(waveEquationStandardSycl PRIVATE ${OpenCL_LIBRARY} sycl DacppLib RewriterLib)
