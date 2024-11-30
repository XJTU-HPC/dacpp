# cmake 最低版本要求
cmake_minimum_required(VERSION 3.0)

# 项目信息
project(dac2sycl)


add_executable(BufferTemplateTest BufferTemplateTest.cpp)
add_executable(UsmTemplateTest UsmTemplateTest.cpp)

set_target_properties(UsmTemplateTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)
set_target_properties(BufferTemplateTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator/tests
)

target_link_libraries(UsmTemplateTest DacppLib RewriterLib)
target_link_libraries(BufferTemplateTest DacppLib RewriterLib)
