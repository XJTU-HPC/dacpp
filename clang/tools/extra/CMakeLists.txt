set(LLVM_LINK_COMPONENTS support)
include("/data/powerzhang/llvm/clang/cmake/modules/AddClang.cmake")
add_clang_executable(loop-convert
  LoopConvert.cpp
  )
target_link_libraries(loop-convert
  PRIVATE
  clangTooling
  clangBasic
  clangASTMatchers
  )
