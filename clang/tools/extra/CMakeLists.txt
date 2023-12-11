set(LLVM_LINK_COMPONENTS support)

add_clang_executable(vector-convert
  VectorConvert.cpp
  )
target_link_libraries(vector-convert
  clangTooling
  clangBasic
  clangASTMatchers
  )