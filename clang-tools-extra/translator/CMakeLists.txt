set(LLVM_LINK_COMPONENTS support)

add_clang_executable(translator
  translator.cpp
  rewriteLib.cpp
  dacppLib/exception.cpp
  )
target_link_libraries(translator
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )