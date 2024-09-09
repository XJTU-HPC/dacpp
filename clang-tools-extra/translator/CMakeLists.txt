set(LLVM_LINK_COMPONENTS support)

add_clang_executable(translator
  dacppLib/exception.cpp  
  parser/rewriteLib.cpp
  parser/ASTLib.cpp
  rewriter/sub_template.cpp
  rewriter/dacInfo.cpp
  syclLib/DataReconstructor.cpp
  test/test.cpp
  translator.cpp
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