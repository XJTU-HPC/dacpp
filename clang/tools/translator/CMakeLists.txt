set(LLVM_LINK_COMPONENTS support)

add_clang_executable(translator
  dacppLib/exception.cpp
  parser/Split.cpp
  parser/Param.cpp
  parser/ASTParse.cpp
  parser/DacppStructure.cpp
  rewriter/Rewriter.cpp
  rewriter/dacInfo.cpp
  rewriter/sub_template.cpp
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