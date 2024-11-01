set(LLVM_LINK_COMPONENTS support)

add_clang_executable(translator
  dacppLib/exception.cpp
  parser/Split.cpp
  parser/Param.cpp
  parser/ASTParse.cpp
  parser/DacppStructure.cpp
  rewriter/Rewriter.cpp
  test/test.cpp
  translator.cpp
  dac2sycl/dacppLib/lib/dacInfo.cpp
  dac2sycl/dacppLib/lib/sub_template.cpp
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