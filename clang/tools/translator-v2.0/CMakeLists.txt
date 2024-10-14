set(LLVM_LINK_COMPONENTS support)

add_clang_executable(translator-v2.0
  dacppLib/exception.cpp
  parser/Split.cpp
  parser/Param.cpp
  parser/ASTParse.cpp
  parser/DacppStructure.cpp
  test/test.cpp
  translator.cpp
  )
target_link_libraries(translator-v2.0
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )