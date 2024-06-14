#include <string>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "rewriteLib.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;

dacpp::Source2Source s2s;

class DacHandler : public MatchFinder::MatchCallback {
private:
  Rewriter &Rewrite;
  
public:
  DacHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    s2s.setRewriter(&Rewrite);
    if (const BinaryOperator* dacExpr = Result.Nodes.getNodeAs<clang::BinaryOperator>("dac_expr")) {
      // 获得 Shell 信息
      s2s.setShells(dacExpr);

      // 获得 calc 信息
      s2s.setCalcs(dacExpr);
    }
    else if (const FunctionDecl* mainFunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("main")) {
      s2s.setMainFunc(mainFunc);
    }
  }

};

// ASTConsumer 是一个抽象接口，由需要遍历 AST 的用户来实现
// 源码文件解析得到的 Clang AST 会通过 ASTConsumer 回传给外部工具
// 可以重载两个函数：
// HandleTopLevelDecl()：解析顶级声明时被调用
// HandleTranslationUnit()：在整个文件都解析完后调用
// ASTConsumer 是一个用来在AST上写一些通用动作的接口
class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(Rewriter &R) : HandleForDac(R) {
    // 可以通过 addMatcher 添加用户构造的匹配器到 MatchFinder中
    Matcher.addMatcher(binaryOperator(hasOperatorName("<->")).bind("dac_expr"), &HandleForDac);
    Matcher.addMatcher(functionDecl(hasName("main")).bind("main"), &HandleForDac);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    Matcher.matchAST(Context);
  }

private:
  DacHandler HandleForDac;
  MatchFinder Matcher;
};

// ASTFrontendAction 是为使用基于 AST consumer 的前端动作的抽象基类
// FrontendAction 是允许用户特定动作作为编译的一部分执行的接口
// 需要实现 CreateASTConsumer 方法，该方法对于每个翻译单元返回一个 ASTConsumer
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    s2s.rewriteDac();
    s2s.rewriteMain();
    // this will output to screen as what you got.
    // rewriter_.getEditBuffer(rewriter_.getSourceMgr().getMainFileID())
    //     .write(llvm::outs());
    
    // 生成sycl文件
    std::error_code error_code;
    llvm::raw_fd_ostream outFile("/data/zjx/dacpp/clang-tools-extra/translator/output.cpp", error_code, llvm::sys::fs::F_None);
    // this will write the result to outFile
    rewriter_.getEditBuffer(rewriter_.getSourceMgr().getMainFileID()).write(outFile);
    outFile.close();
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    rewriter_.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<MyASTConsumer>(rewriter_);
  }

private:
  Rewriter rewriter_;
};

// 命令行选项的帮助信息
static llvm::cl::OptionCategory translator("translator options");

int main(int argc, const char **argv) {
  // 所有命令行 Clang 工具通用的选项解析器
  CommonOptionsParser op(argc, argv, translator);
  
  // 运行一个前端动作的工具
  ClangTool tool(op.getCompilations(), op.getSourcePathList());

  // 在命令行指定的所有文件上运行一个动作
  return tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}