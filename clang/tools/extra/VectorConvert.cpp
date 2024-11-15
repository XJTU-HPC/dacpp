#include <sstream>
#include <string>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/ASTContext.h"  
#include "clang/Basic/Diagnostic.h"  
#include "clang/Basic/FileManager.h"  
#include "clang/Basic/SourceManager.h" 
#include "clang/Lex/Preprocessor.h"  
#include "clang/Parse/ParseAST.h"  
#include "clang/Rewrite/Core/Rewriter.h"  
#include "iostream"  
#include "vector"  

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory ToolingSampleCategory("Tooling Sample");

//通过实现 RecursiveASTVisitor，我们可以指定遍历并重写哪些 AST 节点
class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {//定义一个MyASTVisitor类，继承RecursiveASTVisitor类。遍历AST
public:
  MyASTVisitor(Rewriter &R) : TheRewriter(R) {}//构造函数，接受一个Rewriter对象的引用，并使用它初始化TheRewriter成员变量。Rewriter对象是Clang用来实时代码改写的

  bool VisitTypedefNameDecl(TypedefNameDecl *TD) {
      //当遍历到一个TypedefNameDecl（类型别名声明）节点时返回一个布尔值，如果返回true，则表示遍历应该继续；如果返回false，则表示遍历应该停止。
    if (TD->getUnderlyingType()->isVectorType()) {  
      // 将std::vector类型转换为SYCL的vector类型  
      QualType VecType = Context->getSYCLVectorType();
      //getSYCLVectorType没找到，或需要自己写 
      TypeSourceInfo *TInfo = Context->getTrivialTypeSourceInfo(VecType, SourceLocation());  
        //getTrivialTypeSourceInfo方法用来获取关于vector类型的信息。
      Replacement Repl = TheRewriter.ReplaceType(TD->getTypeSpecStartLoc(), TInfo->getTypeLoc()); 
        //调用TheRewriter的ReplaceType方法，将原来的类型替换为SYCL的向量类型。TD->getTypeSpecStartLoc()返回类型别名声明的起始位置，TInfo->getTypeLoc()返回新类型的源代码位置信息。
      TheRewriter.ReplaceText(TD->getSourceRange(), Repl);  
       //调用TheRewriter的ReplaceText方法，用新的类型替换原来的类型别名声明。TD->getSourceRange()返回类型别名声明的源代码范围，Repl是上一步中生成的类型替换。
    }  
    return true;  
  }  

private:
  Rewriter &TheRewriter;
  ASTContext *Context;  
};

//实现ASTConsumer接口，用于读取生成的AST
class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(Rewriter &R) : Visitor(R) {}
  //MyASTConsumer类继承自ASTConsumer，重写HandleTopLevelDecl方法，这个方法在Clang解析源代码并构建AST时被调用，每次解析一个顶级对象（例如一个函数或一个全局声明）时，都会调用这个方法。

  bool HandleTopLevelDecl(DeclGroupRef DR) override {
    //DeclGroupRef是一个顶级对象的集合，遍历整个集合并对每个对象调用TraverseDecl方法，以遍历和访问这个对象的子节点。
    for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
      // 使用我们的AST访问器遍历声明
      Visitor.TraverseDecl(*b);
      (*b)->dump();
       // 调用dump方法来打印这个对象的源代码表示
    }
    return true;
  }

private:
  MyASTVisitor Visitor;
};

// 对于提供给工具的每个源文件，都会创建一个新的FrontendAction
class MyFrontendAction : public ASTFrontendAction {//FrontendAction类主要负责处理源代码文件，并生成AST。
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {//打印当前处理的源文件的名称，并将重写的内容输出到标准输出。这个方法使用TheRewriter对象来获取源文件的名称和重写的内容。
    SourceManager &SM = TheRewriter.getSourceMgr();
    llvm::errs() << "** EndSourceFileAction for: "
                 << SM.getFileEntryForID(SM.getMainFileID())->getName() << "\n";

    //送回重写的缓冲区
    TheRewriter.getEditBuffer(SM.getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {//为源文件创建一个新的ASTConsumer,打印了正在为哪个文件创建ASTConsumer，并设置了TheRewriter的SourceManager。创建并返回一个新的MyASTConsumer对象，这个对象使用TheRewriter进行初始化。
    llvm::errs() << "** Creating AST consumer for: " << file << "\n";
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<MyASTConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  llvm::Expected<CommonOptionsParser> op=CommonOptionsParser::create(argc, argv, ToolingSampleCategory);
  //使用CommonOptionsParser::create方法创建一个CommonOptionsParser实例,用于解析命令行参数,这个实例被存储在op中，这是一个llvm::Expected类型的对象，这是LLVM用来处理可能出错的返回类型的通用方式。
  ClangTool Tool(op.get().getCompilations(), op.get().getSourcePathList());
	//创建一个ClangTool实例。ClangTool是一个用于处理C++源代码的工具，它接受编译命令和源文件路径列表作为输入。
  //ClangTool::run 接受一个FrontendActionFactory，然后用于创建实现 FrontendAction 接口的新对象。这里我们使用辅助类 newFrontendActionFactory 创建一个默认factory，该factory将每次返回一个新的MyFrontendAction对象。
  return Tool.run(newFrontendActionFactory<>().get());//最后，主函数返回tool运行的结果打印出来，这可能是成功或失败的代码。
}