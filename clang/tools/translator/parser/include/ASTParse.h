#ifndef ASTPARSE_H
#define ASTPARSE_H

#include <vector>
#include <queue>

#include "clang/AST/AST.h"

#include "Param.h"

using namespace clang;

namespace dacppTranslator {

// 从结点中找到特定结点，深度优先搜索
template <typename NodeType>
NodeType* getNode(Stmt* curStmt) {
    if(!curStmt) return nullptr;
    for(Stmt::child_iterator it = curStmt->child_begin(); it != curStmt->child_end(); it++) {
        if(isa<NodeType>(*it)) return dyn_cast<NodeType>(*it);
        if(getNode<NodeType>(*it)) {
            return getNode<NodeType>(*it);
        }
    }
    return nullptr;
}

// 从结点中找到特定结点，广度优先搜索
template <typename NodeType>
NodeType* getNodeBFS(Stmt* curStmt) {
    std::queue<Stmt*> q;
    q.push(curStmt);
    while(!q.empty()) {
        Stmt* cur = q.front();
        q.pop();
        for(Stmt::child_iterator it = cur->child_begin(); it != cur->child_end(); it++) {
            if(isa<NodeType>(*it)) {
                return dyn_cast<NodeType>(*it);
            }
            q.push(*it);
        }
    }
    return nullptr;
}



void getSplitExpr(Expr* curExpr, std::string& name, std::vector<Expr*>& splits);

// 将 clang 节点对应的代码转换成 std::string
std::string stmt2String(Stmt *stmt);

// 判断数据的输入输出属性
bool inputOrOutput(std::string dataType);

}

//===----------------------------------------------------------------------===//
// StmtPrinter Visitor
//===----------------------------------------------------------------------===//

namespace declvisitor {
/// A simple visitor class that helps create declaration visitors.
template<template <typename> class Ptr, typename ImplClass, typename RetTy=void>
class Base {
public:
#define PTR(CLASS) typename Ptr<CLASS>::type
#define DISPATCH(NAME, CLASS) \
  return static_cast<ImplClass*>(this)->Visit##NAME(static_cast<PTR(CLASS)>(D))

  RetTy Visit(PTR(Decl) D) {
    switch (D->getKind()) {
#define DECL(DERIVED, BASE) \
      case Decl::DERIVED: DISPATCH(DERIVED##Decl, DERIVED##Decl);
#define ABSTRACT_DECL(DECL)
#include "clang/AST/DeclNodes.inc"
    }
    llvm_unreachable("Decl that isn't part of DeclNodes.inc!");
  }

  // If the implementation chooses not to implement a certain visit
  // method, fall back to the parent.
#define DECL(DERIVED, BASE) \
  RetTy Visit##DERIVED##Decl(PTR(DERIVED##Decl) D) { DISPATCH(BASE, BASE); }
#include "clang/AST/DeclNodes.inc"

  RetTy VisitDecl(PTR(Decl) D) { return RetTy(); }

#undef PTR
#undef DISPATCH
};

} // namespace declvisitor

/// A simple visitor class that helps create declaration visitors.
///
/// This class does not preserve constness of Decl pointers (see also
/// ConstDeclVisitor).
template <typename ImplClass, typename RetTy = void>
class DeclVisitor
    : public declvisitor::Base<std::add_pointer, ImplClass, RetTy> {};

/// A simple visitor class that helps create declaration visitors.
///
/// This class preserves constness of Decl pointers (see also DeclVisitor).
template <typename ImplClass, typename RetTy = void>
class ConstDeclVisitor
    : public declvisitor::Base<llvm::make_const_ptr, ImplClass, RetTy> {};

  class DeclPrinter : public DeclVisitor<DeclPrinter>, public StmtVisitor<DeclPrinter> {
    PrintingPolicy Policy;
    const ASTContext &Context;
    unsigned Indentation;
    bool PrintInstantiation;

    PrinterHelper* Helper;
    std::string NL;

    raw_ostream& Indent() { return Indent(Indentation); }
    raw_ostream& Indent(unsigned Indentation);
    void ProcessDeclGroup(SmallVectorImpl<Decl*>& Decls);

    void Print(AccessSpecifier AS);
    void PrintConstructorInitializers(CXXConstructorDecl *CDecl,
                                      std::string &Proto);

    /// Print an Objective-C method type in parentheses.
    ///
    /// \param Quals The Objective-C declaration qualifiers.
    /// \param T The type to print.
    void PrintObjCMethodType(ASTContext &Ctx, Decl::ObjCDeclQualifier Quals,
                             QualType T);

    void PrintObjCTypeParams(ObjCTypeParamList *Params);

  public:
    raw_ostream &Out;

    virtual void PrintStmt(Stmt *S) { PrintStmt(S, Policy.Indentation); }

    virtual void PrintStmt(Stmt *S, int SubIndent) {
      Indentation += SubIndent;
      if (isa_and_nonnull<Expr>(S)) {
        // If this is an expr used in a stmt context, indent and newline it.
        Indent();
        Visit(S);
        Out << ";" << NL;
      } else if (S) {
        Visit(S);
      } else {
        Indent() << "<<<NULL STATEMENT>>>" << NL;
      }
      Indentation -= SubIndent;
    }

    virtual void PrintInitStmt(Stmt *S, unsigned PrefixWidth) {
      // FIXME: Cope better with odd prefix widths.
      Indentation += (PrefixWidth + 1) / 2;
      if (auto *DS = dyn_cast<DeclStmt>(S))
        PrintRawDeclStmt(DS);
      else
        PrintExpr(cast<Expr>(S));
      Out << "; ";
      Indentation -= (PrefixWidth + 1) / 2;
    }

    virtual void PrintControlledStmt(Stmt *S) {
      if (auto *CS = dyn_cast<CompoundStmt>(S)) {
        Out << " ";
        PrintRawCompoundStmt(CS);
        Out << NL;
      } else {
        Out << NL;
        PrintStmt(S);
      }
    }

    virtual void PrintRawCompoundStmt(CompoundStmt *S);
    virtual void PrintRawDecl(Decl *D);
    virtual void PrintRawDeclStmt(const DeclStmt *S);
    virtual void PrintRawIfStmt(IfStmt *If);
    virtual void PrintRawCXXCatchStmt(CXXCatchStmt *Catch);
    virtual void PrintCallArgs(CallExpr *E);
    virtual void PrintRawSEHExceptHandler(SEHExceptStmt *S);
    virtual void PrintRawSEHFinallyStmt(SEHFinallyStmt *S);
    virtual void PrintOMPExecutableDirective(OMPExecutableDirective *S,
                                     bool ForceNoStmt = false);
    virtual void PrintFPPragmas(CompoundStmt *S);

    virtual void PrintExpr(Expr *E) {
      if (E)
        Visit(E);
      else
        Out << "<null expr>";
    }

    virtual void Visit(Stmt* S) {
      if (Helper && Helper->handledStmt(S,Out))
          return;
      StmtVisitor<DeclPrinter>::Visit(S);
    }

    virtual void VisitStmt(Stmt *Node) LLVM_ATTRIBUTE_UNUSED {
      Indent() << "<<unknown stmt type>>" << NL;
    }

    virtual void VisitExpr(Expr *Node) LLVM_ATTRIBUTE_UNUSED {
      Out << "<<unknown expr type>>";
    }

    virtual void VisitCXXNamedCastExpr(CXXNamedCastExpr *Node);

#define ABSTRACT_STMT(CLASS)
#define STMT(CLASS, PARENT) \
    virtual void Visit##CLASS(CLASS *Node);
#include "clang/AST/StmtNodes.inc"

  virtual ~DeclPrinter() = default;

    DeclPrinter(raw_ostream &Out, PrinterHelper *helper, const PrintingPolicy &Policy,
                const ASTContext &Context, unsigned Indentation = 0,StringRef NL = "\n", 
                bool PrintInstantiation = false)
        : Policy(Policy), Context(Context), Indentation(Indentation),
          PrintInstantiation(PrintInstantiation), Helper(helper), NL(NL), Out(Out) {}

    virtual void VisitDeclContext(DeclContext *DC, bool Indent = true);

    virtual void VisitTranslationUnitDecl(TranslationUnitDecl *D);
    virtual void VisitTypedefDecl(TypedefDecl *D);
    virtual void VisitTypeAliasDecl(TypeAliasDecl *D);
    virtual void VisitEnumDecl(EnumDecl *D);
    virtual void VisitRecordDecl(RecordDecl *D);
    virtual void VisitEnumConstantDecl(EnumConstantDecl *D);
    virtual void VisitEmptyDecl(EmptyDecl *D);
    virtual void VisitFunctionDecl(FunctionDecl *D);
    virtual void VisitFriendDecl(FriendDecl *D);
    virtual void VisitFieldDecl(FieldDecl *D);
    virtual void VisitVarDecl(VarDecl *D);
    virtual void VisitLabelDecl(LabelDecl *D);
    virtual void VisitParmVarDecl(ParmVarDecl *D);
    virtual void VisitFileScopeAsmDecl(FileScopeAsmDecl *D);
    virtual void VisitTopLevelStmtDecl(TopLevelStmtDecl *D);
    virtual void VisitImportDecl(ImportDecl *D);
    virtual void VisitStaticAssertDecl(StaticAssertDecl *D);
    virtual void VisitNamespaceDecl(NamespaceDecl *D);
    virtual void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    virtual void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    virtual void VisitCXXRecordDecl(CXXRecordDecl *D);
    virtual void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    virtual void VisitTemplateDecl(const TemplateDecl *D);
    virtual void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    virtual void VisitClassTemplateDecl(ClassTemplateDecl *D);
    virtual void VisitClassTemplateSpecializationDecl(
                                            ClassTemplateSpecializationDecl *D);
    virtual void VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
    virtual void VisitObjCMethodDecl(ObjCMethodDecl *D);
    virtual void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    virtual void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    virtual void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    virtual void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    virtual void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    virtual void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    virtual void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    virtual void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
    virtual void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    virtual void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    virtual void VisitUsingDecl(UsingDecl *D);
    virtual void VisitUsingEnumDecl(UsingEnumDecl *D);
    virtual void VisitUsingShadowDecl(UsingShadowDecl *D);
    virtual void VisitOMPThreadPrivateDecl(OMPThreadPrivateDecl *D);
    virtual void VisitOMPAllocateDecl(OMPAllocateDecl *D);
    virtual void VisitOMPRequiresDecl(OMPRequiresDecl *D);
    virtual void VisitOMPDeclareReductionDecl(OMPDeclareReductionDecl *D);
    virtual void VisitOMPDeclareMapperDecl(OMPDeclareMapperDecl *D);
    virtual void VisitOMPCapturedExprDecl(OMPCapturedExprDecl *D);
    virtual void VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *TTP);
    virtual void VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *NTTP);
    virtual void VisitHLSLBufferDecl(HLSLBufferDecl *D);

    virtual void printTemplateParameters(const TemplateParameterList *Params,
                                 bool OmitTemplateKW = false);
    virtual void printTemplateArguments(llvm::ArrayRef<TemplateArgument> Args,
                                const TemplateParameterList *Params);
    virtual void printTemplateArguments(llvm::ArrayRef<TemplateArgumentLoc> Args,
                                const TemplateParameterList *Params);
    enum class AttrPosAsWritten { Default = 0, Left, Right };
    virtual bool
    prettyPrintAttributes(const Decl *D,
                          AttrPosAsWritten Pos = AttrPosAsWritten::Default);
    virtual void prettyPrintPragmas(Decl *D);
    virtual void printDeclType(QualType T, StringRef DeclName, bool Pack = false);

    bool isImplicitThis(const Expr *E);
    void printPretty(const Stmt* S, raw_ostream &Out, PrinterHelper *Helper,
                           const PrintingPolicy &Policy, unsigned Indentation,
                           StringRef NL, const ASTContext *Context) ;
    void printPrettyControlled(Stmt* S, raw_ostream &Out, PrinterHelper *Helper,
                                     const PrintingPolicy &Policy,
                                     unsigned Indentation, StringRef NL,
                                     const ASTContext *Context) ;
    void print(Decl *D, raw_ostream &Out, const PrintingPolicy &Policy,
                     unsigned Indentation, bool PrintInstantiation) ;
    QualType GetBaseType(QualType T) ;
    QualType getDeclType(Decl* D) ;
    void printGroup(Decl** Begin, unsigned NumDecls,
                          raw_ostream &Out, const PrintingPolicy &Policy,
                          unsigned Indentation) ;
    void printExplicitSpecifier(ExplicitSpecifier ES, llvm::raw_ostream &Out,
                                       PrintingPolicy &Policy, unsigned Indentation,
                                       const ASTContext &Context) ;
  };


#endif