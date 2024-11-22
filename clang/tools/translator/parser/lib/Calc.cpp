#include <string>

#include "clang/AST/Attr.h"
#include "clang/AST/DeclFriend.h"
#include "clang/Basic/Module.h"


#include "ASTParse.h"
#include "Calc.h"
#include "DacppStructure.h"
#include "Param.h"
#include "Shell.h"





/**
 * 存储计算结构信息类实现
 */
dacppTranslator::Calc::Calc() {
}

void dacppTranslator::Calc::setName(std::string name) {
    this->name = name;
}
    
std::string dacppTranslator::Calc::getName() {
    return name;
}

void dacppTranslator::Calc::setParam(Param* param) {
    params.push_back(param);
}

dacppTranslator::Param* dacppTranslator::Calc::getParam(int idx) {
    return params[idx];
}

int dacppTranslator::Calc::getNumParams() {
    return params.size();
}

struct Visitor : DeclPrinter {
    ASTContext &Context;
    dacppTranslator::Calc *calc;
    PrinterHelper *helper;
    const PrintingPolicy &Policy;
    raw_ostream &os;
  
    Visitor(dacppTranslator::Calc *calc, raw_ostream &os, PrinterHelper *helper,
                const PrintingPolicy &Policy, unsigned Indentation = 0,
                StringRef NL = "\n", ASTContext *Context = nullptr)
        : DeclPrinter(os, helper, Policy, *Context, Indentation, NL), Context(*Context)
        , calc(calc), helper(helper), Policy(Policy), os(os) {}

    void Visit(Stmt* S) override {
      bool bHandled = false; /* 是否被处理的标志。  */
      BinaryOperator* dacExpr;
        if (isa<ExprWithCleanups>(S) &&
           (dacExpr = dacppTranslator::getNode<BinaryOperator>(S)) &&
           dacExpr->getOpcode() == BO_LMG) {
            calc->body.push_back("Expression");
            std::vector<std::vector<int>> shapes(calc->getNumParams(), std::vector<int>());
            for (int paramCount = 0; paramCount < calc->getNumParams(); paramCount++) {
                dacppTranslator::Param* param = calc->getParam(paramCount);
                for(int dimCount = 0; dimCount < param->getDim(); dimCount++) {
                    shapes[paramCount].push_back(param->getShape(dimCount));
                }
            }
            calc->setExpr(dacExpr, shapes);
            /* 已被处理。  */
            bHandled = true;
        }
      /* 如果未得到处理，则使用默认方法处理它。  */
      if (!bHandled)
        DeclPrinter::Visit(S);
    }

void VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) override {
  bool bHandled = false; /* 是否被处理的标志。  */
  std::string Msg;
  llvm::raw_string_ostream SS(Msg);
  if (const auto *Call = dyn_cast<CallExpr>(Node)) {
    if (auto *Callee = Call->getCallee()) {
      if (auto *ME = dyn_cast<MemberExpr>(Callee)) {
        if (!isImplicitThis(ME->getBase())) {
          if (const Expr *Base = ME->getBase()) {
            if (auto *DRE = dyn_cast<DeclRefExpr>(Base)) {
              if (strstr (DRE->getDecl()->getType().getAsString().c_str(), "Tensor") &&
                  ! strcmp ("getShape", ME->getMemberNameInfo().getAsString().c_str())) {
                /* 已被处理。  */
                bHandled = true;

                for(int i = 0; i < calc->getNumParams(); i++) {
                  if (!strcmp (DRE->getNameInfo().getAsString().c_str(), calc->getParam(i)->getName().c_str())) {
                    Call->getArg(0)->printPretty(SS, helper, Policy);
                    os << calc->getParam(i)->getShape(atoi(SS.str().c_str()));
                    break;
                  }
                }

              }
            }
          }
        }
      }
    }
  }

  /* 如果未得到处理，则使用默认方法处理它。  */
  if (!bHandled)
      DeclPrinter::VisitCXXMemberCallExpr(Node);
}

} ;


void dacppTranslator::Calc::setBody(Stmt* body) {
  std::string Msg;
  llvm::raw_string_ostream SS(Msg);
    Visitor V (this, SS, nullptr, PrintingPolicy (LangOptions ()), 0, "\n", nullptr);
    V .Visit (body);
    this->body.push_back(Msg);
}

std::string dacppTranslator::Calc::getBody(int idx) {
    return body[idx];
}

int dacppTranslator::Calc::getNumBody() {
    return body.size();
}

void dacppTranslator::Calc::setExpr(const BinaryOperator* dacExpr, std::vector<std::vector<int>> shapes) {
    Shell* shell = new Shell();
    Calc* calc = new Calc();
    Expression* expr = new Expression();
    expr->setDacExpr(dacExpr);
    expr->setShell(shell);
    expr->setCalc(calc);
    shell->setFather(expr);
    calc->setFather(expr);
    shell->parseShell(dacExpr, shapes);
    calc->parseCalc(dacExpr);
    exprs.push_back(expr);
}

dacppTranslator::Expression* dacppTranslator::Calc::getExpr(int idx) {
    return exprs[idx];
}

int dacppTranslator::Calc::getNumExprs() {
    return exprs.size();
}

void dacppTranslator::Calc::setFather(Expression* expr) {
    this->father = expr;
}

dacppTranslator::Expression* dacppTranslator::Calc::getFather() {
    return father;
}

void dacppTranslator::Calc::setCalcLoc(FunctionDecl* calcLoc) {
    this->calcLoc = calcLoc;
}
FunctionDecl* dacppTranslator::Calc::getCalcLoc() {
    return calcLoc;
}

// 解析Calc节点，将解析到的信息存储到Calc类中
void dacppTranslator::Calc::parseCalc(const BinaryOperator* dacExpr) {
    Shell* shell = getFather()->getShell();

    // 获取 DAC 数据关联表达式右值
    Expr* dacExprRHS = dacExpr->getRHS();
    FunctionDecl* calcFunc = dyn_cast<FunctionDecl>(dyn_cast<DeclRefExpr>(dacExprRHS)->getDecl());
    
    // 设置 AST 中 Calc 节点的位置
    setCalcLoc(calcFunc);

    // 设置 Calc 函数名称
    setName(calcFunc->getNameAsString());

    // 设置 Calc 参数列表
    for(unsigned int paramsCount = 0; paramsCount < calcFunc->getNumParams(); paramsCount++) {
        Param* param = new Param();

        // 获取参数读写属性
        param->setRw(inputOrOutput(calcFunc->getParamDecl(paramsCount)->getType().getAsString()));

        // 设置参数类型
        std::string type = calcFunc->getParamDecl(paramsCount)->getType().getAsString();
        param->setType(type);
        
        // 设置参数名称
        param->setName(calcFunc->getParamDecl(paramsCount)->getNameAsString());
        
        // 设置参数形状
        ShellParam* shellParam = shell->getShellParam(paramsCount);

        for (int i = 0; i < shellParam->getNumSplit(); i++) {
            Split* sp = shellParam->getSplit(i);
            if(sp->type.compare("RegularSplit") == 0) {
                RegularSplit* rsp = static_cast<RegularSplit*>(sp);
                param->setShape(rsp->getSplitSize());
            } else if(sp->type.compare("IndexSplit") == 0) {
                IndexSplit* isp = static_cast<IndexSplit*>(sp);
            } else {
                param->setShape(shellParam->getShape(i));
            }
        }
        if (param->getDim() == 0) {
          param->setShape(1);
        }
        setParam(param);
    }
    setBody(calcFunc->getBody());
}