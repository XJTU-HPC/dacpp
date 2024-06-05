#include <string>
#include <vector>
#include <sstream>

#include "clang/AST/AST.h"

#include "rewriteLib.h"
#include "ASTLib.hpp"

using namespace clang;

void dacpp::Source2Source::setHeaderFile(std::string file) {
    HeaderFile headerfile(file);
    headerFiles_.push_back(headerfile);
}

void dacpp::Source2Source::setCalc(FunctionDecl* calcFunc) {
    Calc calc; 
    calc.setName(calcFunc->getNameAsString());
    for(int paramsCount = 0; paramsCount < calcFunc->getNumParams(); paramsCount++) {
        Param param;
        std::string type = calcFunc->getParamDecl(paramsCount)->getType().getAsString();
        if(type == "Tensor<int>" || type == "dacpp::Tensor<int>" || type == "Tensor<int> &" || type == "dacpp::Tensor<int> &") {
            param.setType("int");
        }
        else {
            param.setType("float");
        }
        param.setName(calcFunc->getParamDecl(paramsCount)->getNameAsString());
        calc.setParams(param);
    }
    calc.setBody(stmt2String(calcFunc->getBody()));
    calcs_.push_back(calc);
    rewriter_->ReplaceText(calcFunc->getSourceRange(), "");
}

void dacpp::Source2Source::setShell(const BinaryOperator* dacExpr) {
    Shell shell;
    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacExpr->getLHS();
    // 需要检查数据关联表达式语法
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    FunctionDecl* shellFunc = shellCall->getDirectCallee();
    shell.setName(shellFunc->getNameAsString());
    for(int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {
        Param param;
        param.setType(shellFunc->getParamDecl(paramsCount)->getType().getAsString());
        param.setName(shellFunc->getParamDecl(paramsCount)->getNameAsString());
        shell.setParams(param);
    }
    // 获取 DAC 数据关联表达式右值
    Expr* dacExprRHS = dacExpr->getRHS();
    // 需要检查数据关联表达式语法
    FunctionDecl* calcFunc = dyn_cast<FunctionDecl>(dyn_cast<DeclRefExpr>(dacExprRHS)->getDecl());
    shell.setCalcName(calcFunc->getNameAsString());

    // 获取数据划分信息
    Stmt* shellFuncBody = shellFunc->getBody();
    for(Stmt::child_iterator it = shellFuncBody->child_begin(); it != shellFuncBody->child_end(); it++) {
        if(!isa<DeclStmt>(*it)) { continue; }
        Decl* curDecl = dyn_cast<DeclStmt>(*it)->getSingleDecl();
        if(!isa<VarDecl>(curDecl)) { continue; }
        VarDecl* curVarDecl = dyn_cast<VarDecl>(curDecl);
        if(!curVarDecl->getType().getAsString().compare("dacpp::list") == 0) { continue; }
        InitListExpr* ILE = getNode<InitListExpr>(*it);
        for(int i = 0; i < ILE->getNumInits(); i++) {
            ShellParam shellParam;
            Expr* curExpr = ILE->getInit(i);
            std::vector<InitListExpr*> initListExprs;
            getSlice(shellParam, curExpr, initListExprs);
            for(int i = 0; i < initListExprs.size(); i++) {
                Param param;
                if(getNode<DeclRefExpr>(initListExprs[i])) {
                    VarDecl* vd = dyn_cast<VarDecl>(getNode<DeclRefExpr>(initListExprs[i])->getDecl());                    
                    param.setType(vd->getType().getAsString());
                    param.setName(vd->getNameAsString());
                }
                shellParam.setParams(param);
            }
            for(int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {            
                if(shellParam.getName() != shellFunc->getParamDecl(paramsCount)->getNameAsString()) { continue; }
                shellParam.setRw(inputOrOutput(shellFunc->getParamDecl(paramsCount)->getType().getAsString()));
            }
            shell.setShellParams(shellParam);
        }
    }
    
    // 测试代码
    // for(int i = 0; i < shell.getNumShellParams(); i++) {
    //     llvm::outs() << shell.getShellParam(i).getRw() << "\n";
    //     llvm::outs() << shell.getShellParam(i).getName() << "\n";
    //     for(int j = 0; j < shell.getShellParam(i).getNumOps(); j++) {
    //         llvm::outs() << shell.getShellParam(i).getOp(j).getType() << " ";
    //         llvm::outs() << shell.getShellParam(i).getOp(j).getName() << " ";
    //     }
    //     llvm::outs() << "\n";
    // }
    shells_.push_back(shell);
    rewriter_->ReplaceText(shellFunc->getSourceRange(), "");
}

void dacpp::Source2Source::rewriteDac() {
    std::stringstream sstream;
    for(int i = 0; i < getNumHeaderFiles(); i++) {
        sstream << getHeaderFile(i).getFile() << "\n";
    }
    for(int i = 0; i < getNumCalcs(); i++) {
        Calc calc = getCalc(i);
        sstream << "void " << calc.getName() << "(";
        for(int j = 0; j < calc.getNumParams(); j++) {
            sstream << calc.getParam(j).getType() << " " << calc.getParam(j).getName();
            if(j != calc.getNumParams() - 1) sstream << ", ";
        }
        sstream << ")\n";
        sstream << calc.getBody() + "\n";
    }
    
    std::string strResult = sstream.str() + "\n";
    rewriter_->InsertText(mainFunc_->getBeginLoc(), strResult);
}

void dacpp::Source2Source::recursiveRewriteMain(Stmt* curStmt) {
    if(isa<BinaryOperator>(curStmt)) {
        BinaryOperator* binaryOperator = dyn_cast<BinaryOperator>(curStmt);
        if(binaryOperator->getOpcodeStr().equals("<->")) {
            // 获取 DAC 数据关联表达式左值
            Expr* dacExprLHS = binaryOperator->getLHS();
            CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
            rewriter_->ReplaceText(binaryOperator->getSourceRange(), stmt2String(shellCall));
        }
        return;
    }
    // 修改 Tensor 多重方括号调用
    
    for(Stmt::child_iterator it = curStmt->child_begin(); it != curStmt->child_end(); it++) {
        recursiveRewriteMain(*it);
    }
}

void dacpp::Source2Source::rewriteMain() {
    Stmt* mainFuncBody = mainFunc_->getBody();
    recursiveRewriteMain(mainFuncBody);
}