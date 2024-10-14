#include <string>

#include "clang/AST/AST.h"

#include "Param.h"
#include "DacppStructure.h"
#include "ASTParse.h"

using namespace clang;

/*
    头文件
*/
dacppTranslator::HeaderFile::HeaderFile() {
}

dacppTranslator::HeaderFile::HeaderFile(std::string name) {
    this->name = name;
}

void dacppTranslator::HeaderFile::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::HeaderFile::getName() {
    return name;
}

/*
    命名空间
*/
dacppTranslator::NameSpace::NameSpace() {
}

dacppTranslator::NameSpace::NameSpace(std::string name) {
    this->name = name;
}

void dacppTranslator::NameSpace::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::NameSpace::getName() {
    return name;
}

/*
    划分结构
*/
dacppTranslator::Shell::Shell() {
}

void dacppTranslator::Shell::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::Shell::getName() {
    return name;
}

void dacppTranslator::Shell::setParam(Param param) {
    params.push_back(param);
}

dacppTranslator::Param dacppTranslator::Shell::getParam(int idx) {
    return params[idx];
}

int dacppTranslator::Shell::getNumParams() {
    return params.size();
}

void dacppTranslator::Shell::setSplit(Split split) {
    splits.push_back(split);
}

dacppTranslator::Split dacppTranslator::Shell::getSplit(int idx) {
    return splits[idx];
}

int dacppTranslator::Shell::getNumSplits() {
    return splits.size();
}

void dacppTranslator::Shell::setShellParam(ShellParam param) {
    shellParams.push_back(param);
}

dacppTranslator::ShellParam dacppTranslator::Shell::getShellParam(int idx) {
    return shellParams[idx];
}

int dacppTranslator::Shell::getNumShellParams() {
    return shellParams.size();
}

void dacppTranslator::Shell::setExpr(Expression* expr) {
    this->expr = expr;
}

dacppTranslator::Expression* dacppTranslator::Shell::getExpr() {
    return expr;
}

void dacppTranslator::Shell::setShellLoc(FunctionDecl* shellLoc) {
    this->shellLoc = shellLoc;
}

FunctionDecl* dacppTranslator::Shell::getShellLoc() {
    return shellLoc;
}

void dacppTranslator::Shell::parseShell(const BinaryOperator* dacExpr) {
    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacExpr->getLHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    FunctionDecl* shellFunc = shellCall->getDirectCallee();
    
    // 设置 AST 中 Shell 节点的位置
    setShellLoc(shellFunc);
    
    // 获取实参的形状
    std::vector<std::vector<int>> shapes(shellFunc->getNumParams());
    for(int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {
        Expr* curExpr = shellCall->getArg(paramsCount);
        DeclRefExpr* declRefExpr;
        if(isa<DeclRefExpr>(curExpr)) {
            declRefExpr = dyn_cast<DeclRefExpr>(curExpr);
        }
        else {
            declRefExpr = getNode<DeclRefExpr>(curExpr);
        }
        int count = 0;
    
        for(Stmt::child_iterator it = dyn_cast<VarDecl>(declRefExpr->getDecl())->getInit()->child_begin(); it != dyn_cast<VarDecl>(declRefExpr->getDecl())->getInit()->child_end(); it++) {
            if(count != 1) {
                count++;
                continue;
            }
            VarDecl* shapeDecl = dyn_cast<VarDecl>(getNode<DeclRefExpr>(*it)->getDecl());
            InitListExpr* initListExpr = getNode<InitListExpr>(shapeDecl->getInit());
            for(Stmt::child_iterator it = initListExpr->child_begin(); it != initListExpr->child_end(); it++) {
                IntegerLiteral* integer = dyn_cast<IntegerLiteral>(*it);
                shapes[paramsCount].push_back(std::stoi(integer->getValue().toString(10, true)));
            }
            count++;
        }
    }

    // 设置 Shell 函数名称
    setName(shellFunc->getNameAsString());

    // 设置 Shell 参数列表
    for(unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {
        Param param;

        // 获取参数读写属性
        param.setRw(inputOrOutput(shellFunc->getParamDecl(paramsCount)->getType().getAsString()));

        // 设置参数类型
        std::string type = shellFunc->getParamDecl(paramsCount)->getType().getAsString();
        param.setType(type);
        
        // 设置参数名称
        param.setName(shellFunc->getParamDecl(paramsCount)->getNameAsString());
        
        // 设置参数形状
        for(int i = 0; i < shapes[paramsCount].size(); i++) {
            param.setShape(shapes[paramsCount][i]);
        }

        setParam(param);
    }

    // 获取划分结构函数体
    Stmt* shellFuncBody = shellFunc->getBody();
    /*
        InitList 节点下保存划分结构初始化节点列表
        
    */
    for(Stmt::child_iterator it = shellFuncBody->child_begin(); it != shellFuncBody->child_end(); it++) {
        if(!isa<DeclStmt>(*it)) { continue; }
        Decl* curDecl = dyn_cast<DeclStmt>(*it)->getSingleDecl();
        if(!isa<VarDecl>(curDecl)) { continue; }
        VarDecl* curVarDecl = dyn_cast<VarDecl>(curDecl);
        if(curVarDecl->getType().getAsString().compare("dacpp::list") == 0) {
            InitListExpr* ILE = getNode<InitListExpr>(*it);
            for(unsigned int i = 0; i < ILE->getNumInits(); i++) {
                ShellParam shellParam;
                Expr* curExpr = ILE->getInit(i);
                std::vector<Expr*> astExprs;
                std::string name = "zjx";
                getSplitExpr(curExpr, name, astExprs);
                shellParam.setName(name);
                for(unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {            
                    if(shellParam.getName() != shellFunc->getParamDecl(paramsCount)->getNameAsString()) { continue; }
                    shellParam.setRw(inputOrOutput(shellFunc->getParamDecl(paramsCount)->getType().getAsString()));
                    shellParam.setType(getParam(paramsCount).getType());
                    shellParam.setName(getParam(paramsCount).getName());
                    for(int shapeIdx = 0; shapeIdx < getParam(paramsCount).getDim(); shapeIdx++) {
                        shellParam.setShape(getParam(paramsCount).getShape(shapeIdx));
                    }
                }
                for(unsigned int i = 0; i < astExprs.size(); i++) {
                    if(getNode<DeclRefExpr>(astExprs[i])) {
                        VarDecl* vd = dyn_cast<VarDecl>(getNode<DeclRefExpr>(astExprs[i])->getDecl());
                        if(vd->getType().getAsString().compare("dacpp::RegularSplit") == 0) {
                            RegularSplit sp;
                            sp.setId(getNode<StringLiteral>(vd->getInit())->getString().str());
                            sp.setDimIdx(i);
                            CXXConstructExpr* cXXConstCastExpr = getNode<CXXConstructExpr>(vd->getInit());
                            int count = 0;
                            for(Stmt::child_iterator it = cXXConstCastExpr->child_begin(); it != cXXConstCastExpr->child_end(); it++) {
                                if(count == 1) {
                                    sp.setSplitSize(std::stoi((dyn_cast<IntegerLiteral>(*it))->getValue().toString(10, true)));
                                } else if(count == 2) {
                                    sp.setSplitStride(std::stoi((dyn_cast<IntegerLiteral>(*it))->getValue().toString(10, true)));
                                }
                                count++;
                            }
                            sp.setSplitNumber(shellParam.getShape(i));
                            shellParam.setSplit(sp);
                        } else if(vd->getType().getAsString().compare("dacpp::Index") == 0) {
                            IndexSplit sp;
                            sp.setId(getNode<StringLiteral>(vd->getInit())->getString().str());
                            sp.setDimIdx(i);
                            sp.setSplitNumber(shellParam.getShape(i));
                            shellParam.setSplit(sp);
                        } 
                    } else {
                        Split sp;
                        sp.setId("void");
                        sp.setDimIdx(i);
                        shellParam.setSplit(sp);
                    }
                    /*
                    for(int j = 0; j < shell->getNumOps(); j++) {
                        if(op.getName() != shell->getOp(j).getName()) { continue; }
                        shell->setOpSplitSize(j, shellParam.getShape(i));
                    }
                    shellParam.setParams(op);
                    */
                }
                setShellParam(shellParam);
            }
        }
        /*
        // 设置 Shell 划分列表
        else if(curVarDecl->getType().getAsString().compare("int") == 0) {
            DacOp op;                 
            op.setType(curVarDecl->getType().getAsString());
            op.setName(curVarDecl->getNameAsString());
            shell->setOps(op);
        }
        */
    }

}

/*
    计算
*/
dacppTranslator::Calc::Calc() {
}

void dacppTranslator::Calc::setName(std::string name) {
    this->name = name;
}
    
std::string dacppTranslator::Calc::getName() {
    return name;
}

void dacppTranslator::Calc::setParam(Param param) {
    params.push_back(param);
}

dacppTranslator::Param dacppTranslator::Calc::getParam(int idx) {
    return params[idx];
}

int dacppTranslator::Calc::getNumParams() {
    return params.size();
}

void dacppTranslator::Calc::setBody(Stmt* body) {

}

std::string dacppTranslator::Calc::getBody(int idx) {
    return body[idx];
}

int dacppTranslator::Calc::getNumBody() {
    return body.size();
}

void dacppTranslator::Calc::setExpression(const BinaryOperator* dacExpr) {

}

dacppTranslator::Expression* dacppTranslator::Calc::getExpression(int idx) {
    return exprs[idx];
}

int dacppTranslator::Calc::getNumExpressions() {
    return exprs.size();
}

void dacppTranslator::Calc::setExpr(Expression* expr) {
    this->expr = expr;
}

dacppTranslator::Expression* dacppTranslator::Calc::getExpr() {
    return expr;
}

void dacppTranslator::Calc::setCalcLoc(FunctionDecl* calcLoc) {
    this->calcLoc = calcLoc;
}
FunctionDecl* dacppTranslator::Calc::getCalcLoc() {
    return calcLoc;
}

#include "../test/test.h"
void dacppTranslator::Calc::parseCalc(const BinaryOperator* dacExpr) {

    Shell* shell = getExpr()->getShell();

    // 获取 DAC 数据关联表达式右值
    Expr* dacExprRHS = dacExpr->getRHS();
    FunctionDecl* calcFunc = dyn_cast<FunctionDecl>(dyn_cast<DeclRefExpr>(dacExprRHS)->getDecl());
    
    // 设置 AST 中 Calc 节点的位置
    setCalcLoc(calcFunc);

    // 设置 Calc 函数名称
    setName(calcFunc->getNameAsString());

    // 设置 Calc 参数列表
    for(unsigned int paramsCount = 0; paramsCount < calcFunc->getNumParams(); paramsCount++) {
        Param param;

        // 设置参数类型
        std::string type = calcFunc->getParamDecl(paramsCount)->getType().getAsString();
        param.setType(type);
        
        // 设置参数名称
        param.setName(calcFunc->getParamDecl(paramsCount)->getNameAsString());
        
        Expr* curExpr = ILE->getInit(paramsCount);
        ShellParam shellParam = shell->getShellParam(paramsCount);

        std::vector<InitListExpr*> initListExprs;
        // getSlice(shellParam, curExpr, initListExprs);
        for(unsigned int i = 0; i < initListExprs.size(); i++) {
            if(getNode<DeclRefExpr>(initListExprs[i])) {
            }
            else {
                param.setShape(shellParam.getShape(i));
            }
        }

        setParam(param);
    }

    // setBody(calcFunc->getBody());
}

/*
    数据关联计算表达式
*/
dacppTranslator::Expression::Expression() {
}

void dacppTranslator::Expression::setShell(Shell* shell) {
    this->shell = shell;
}

dacppTranslator::Shell* dacppTranslator::Expression::getShell() {
    return shell;
}

void dacppTranslator::Expression::setCalc(Calc* calc) {
    this->calc;
}

dacppTranslator::Calc* dacppTranslator::Expression::getCalc() {
    return calc;
}

/*
    DACPP 文件
*/
dacppTranslator::DacppFile::DacppFile() {
    setHeaderFile("<sycl/sycl.hpp>");
    setHeaderFile("\"../DataReconstructor.cpp\"");
    setNameSpace("sycl");
}

void dacppTranslator::DacppFile::setHeaderFile(std::string headerFile) {
    headerFiles.push_back(headerFile);
}

dacppTranslator::HeaderFile dacppTranslator::DacppFile::getHeaderFile(int idx) {
    return headerFiles[idx];
}

int dacppTranslator::DacppFile::getNumHeaderFile() {
    return headerFiles.size();
}

void dacppTranslator::DacppFile::setNameSpace(std::string nameSpace) {
    nameSpaces.push_back(nameSpace);
}

dacppTranslator::NameSpace dacppTranslator::DacppFile::getNameSpace(int idx) {
    return nameSpaces[idx];
}

int dacppTranslator::DacppFile::getNumNameSpace() {
    return nameSpaces.size();
}

void dacppTranslator::DacppFile::setExpression(const BinaryOperator* dacExpr) {
    Shell* shell = new Shell();
    Calc* calc = new Calc();
    Expression* expr = new Expression();
    expr->setShell(shell);
    expr->setCalc(calc);
    shell->setExpr(expr);
    calc->setExpr(expr);
    shell->parseShell(dacExpr);
    calc->parseCalc(dacExpr);
    exprs.push_back(expr);
}

dacppTranslator::Expression* dacppTranslator::DacppFile::getExpression(int idx) {
    return exprs[idx];
}

int dacppTranslator::DacppFile::getNumExpression() {
    return exprs.size();
}

void dacppTranslator::DacppFile::setMainFuncLoc(const FunctionDecl* mainFuncLoc) {
    this->mainFuncLoc = mainFuncLoc;
}

const FunctionDecl* dacppTranslator::DacppFile::getMainFuncLoc() {
    return mainFuncLoc;
}