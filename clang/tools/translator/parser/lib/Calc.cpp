#include <string>
#include <iostream>

#include "clang/AST/AST.h"

#include "Param.h"
#include "Shell.h"
#include "Calc.h"
#include "ASTParse.h"
#include "DacppStructure.h"


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

void dacppTranslator::Calc::setBody(Stmt* body) {
    for(Stmt::child_iterator it = body->child_begin(); it != body->child_end(); it++) {
        if(isa<ExprWithCleanups>(*it) && getNode<BinaryOperator>(*it)) {
            this->body.push_back("Expression");
            BinaryOperator* dacExpr = getNode<BinaryOperator>(*it);
            std::vector<std::vector<int>> shapes(getNumParams(), std::vector<int>());
            for (int paramCount = 0; paramCount < getNumParams(); paramCount++) {
                Param* param = getParam(paramCount);
                for(int dimCount = 0; dimCount < param->getDim(); dimCount++) {
                    shapes[paramCount].push_back(param->getShape(dimCount));
                }
            }
            setExpr(dacExpr, shapes);
            continue;
        }
        std::string temp = stmt2String(*it);
        for(int i = 0; i < getNumParams(); i++) {
            if(temp.find(getParam(i)->getName() + ".getShape") != -1) {
                int pos = temp.find(getParam(i)->getName() + ".getShape");
                int dim = (int)temp[pos + getParam(i)->getName().size() + 10] - (int)'0';
                temp.replace(pos, getParam(i)->getName().size() + 12, std::to_string(getParam(i)->getShape(dim)));
            }
            if(temp.find(getParam(i)->getName()) != -1 && getParam(i)->getDim() == 0) {
                int pos = temp.find(getParam(i)->getName());
                temp.replace(pos, getParam(i)->getName().size(), getParam(i)->getName() + "[0]");
            }
        }
        this->body.push_back(temp);
    }
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

        for(int i = 0; i < shell->getNumSplits(); i++) {
            if(shell->getSplit(i)->type == "dacpp::RegularSplit") {
                RegularSplit* sp = static_cast<RegularSplit*>(shell->getSplit(i));
                param->setShape(sp->getSplitSize());
            }
            else if(shell->getSplit(i)->type == "dacpp::IndexSplit") {
                IndexSplit* sp = static_cast<IndexSplit*>(shell->getSplit(i));
            } else {
                param->setShape(shellParam->getShape(i));
            }
        }
        setParam(param);
    }
    setBody(calcFunc->getBody());
}