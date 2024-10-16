#ifndef TRANSLATOR_PARSER_DACPPSTRUCTURE_H
#define TRANSLATOR_PARSER_DACPPSTRUCTURE_H

#include <string>
#include <vector>

#include "clang/AST/AST.h"

#include "Split.h"
#include "Param.h"

using namespace clang;

namespace dacppTranslator {

/*
    头文件
*/
class HeaderFile {

private:
    std::string name;

public:
    HeaderFile();
    HeaderFile(std::string name);

    void setName(std::string name);
    std::string getName();

};

/*
    命名空间
*/
class NameSpace {

private:
    std::string name;

public:
    NameSpace();
    NameSpace(std::string name);

    void setName(std::string name);
    std::string getName();

};

/*
    数据关联计算表达式
*/
class Expression;

/*
    划分结构
*/
class Shell {

// rewriter_->ReplaceText(shellFunc->getSourceRange(), "");

private:
    std::string name; // 函数名
    std::vector<Param*> params; // 参数列表
    std::vector<Split*> splits; // 划分列表
    std::vector<ShellParam*> shellParams; // 划分结构参数
    Expression* expr; // 所属的数据管理计算表达式
    FunctionDecl* shellLoc; // AST 中 Shell 节点的位置

public:
    Shell();

    void setName(std::string name);
    std::string getName();

    void setParam(Param* param);
    Param* getParam(int idx);
    int getNumParams();

    void setSplit(Split* split);
    Split* getSplit(int idx);
    int getNumSplits();

    void setShellParam(ShellParam* param);
    ShellParam* getShellParam(int idx);
    int getNumShellParams();

    void setExpr(Expression* expr);
    Expression* getExpr();

    void setShellLoc(FunctionDecl* expr);
    FunctionDecl* getShellLoc();

    void parseShell(const BinaryOperator* dacExpr);

};

/*
    计算
*/
class Calc {

private:
    std::string name; // 函数名
    std::vector<Param*> params; // 参数
    std::vector<std::string> body; // 函数体
    std::vector<Expression*> exprs; // 数据关联计算表达式
    Expression* expr; // 所属的数据管理计算表达式
    FunctionDecl* calcLoc; // AST 中 Calc 节点的位置

public:
    Calc();

    void setName(std::string name);
    std::string getName();

    void setParam(Param* param);
    Param* getParam(int idx);
    int getNumParams();

    void setBody(Stmt* body);
    std::string getBody(int idx);
    int getNumBody();

    void setExpression(const BinaryOperator* dacExpr);
    Expression* getExpression(int idx);
    int getNumExpressions();

    void setExpr(Expression* expr);
    Expression* getExpr();

    void setCalcLoc(FunctionDecl* calcLoc);
    FunctionDecl* getCalcLoc();

    void parseCalc(const BinaryOperator* dacExpr);

};

/*
    数据关联计算表达式
*/
class Expression {

private:
    Shell* shell;
    Calc* calc;

public:
    Expression();

    void setShell(Shell* shell);
    Shell* getShell();

    void setCalc(Calc* calc);
    Calc* getCalc();

};

/*
    DACPP 文件
*/
class DacppFile {

private:
    std::vector<HeaderFile*> headerFiles;
    std::vector<NameSpace*> nameSpaces;
    std::vector<Expression*> exprs;
    const FunctionDecl* mainFuncLoc;

public:
    DacppFile();
    
    void setHeaderFile(std::string headerfile);
    HeaderFile* getHeaderFile(int idx);
    int getNumHeaderFile();

    void setNameSpace(std::string nameSpace);
    NameSpace* getNameSpace(int idx);
    int getNumNameSpace();

    void setExpression(const BinaryOperator* dacExpr);
    Expression* getExpression(int idx);
    int getNumExpression();

    void setMainFuncLoc(const FunctionDecl* mainFuncLoc);
    const FunctionDecl* getMainFuncLoc();

};

}

#endif