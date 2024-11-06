#include <string>

#include "clang/AST/AST.h"

#include "Param.h"
#include "DacppStructure.h"
#include "ASTParse.h"
#include "/data/zjx/dacpp/clang/tools/translator/test/test.h"


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

void dacppTranslator::Shell::setParam(Param* param) {
    params.push_back(param);
}

dacppTranslator::Param* dacppTranslator::Shell::getParam(int idx) {
    return params[idx];
}

int dacppTranslator::Shell::getNumParams() {
    return params.size();
}

void dacppTranslator::Shell::setSplit(Split* split) {
    splits.push_back(split);
}

dacppTranslator::Split* dacppTranslator::Shell::getSplit(int idx) {
    return splits[idx];
}

int dacppTranslator::Shell::getNumSplits() {
    return splits.size();
}

void dacppTranslator::Shell::setShellParam(ShellParam* param) {
    shellParams.push_back(param);
}

dacppTranslator::ShellParam* dacppTranslator::Shell::getShellParam(int idx) {
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

// 解析Shell
void dacppTranslator::Shell::parseShell(const BinaryOperator* dacExpr) {
    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacExpr->getLHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    FunctionDecl* shellFunc = shellCall->getDirectCallee();
    
    // 设置 AST 中 Shell 节点的位置
    setShellLoc(shellFunc);
    
    // 获取实参的形状
    // 这里的实参目前指的是 DACPP:Tensor
    // TODO 
    // 实参形状这里直接在定义实参的位置找到了实参的初始化列表，在翻译的时候将实参形状硬编码到了函数中，如果多次调用同一函数，如果实参不同，会生成多个SYCL函数
    // Tensor的初始化用到了两个std::vector，如果在生成之后对其进行了push_back，则不能得到正确的形状，会出现bug
    // 如果Tensor初始化列表中的vector是从文件中读取的，也无法获得正确形状，这种情况需要在SYCL文件中把形状修改为软编码，比如Tensor.getShape(idx)
    std::vector<std::vector<int>> shapes(shellFunc->getNumParams());
    for(unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {
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
        Param* param = new Param();

        // 获取参数读写属性
        param->setRw(inputOrOutput(shellFunc->getParamDecl(paramsCount)->getType().getAsString()));

        // 设置参数类型
        std::string type = shellFunc->getParamDecl(paramsCount)->getType().getAsString();
        param->setType(type);
        
        // 设置参数名称
        param->setName(shellFunc->getParamDecl(paramsCount)->getNameAsString());
        
        // 设置参数形状
        for(unsigned int i = 0; i < shapes[paramsCount].size(); i++) {
            param->setShape(shapes[paramsCount][i]);
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
                ShellParam* shellParam = new ShellParam();
                Expr* curExpr = ILE->getInit(i);
                std::vector<Expr*> astExprs;

                std::string name = "";
                getSplitExpr(curExpr, name, astExprs);
                shellParam->setName(name);
                for(unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {            
                    if(shellParam->getName() != shellFunc->getParamDecl(paramsCount)->getNameAsString()) { continue; }
                    shellParam->setRw(inputOrOutput(shellFunc->getParamDecl(paramsCount)->getType().getAsString()));
                    shellParam->setType(getParam(paramsCount)->getType());
                    shellParam->setName(getParam(paramsCount)->getName());
                    for(int shapeIdx = 0; shapeIdx < getParam(paramsCount)->getDim(); shapeIdx++) {
                        shellParam->setShape(getParam(paramsCount)->getShape(shapeIdx));
                    }
                }
                for(unsigned int i = 0; i < astExprs.size(); i++) {
                    if(getNode<DeclRefExpr>(astExprs[i])) {
                        VarDecl* vd = dyn_cast<VarDecl>(getNode<DeclRefExpr>(astExprs[i])->getDecl());

                        if(vd->getType().getAsString().compare("dacpp::RegularSplit") == 0) {
                            RegularSplit* sp = new RegularSplit();
                            sp->type = "dacpp::RegularSplit";
                            sp->setId(getNode<StringLiteral>(vd->getInit())->getString().str());
                            sp->setDimIdx(i);
                            CXXConstructExpr* CCE = getNode<CXXConstructExpr>(vd->getInit());
                            int count = 0;
                            for (CXXConstructExpr::arg_iterator I = CCE->arg_begin(),
                                                                E = CCE->arg_end();
                                I != E; ++I) {
                                if(count == 1) {
                                    /* TODO: 计算常量表达式的值。  */
                                    sp->setSplitSize(std::stoi((dyn_cast<IntegerLiteral>(*I))->getValue().toString(10, true)));
                                } else if(count == 2) {
                                    /* TODO: 计算常量表达式的值。  */
                                    sp->setSplitStride(std::stoi((dyn_cast<IntegerLiteral>(*I))->getValue().toString(10, true)));
                                }
                                count++;
                            }
                            /* TODO: 处理除不尽的情况。  */
                            sp->setSplitNumber((shellParam->getShape(i) - sp->getSplitSize() + 1) / sp->getSplitStride());
                            shellParam->setSplit(sp);
                        } else if(vd->getType().getAsString().compare("dacpp::Index") == 0) {
                            IndexSplit* sp = new IndexSplit();
                            sp->type = "dacpp::IndexSplit";
                            sp->setId(getNode<StringLiteral>(vd->getInit())->getString().str());
                            sp->setDimIdx(i);
                            sp->setSplitNumber(shellParam->getShape(i));
                            shellParam->setSplit(sp);
                        } 
                    } else {
                        Split* sp = new Split();
                        sp->type = "zjx";
                        sp->setId("void");
                        sp->setDimIdx(i);
                        shellParam->setSplit(sp);
                    }
                }
                setShellParam(shellParam);
            }
        }
        // 设置 Shell 划分列表
        else if(curVarDecl->getType().getAsString().compare("dacpp::Index") == 0) {
            IndexSplit* sp = new IndexSplit();                 
            sp->setId("i");
            sp->setSplitNumber(10);
            setSplit(sp);
        }
        else if(curVarDecl->getType().getAsString().compare("dacpp::RegularSplit") == 0) {
            RegularSplit* sp = new RegularSplit();                 
            sp->setId("i");
            sp->setSplitNumber(10);
            sp->setSplitSize(2);
            sp->setSplitStride(2);
            setSplit(sp);
        }
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
    this->calc = calc;
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
    headerFiles.push_back(new HeaderFile(headerFile));
}

dacppTranslator::HeaderFile* dacppTranslator::DacppFile::getHeaderFile(int idx) {
    return headerFiles[idx];
}

int dacppTranslator::DacppFile::getNumHeaderFile() {
    return headerFiles.size();
}

void dacppTranslator::DacppFile::setNameSpace(std::string nameSpace) {
    nameSpaces.push_back(new NameSpace(nameSpace));
}

dacppTranslator::NameSpace* dacppTranslator::DacppFile::getNameSpace(int idx) {
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