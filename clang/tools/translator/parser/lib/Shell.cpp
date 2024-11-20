#include <string>

#include "llvm/ADT/StringExtras.h"

#include "Shell.h"
#include "ASTParse.h"


/**
 * 存储划分结构信息类实现
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

void dacppTranslator::Shell::setFather(Expression* expr) {
    this->father = expr;
}

dacppTranslator::Expression* dacppTranslator::Shell::getFather() {
    return father;
}

void dacppTranslator::Shell::setShellLoc(FunctionDecl* shellLoc) {
    this->shellLoc = shellLoc;
}

FunctionDecl* dacppTranslator::Shell::getShellLoc() {
    return shellLoc;
}

// 解析Shell节点，将解析到的信息存储到Shell类中
void dacppTranslator::Shell::parseShell(const BinaryOperator* dacExpr, std::vector<std::vector<int>> shapes) {
    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacExpr->getLHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    FunctionDecl* shellFunc = shellCall->getDirectCallee();
    
    // 设置 AST 中 Shell 节点的位置
    setShellLoc(shellFunc);

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
                            sp->type = "RegularSplit";
                            sp->setId(getNode<StringLiteral>(vd->getInit())->getString().str());
                            sp->setDimIdx(i);
                            CXXConstructExpr* CCE = getNode<CXXConstructExpr>(vd->getInit());
                            int count = 0;
                            for (CXXConstructExpr::arg_iterator I = CCE->arg_begin(),
                                                                E = CCE->arg_end();
                                I != E; ++I) {
                                if(count == 1) {
                                    /* TODO(mtc): 计算常量表达式的值。  */
                                    sp->setSplitSize(std::stoi(toString((dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
                                } else if(count == 2) {
                                    /* TODO(mtc): 计算常量表达式的值。  */
                                    sp->setSplitStride(std::stoi(toString((dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
                                }
                                count++;
                            }
                            /* TODO: 处理除不尽的情况。  */
                            sp->setSplitNumber((shellParam->getShape(i) - sp->getSplitSize()) / sp->getSplitStride() + 1);
                            for(int m = 0; m < getNumSplits(); m++) {
                                llvm::outs() << getSplit(m)->getId() << "  " << getSplit(m)->type;
                                if(getSplit(m)->getId().compare(sp->getId()) == 0 && getSplit(m)->type.compare("RegularSplit") == 0) {
                                    RegularSplit* isp = static_cast<RegularSplit*>(getSplit(m));
                                    isp->setSplitNumber(sp->getSplitNumber());
                                    llvm::outs() << isp->getSplitNumber();
                                }
                            }
                            shellParam->setSplit(sp);
                        } else if(vd->getType().getAsString().compare("dacpp::Index") == 0) {
                            IndexSplit* sp = new IndexSplit();
                            sp->type = "IndexSplit";
                            sp->setId(getNode<StringLiteral>(vd->getInit())->getString().str());
                            sp->setDimIdx(i);
                            sp->setSplitNumber(shellParam->getShape(i));
                            for(int m = 0; m < getNumSplits(); m++) {
                                llvm::outs() << getSplit(m)->getId() << "  " << getSplit(m)->type;
                                if(getSplit(m)->getId().compare(sp->getId()) == 0 && getSplit(m)->type.compare("IndexSplit") == 0) {
                                    IndexSplit* isp = static_cast<IndexSplit*>(getSplit(m));
                                    isp->setSplitNumber(sp->getSplitNumber());
                                    llvm::outs() << isp->getSplitNumber();
                                }
                            }
                            shellParam->setSplit(sp);
                        } 
                    } else {
                        Split* sp = new Split();
                        sp->type = "Split";
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
            sp->setId(curVarDecl->getNameAsString());
            sp->type = "IndexSplit";
            setSplit(sp);
        }
        else if(curVarDecl->getType().getAsString().compare("dacpp::RegularSplit") == 0) {
            RegularSplit* sp = new RegularSplit();
            sp->setId(curVarDecl->getNameAsString());
            CXXConstructExpr* CCE = getNode<CXXConstructExpr>(curVarDecl->getInit());
            int count = 0;
            for (CXXConstructExpr::arg_iterator I = CCE->arg_begin(),
                                                E = CCE->arg_end();
                I != E; ++I) {
                if(count == 1) {
                    /* TODO: 计算常量表达式的值。  */
                    sp->setSplitSize(std::stoi(toString((dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
                } else if(count == 2) {
                    /* TODO: 计算常量表达式的值。  */
                    sp->setSplitStride(std::stoi(toString((dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
                }
                count++;
            }
            sp->type = "RegularSplit";
            setSplit(sp);
        }
    }
}