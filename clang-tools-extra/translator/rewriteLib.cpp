#include <string>
#include <vector>
#include <sstream>

#include "clang/AST/AST.h"

#include "rewriteLib.h"
#include "ASTLib.h"
#include "ASTLib2.hpp"
#include "dacInfo.h"
#include "sub_template.h"
#include "test/test.h"

using namespace clang;


/*
    dacpp::Calc 成员函数定义
*/
void dacpp::Calc::setName(std::string name) { name_ = name; }

std::string dacpp::Calc::getName() const { return name_; }

void dacpp::Calc::setParams(Param param) { params_.push_back(param); }

int dacpp::Calc::getNumParams() const { return params_.size(); }

dacpp::Param dacpp::Calc::getParam(int idx) const { return params_[idx]; }

void dacpp::Calc::setBody(clang::Stmt* body) {
    for(Stmt::child_iterator it = body->child_begin(); it != body->child_end(); it++) {
        if(isa<ExprWithCleanups>(*it) && getNode<BinaryOperator>(*it)) {
            body_.push("Expression");
            setExpression(getNode<BinaryOperator>(*it));
            continue;
        }
        std::string temp = stmt2String(*it);
        for(int i = 0; i < getNumParams(); i++) {
            if(temp.find(getParam(i).getName() + ".getShape") != -1) {
                int pos = temp.find(getParam(i).getName() + ".getShape");
                int dim = (int)temp[pos + getParam(i).getName().size() + 10] - (int)'0';
                temp.replace(pos, getParam(i).getName().size() + 12, std::to_string(getParam(i).getShape(dim)));
            }
            if(temp.find(getParam(i).getName()) != -1 && getParam(i).getNumShapes() == 0) {
                int pos = temp.find(getParam(i).getName());
                temp.replace(pos, getParam(i).getName().size(), getParam(i).getName() + "[0]");
            }
        }
        body_.push(temp);
    }
}

std::string dacpp::Calc::getBody() {
    if(body_.empty()) {
        return "None";
    }
    std::string temp = body_.front();
    body_.pop();
    return temp;
}

void dacpp::Calc::setExpression(const BinaryOperator* dacExpr) {
    Expression* curExpr = new Expression();
    Calc* calc = new Calc();
    std::vector<std::vector<int>> shapes(params_.size());
    for(int i = 0; i < params_.size(); i++) {
        for(int j = 0; j < params_[i].getDim(); j++) {
            shapes[i].push_back(params_[i].getShape(j));
        }
    }
    Source2Source::setCalc(dacExpr, calc, shapes);
    //calcTest(calc);
    curExpr->setCalc(calc);
    Shell* shell = new Shell();
    Source2Source::setShell(dacExpr, shell, shapes);
    //shellTest(shell);
    curExpr->setShell(shell);
    exprs_.push(curExpr);
}

dacpp::Expression* dacpp::Calc::getExpression() {
    dacpp::Expression* temp = exprs_.front();
    exprs_.pop();
    return temp;
}


/*
    dacpp::Source2Source 成员函数定义
*/
void dacpp::Source2Source::setHeaderFiles(std::string name) {
    HeaderFile headerfile;
    headerfile.setName(name);
    headerFiles_.push_back(headerfile);
}

void dacpp::Source2Source::setNameSpaces(std::string name) {
    NameSpace nameSpace;
    nameSpace.setName(name);
    nameSpaces_.push_back(nameSpace);
}

void dacpp::Source2Source::setCalc(const clang::BinaryOperator* dacExpr, Calc* calc, std::vector<std::vector<int>> shapes) {

    Shell* shell = new Shell();
    
    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacExpr->getLHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    FunctionDecl* shellFunc = shellCall->getDirectCallee();
    
    shell->setName(shellFunc->getNameAsString());
    for(unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {
        Param param;
        std::string type = shellFunc->getParamDecl(paramsCount)->getType().getAsString();
        param.setType(type);

        param.setBasicType(getBasicType(type));
        
        param.setName(shellFunc->getParamDecl(paramsCount)->getNameAsString());
        
        for(int i = 0; i < shapes[paramsCount].size(); i++) {
            param.setShapes(shapes[paramsCount][i]);
        }
        param.setDim(shapes[paramsCount].size());

        shell->setParams(param);
    }

    InitListExpr* ILE;
    // 获取关联结构信息
    Stmt* shellFuncBody = shellFunc->getBody();
    for(Stmt::child_iterator it = shellFuncBody->child_begin(); it != shellFuncBody->child_end(); it++) {
        if(!isa<DeclStmt>(*it)) { continue; }
        Decl* curDecl = dyn_cast<DeclStmt>(*it)->getSingleDecl();
        if(!isa<VarDecl>(curDecl)) { continue; }
        VarDecl* curVarDecl = dyn_cast<VarDecl>(curDecl);
        if(curVarDecl->getType().getAsString().compare("dacpp::list") == 0) {
            ILE = getNode<InitListExpr>(*it);
            for(unsigned int i = 0; i < ILE->getNumInits(); i++) {
                ShellParam shellParam;
                Expr* curExpr = ILE->getInit(i);
                std::vector<InitListExpr*> initListExprs;
                getSlice(shellParam, curExpr, initListExprs);
                for(unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {            
                    if(shellParam.getName() != shellFunc->getParamDecl(paramsCount)->getNameAsString()) { continue; }
                    shellParam.setRw(inputOrOutput(shellFunc->getParamDecl(paramsCount)->getType().getAsString()));
                    shellParam.setType(shell->getParam(paramsCount).getType());
                    shellParam.setName(shell->getParam(paramsCount).getName());
                    shellParam.setBasicType(shell->getParam(paramsCount).getBasicType());
                    shellParam.setDim(shell->getParam(paramsCount).getDim());
                    for(int shapeIdx = 0; shapeIdx < shell->getParam(paramsCount).getNumShapes(); shapeIdx++) {
                        shellParam.setShapes(shell->getParam(paramsCount).getShape(shapeIdx));
                    }
                }
                for(unsigned int i = 0; i < initListExprs.size(); i++) {
                    DacOp op;
                    if(getNode<DeclRefExpr>(initListExprs[i])) {
                        VarDecl* vd = dyn_cast<VarDecl>(getNode<DeclRefExpr>(initListExprs[i])->getDecl());                    
                        op.setType(vd->getType().getAsString());
                        op.setName(vd->getNameAsString());
                        op.setSplitSize(shellParam.getShape(i));
                    }
                    else {
                        op.setType("void");
                        op.setName("void");
                        op.setSplitSize(1);
                    }
                    for(int j = 0; j < shell->getNumOps(); j++) {
                        if(op.getName() != shell->getOp(j).getName()) { continue; }
                        shell->setOpSplitSize(j, shellParam.getShape(i));
                    }
                    shellParam.setParams(op);
                }
                shell->setShellParams(shellParam);
            }
        }
        else if(curVarDecl->getType().getAsString().compare("int") == 0) {
            DacOp op;                 
            op.setType(curVarDecl->getType().getAsString());
            op.setName(curVarDecl->getNameAsString());
            shell->setOps(op);
        }
    }

    // 获取 DAC 数据关联表达式右值
    Expr* dacExprRHS = dacExpr->getRHS();
    FunctionDecl* calcFunc = dyn_cast<FunctionDecl>(dyn_cast<DeclRefExpr>(dacExprRHS)->getDecl());
    calc->setName(calcFunc->getNameAsString());

    for(unsigned int paramsCount = 0; paramsCount < calcFunc->getNumParams(); paramsCount++) {
        Param param;

        std::string type = calcFunc->getParamDecl(paramsCount)->getType().getAsString();
        param.setType(type);

        param.setBasicType(getBasicType(type));

        param.setName(calcFunc->getParamDecl(paramsCount)->getNameAsString());
        
        Expr* curExpr = ILE->getInit(paramsCount);
        ShellParam shellParam = shell->getShellParam(paramsCount);

        std::vector<InitListExpr*> initListExprs;
        getSlice(shellParam, curExpr, initListExprs);
        for(unsigned int i = 0; i < initListExprs.size(); i++) {
            if(getNode<DeclRefExpr>(initListExprs[i])) {
            }
            else {
                param.setShapes(shellParam.getShape(i));
            }
        }
        param.setDim(param.getNumShapes());

        calc->setParams(param);
    }

    calc->setBody(calcFunc->getBody());

    // rewriter_->ReplaceText(calcFunc->getSourceRange(), "");
}

void dacpp::Source2Source::setCalcs(dacpp::Calc* calc) {
    calcs_.push_back(calc);
}

void dacpp::Source2Source::setShell(const clang::BinaryOperator* dacExpr, dacpp::Shell* shell, std::vector<std::vector<int>> shapes) {

    // 获取 DAC 数据关联表达式右值
    Expr* dacExprRHS = dacExpr->getRHS();
    FunctionDecl* calcFunc = dyn_cast<FunctionDecl>(dyn_cast<DeclRefExpr>(dacExprRHS)->getDecl());
    shell->setCalcName(calcFunc->getNameAsString());
    
    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacExpr->getLHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    FunctionDecl* shellFunc = shellCall->getDirectCallee();
    
    shell->setName(shellFunc->getNameAsString());
    for(unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {
        Param param;
        std::string type = shellFunc->getParamDecl(paramsCount)->getType().getAsString();
        param.setType(type);

        param.setBasicType(getBasicType(type));
        
        param.setName(shellFunc->getParamDecl(paramsCount)->getNameAsString());
        
        for(int i = 0; i < shapes[paramsCount].size(); i++) {
            param.setShapes(shapes[paramsCount][i]);
        }
        param.setDim(shapes[paramsCount].size());

        shell->setParams(param);
    }

    // 获取关联结构信息
    Stmt* shellFuncBody = shellFunc->getBody();
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
                std::vector<InitListExpr*> initListExprs;
                getSlice(shellParam, curExpr, initListExprs);
                for(unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {            
                    if(shellParam.getName() != shellFunc->getParamDecl(paramsCount)->getNameAsString()) { continue; }
                    shellParam.setRw(inputOrOutput(shellFunc->getParamDecl(paramsCount)->getType().getAsString()));
                    shellParam.setType(shell->getParam(paramsCount).getType());
                    shellParam.setName(shell->getParam(paramsCount).getName());
                    shellParam.setBasicType(shell->getParam(paramsCount).getBasicType());
                    shellParam.setDim(shell->getParam(paramsCount).getDim());
                    for(int shapeIdx = 0; shapeIdx < shell->getParam(paramsCount).getNumShapes(); shapeIdx++) {
                        shellParam.setShapes(shell->getParam(paramsCount).getShape(shapeIdx));
                    }
                }
                for(unsigned int i = 0; i < initListExprs.size(); i++) {
                    DacOp op;
                    if(getNode<DeclRefExpr>(initListExprs[i])) {
                        VarDecl* vd = dyn_cast<VarDecl>(getNode<DeclRefExpr>(initListExprs[i])->getDecl());                    
                        op.setType(vd->getType().getAsString());
                        op.setName(vd->getNameAsString());
                        op.setSplitSize(shellParam.getShape(i));
                    }
                    else {
                        op.setType("void");
                        op.setName("void");
                        op.setSplitSize(1);
                    }
                    for(int j = 0; j < shell->getNumOps(); j++) {
                        if(op.getName() != shell->getOp(j).getName()) { continue; }
                        shell->setOpSplitSize(j, shellParam.getShape(i));
                    }
                    shellParam.setParams(op);
                }
                shell->setShellParams(shellParam);
            }
        }
        else if(curVarDecl->getType().getAsString().compare("int") == 0) {
            DacOp op;                 
            op.setType(curVarDecl->getType().getAsString());
            op.setName(curVarDecl->getNameAsString());
            shell->setOps(op);
        }
    }

    // rewriter_->ReplaceText(shellFunc->getSourceRange(), "");
}

void dacpp::Source2Source::setShells(dacpp::Shell* shell) {
    shells_.push_back(shell);
}

void recursiveFunc(dacpp::Calc* calc, std::stringstream& sstream) {
    while(1) {
        std::string temp = calc->getBody();
        if(temp == "None") break;
        if(temp == "Expression") {
            recursiveFunc(calc->getExpression()->getCalc(), sstream);
            continue;
        }
        sstream << "    " << temp << "\n";
    }
}

void dacpp::Source2Source::rewriteDac() {
    std::stringstream sstream;
    std::string dacShellParams = "";
    std::string dataRecon = "";
    std::string deviceMemAlloc = "";
    std::string H2DMemMove = "";
    std::string kernelExecute = "";
    std::string reduction = "";
    std::string D2HMemMove = "";
    std::string memFree = "";

    // 头文件
    setHeaderFiles("<sycl/sycl.hpp>");
    setHeaderFiles("\"../DataReconstructor.cpp\"");
    for(int i = 0; i < getNumHeaderFiles(); i++) {
        sstream << getHeaderFile(i).getStr() << "\n";
    }
    sstream << "\n";

    // 命名空间
    setNameSpaces("dacpp");
    setNameSpaces("sycl");
    for(int i = 0; i < getNumNameSpaces(); i++) {
        sstream << getNameSpace(i).getStr() << "\n";
    }
    sstream << "\n";

    // 计算函数
    for(int i = 0; i < getNumCalcs(); i++) {
        Calc* calc = getCalc(i);
        sstream << "void " << calc->getName() << "(";
        for(int j = 0; j < calc->getNumParams(); j++) {
            sstream << calc->getParam(j).getBasicType() << "* " << calc->getParam(j).getName();
            if(j != calc->getNumParams() - 1) sstream << ", ";
        }
        sstream << ") {\n";
        recursiveFunc(calc, sstream);
        sstream << "}\n";
    }

    // 数据关联函数
    for(int i = 0; i < getNumShells(); i++) {
        Shell* shell = getShell(i);
        sstream << "void " << shell->getName() << "(";
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << shell->getParam(j).getType() << " " << shell->getParam(j).getName();
            dacShellParams += shell->getParam(j).getType() + " " + shell->getParam(j).getName();
            if(j != shell->getNumParams() - 1) { 
                sstream << ", ";
                dacShellParams += ", ";
            }
        }
        sstream << ") {\n";
        // 设备选择
        sstream << "    auto selector = gpu_selector_v;\n   queue q(selector);\n\n  DataReconstructor<int> tool;\n";
        
        // 数据重组
        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam shellParam = shell->getShellParam(j);
            std::string indexInfo = "{"; 
            for(int k = 0; k < shellParam.getNumOps(); k++) {
                if(shellParam.getOp(k).getName() == "void") { indexInfo += "false"; }
                else { indexInfo += "true"; }
                if(k != shellParam.getNumOps() - 1) { indexInfo += ", "; }
                else { indexInfo += "}";}
            }
            sstream << CodeGen_DataReconstruct(shellParam.getBasicType(), shellParam.getName(), shellParam.getName() + ".getSize()", indexInfo);
            dataRecon += CodeGen_DataReconstruct(shellParam.getBasicType(), shellParam.getName(), shellParam.getName() + ".getSize()", indexInfo);
        }

        // 设备内存分配
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << CodeGen_DeviceMemAlloc(shell->getParam(j).getBasicType(), shell->getParam(j).getName(), shell->getParam(j).getName() + ".getSize()");
            deviceMemAlloc += CodeGen_DeviceMemAlloc(shell->getParam(j).getBasicType(), shell->getParam(j).getName(), shell->getParam(j).getName() + ".getSize()");
        }
        sstream << "\n";
        deviceMemAlloc += "\n";

        // 数据移动
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << CodeGen_H2DMemMov(shell->getParam(j).getBasicType(),shell->getParam(j).getName(), shell->getParam(j).getName() + ".getSize()");
            H2DMemMove += CodeGen_H2DMemMov(shell->getParam(j).getBasicType(),shell->getParam(j).getName(), shell->getParam(j).getName() + ".getSize()");
        }
        sstream << "\n";
        H2DMemMove += "\n";

        // 索引初始化
        Dac_Ops ops;
        for(int count = 0; count < shell->getNumOps(); count++) {
            Dac_Op op = Dac_Op(shell->getOp(count).getName(), shell->getOp(count).getSplitSize(), 0);
            ops.push_back(op);
        }
	    std::string IndexInit = CodeGen_IndexInit(ops);

        // 嵌入计算
        Args args = Args();
        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam shellParam = shell->getShellParam(j);
            Dac_Ops ops;
            for(int k = 0; k < shellParam.getNumOps(); k++) {
                if(shellParam.getOp(k).getName() == "void") { continue; }
                Dac_Op op = Dac_Op(shell->getOp(k).getName(), shell->getOp(k).getSplitSize(), k);
                int length = 1;
                for(int idx = 0; idx < shellParam.getNumShapes(); idx++) {
                    if(idx == k) continue;
                    length *= shellParam.getShape(idx);
                }
                op.setSplitLength(length);
                ops.push_back(op);
            }
            DacData temp = DacData("d_" + shellParam.getName(), shellParam.getDim(), ops);
            for(int dimIdx = 0; dimIdx < shellParam.getDim(); dimIdx++) {
                temp.setDimLength(dimIdx, shellParam.getShape(dimIdx));
            }
            args.push_back(temp);
        }
        std::string CalcEmbed = CodeGen_CalcEmbed(shell->getCalcName(), args);

        // 内核执行
        sstream << CodeGen_KernelExecute("16", IndexInit, CalcEmbed);
        kernelExecute += CodeGen_KernelExecute("16", IndexInit, CalcEmbed);

        // 归并结果返回
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << CodeGen_D2HMemMov(shell->getParam(j).getName(), shell->getParam(j).getBasicType(), shell->getParam(j).getName() + ".getSize()", false);
            D2HMemMove += CodeGen_D2HMemMov(shell->getParam(j).getName(), shell->getParam(j).getBasicType(), shell->getParam(j).getName() + ".getSize()", false);
        }
        sstream << "\n\n";
        D2HMemMove += "\n\n";

        for(int j = 0; j < shell->getNumShellParams(); j++) {
            if(shell->getShellParam(j).getRw() != 1) continue;
            sstream << "    " + shell->getShellParam(j).getName() + ".array2Tensor(r_" + shell->getShellParam(j).getName() + ");\n"; 
        }

        // 内存释放
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << CodeGen_MemFree(shell->getParam(j).getName());
            memFree += CodeGen_MemFree(shell->getParam(j).getName());
        }
        sstream << "\n}\n";
        memFree += "\n\n";
    }

    std::string strResult = sstream.str() + "\n";
    std::string strRes = CodeGen_DAC2SYCL(getShell(0)->getName(),dacShellParams,
    dataRecon, deviceMemAlloc, H2DMemMove, kernelExecute, reduction, D2HMemMove, memFree);
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