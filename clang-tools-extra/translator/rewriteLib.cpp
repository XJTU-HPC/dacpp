#include <string>
#include <vector>
#include <sstream>

#include "clang/AST/AST.h"

#include "rewriteLib.h"
#include "ASTLib.hpp"
#include "dacInfo.h"
#include "sub_template.h"

using namespace clang;

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

void dacpp::Source2Source::setCalcs(const BinaryOperator* dacExpr) {
    Calc calc;

    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacExpr->getLHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    FunctionDecl* shellFunc = shellCall->getDirectCallee();

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
        }
    }
    
    // 获取 DAC 数据关联表达式右值
    Expr* dacExprRHS = dacExpr->getRHS();
    FunctionDecl* calcFunc = dyn_cast<FunctionDecl>(dyn_cast<DeclRefExpr>(dacExprRHS)->getDecl());

    calc.setName(calcFunc->getNameAsString());

    for(unsigned int paramsCount = 0; paramsCount < calcFunc->getNumParams(); paramsCount++) {
        Param param;

        std::string type = calcFunc->getParamDecl(paramsCount)->getType().getAsString();
        param.setType(type);

        param.setBasicType(getBasicType(type));

        param.setName(calcFunc->getParamDecl(paramsCount)->getNameAsString());
        
        Expr* curExpr = ILE->getInit(paramsCount);
        ShellParam shellParam = getShell(shells_.size() - 1).getShellParam(paramsCount);
        param.setDim(shellParam.getDim());

        std::vector<InitListExpr*> initListExprs;
        getSlice(shellParam, curExpr, initListExprs);
        for(unsigned int i = 0; i < initListExprs.size(); i++) {
            if(getNode<DeclRefExpr>(initListExprs[i])) {
            }
            else {
                param.setShapes(shellParam.getShape(i));
            }
        }

        calc.setParams(param);
    }

    std::string body = stmt2String(calcFunc->getBody());
    for(int i = 0; i < calc.getNumParams(); i++) {
        if(body.find(calc.getParam(i).getName() + ".getShape") != -1) {
            int pos = body.find(calc.getParam(i).getName() + ".getShape");
            int dim = (int)body[pos + calc.getParam(i).getName().size() + 10] - (int)'0';
            body.replace(pos, calc.getParam(i).getName().size() + 12, std::to_string(calc.getParam(i).getShape(dim)));
        }
        if(body.find(calc.getParam(i).getName()) != -1 && calc.getParam(i).getNumShapes() == 0) {
            int pos = body.find(calc.getParam(i).getName());
            body.replace(pos, calc.getParam(i).getName().size(), calc.getParam(i).getName() + "[0]");
        }
    }
    calc.setBody(body);

    // 测试代码
    // for(int i = 0; i < calc.getNumParams(); i++) {
    //     llvm::outs() << calc.getParam(i).getType() << "\n";
    //     llvm::outs() << calc.getParam(i).getName() << "\n";
    //     llvm::outs() << calc.getParam(i).getBasicType() << "\n";
    //     llvm::outs() << calc.getParam(i).getDim() << "\n";
    //     for(int j = 0; j < calc.getParam(i).getNumShapes(); j++) {
    //         llvm::outs() << calc.getParam(i).getShape(j) << " ";
    //     }
    //     llvm::outs() << "\n";
    // }

    calcs_.push_back(calc);

    rewriter_->ReplaceText(calcFunc->getSourceRange(), "");
}

void dacpp::Source2Source::setShells(const BinaryOperator* dacExpr) {
    Shell shell;

    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacExpr->getLHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    FunctionDecl* shellFunc = shellCall->getDirectCallee();
    
    // 获取 DAC 数据关联表达式右值
    Expr* dacExprRHS = dacExpr->getRHS();
    FunctionDecl* calcFunc = dyn_cast<FunctionDecl>(dyn_cast<DeclRefExpr>(dacExprRHS)->getDecl());
    
    shell.setName(shellFunc->getNameAsString());
    for(unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams(); paramsCount++) {
        Param param;
        std::string type = shellFunc->getParamDecl(paramsCount)->getType().getAsString();
        param.setType(type);

        param.setBasicType(getBasicType(type));
        
        param.setName(shellFunc->getParamDecl(paramsCount)->getNameAsString());
        getParamInfo(shellCall->getArg(paramsCount), param);
        shell.setParams(param);
    }

    shell.setCalcName(calcFunc->getNameAsString());

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
                    shellParam.setType(shell.getParam(paramsCount).getType());
                    shellParam.setName(shell.getParam(paramsCount).getName());
                    shellParam.setBasicType(shell.getParam(paramsCount).getBasicType());
                    shellParam.setDim(shell.getParam(paramsCount).getDim());
                    for(int shapeIdx = 0; shapeIdx < shell.getParam(paramsCount).getNumShapes(); shapeIdx++) {
                        shellParam.setShapes(shell.getParam(paramsCount).getShape(shapeIdx));
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
                    for(int j = 0; j < shell.getNumOps(); j++) {
                        if(op.getName() != shell.getOp(j).getName()) { continue; }
                        shell.setOpSplitSize(j, shellParam.getShape(i));
                    }
                    shellParam.setParams(op);
                }
                shell.setShellParams(shellParam);
            }
        }
        else if(curVarDecl->getType().getAsString().compare("int") == 0) {
            DacOp op;                 
            op.setType(curVarDecl->getType().getAsString());
            op.setName(curVarDecl->getNameAsString());
            shell.setOps(op);
        }
    }
    
    // 测试代码
    // for(int i = 0; i < shell.getNumOps(); i++) {
    //     llvm::outs() << shell.getOp(i).getType() << "\n";
    //     llvm::outs() << shell.getOp(i).getName() << "\n";
    //     llvm::outs() << shell.getOp(i).getSplitSize() << "\n";
    // }
    // for(int i = 0; i < shell.getNumParams(); i++) {
    //     llvm::outs() << shell.getParam(i).getType() << "\n";
    //     llvm::outs() << shell.getParam(i).getName() << "\n";
    //     llvm::outs() << shell.getParam(i).getBasicType() << "\n";
    //     llvm::outs() << shell.getParam(i).getDim() << "\n";
    //     for(int j = 0; j < shell.getParam(i).getNumShape(); j++) {
    //         llvm::outs() << shell.getParam(i).getShape(j) << " ";
    //     }
    //     llvm::outs() << "\n";
    // }
    // for(int i = 0; i < shell.getNumShellParams(); i++) {
    //     llvm::outs() << shell.getShellParam(i).getRw() << "\n";
    //     llvm::outs() << shell.getShellParam(i).getType() << "\n";
    //     llvm::outs() << shell.getShellParam(i).getName() << "\n";
    //     llvm::outs() << shell.getShellParam(i).getBasicType() << "\n";
    //     llvm::outs() << shell.getShellParam(i).getDim() << "\n";
    //     for(int j = 0; j < shell.getShellParam(i).getNumShape(); j++) {
    //         llvm::outs() << shell.getShellParam(i).getShape(j) << " ";
    //     }
    //     llvm::outs() << "\n";
    //     for(int j = 0; j < shell.getShellParam(i).getNumOps(); j++) {
    //         llvm::outs() << shell.getShellParam(i).getOp(j).getType() << "\n";
    //         llvm::outs() << shell.getShellParam(i).getOp(j).getName() << "\n";
    //         llvm::outs() << shell.getShellParam(i).getOp(j).getSplitSize() << "\n";
    //     }
    //     llvm::outs() << "\n";
    // }

    shells_.push_back(shell);

    rewriter_->ReplaceText(shellFunc->getSourceRange(), "");
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
        Calc calc = getCalc(i);
        sstream << "void " << calc.getName() << "(";
        for(int j = 0; j < calc.getNumParams(); j++) {
            sstream << calc.getParam(j).getBasicType() << "* " << calc.getParam(j).getName();
            if(j != calc.getNumParams() - 1) sstream << ", ";
        }
        sstream << ")\n";
        sstream << calc.getBody() + "\n";
    }

    // 数据关联函数
    for(int i = 0; i < getNumShells(); i++) {
        Shell shell = getShell(i);
        sstream << "void " << shell.getName() << "(";
        for(int j = 0; j < shell.getNumParams(); j++) {
            sstream << shell.getParam(j).getType() << " " << shell.getParam(j).getName();
            dacShellParams += shell.getParam(j).getType() + " " + shell.getParam(j).getName();
            if(j != shell.getNumParams() - 1) { 
                sstream << ", ";
                dacShellParams += ", ";
            }
        }
        sstream << ") {\n";
        // 设备选择
        sstream << "    auto selector = gpu_selector_v;\n   queue q(selector);\n\n  DataReconstructor<int> tool;\n";
        
        // 数据重组
        for(int j = 0; j < shell.getNumShellParams(); j++) {
            ShellParam shellParam = shell.getShellParam(j);
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
        for(int j = 0; j < shell.getNumParams(); j++) {
            sstream << CodeGen_DeviceMemAlloc(shell.getParam(j).getBasicType(), shell.getParam(j).getName(), shell.getParam(j).getName() + ".getSize()");
            deviceMemAlloc += CodeGen_DeviceMemAlloc(shell.getParam(j).getBasicType(), shell.getParam(j).getName(), shell.getParam(j).getName() + ".getSize()");
        }
        sstream << "\n";
        deviceMemAlloc += "\n";

        // 数据移动
        for(int j = 0; j < shell.getNumParams(); j++) {
            sstream << CodeGen_H2DMemMov(shell.getParam(j).getBasicType(),shell.getParam(j).getName(), shell.getParam(j).getName() + ".getSize()");
            H2DMemMove += CodeGen_H2DMemMov(shell.getParam(j).getBasicType(),shell.getParam(j).getName(), shell.getParam(j).getName() + ".getSize()");
        }
        sstream << "\n";
        H2DMemMove += "\n";

        // 索引初始化
        Dac_Ops ops;
        for(int count = 0; count < shell.getNumOps(); count++) {
            Dac_Op op = Dac_Op(shell.getOp(count).getName(), shell.getOp(count).getSplitSize(), 0);
            ops.push_back(op);
        }
	    std::string IndexInit = CodeGen_IndexInit(ops);

        // 嵌入计算
        Args args = Args();
        for(int j = 0; j < shell.getNumShellParams(); j++) {
            ShellParam shellParam = shell.getShellParam(j);
            Dac_Ops ops;
            for(int k = 0; k < shellParam.getNumOps(); k++) {
                if(shellParam.getOp(k).getName() == "void") { continue; }
                Dac_Op op = Dac_Op(shell.getOp(k).getName(), shell.getOp(k).getSplitSize(), k);
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
        std::string CalcEmbed = CodeGen_CalcEmbed(shell.getCalcName(), args);

        // 内核执行
        sstream << CodeGen_KernelExecute("16", IndexInit, CalcEmbed);
        kernelExecute += CodeGen_KernelExecute("16", IndexInit, CalcEmbed);

        // 归并结果返回
        for(int j = 0; j < shell.getNumParams(); j++) {
            sstream << CodeGen_D2HMemMov(shell.getParam(j).getName(), shell.getParam(j).getBasicType(), shell.getParam(j).getName() + ".getSize()", false);
            D2HMemMove += CodeGen_D2HMemMov(shell.getParam(j).getName(), shell.getParam(j).getBasicType(), shell.getParam(j).getName() + ".getSize()", false);
        }
        sstream << "\n\n";
        D2HMemMove += "\n\n";

        for(int j = 0; j < shell.getNumShellParams(); j++) {
            if(shell.getShellParam(j).getRw() != 1) continue;
            sstream << "    " + shell.getShellParam(j).getName() + ".array2Tensor(r_" + shell.getShellParam(j).getName() + ");\n"; 
        }

        // 内存释放
        for(int j = 0; j < shell.getNumParams(); j++) {
            sstream << CodeGen_MemFree(shell.getParam(j).getName());
            memFree += CodeGen_MemFree(shell.getParam(j).getName());
        }
        sstream << "\n}\n";
        memFree += "\n\n";
    }

    std::string strResult = sstream.str() + "\n";
    std::string strRes = CodeGen_DAC2SYCL(getShell(0).getName(),dacShellParams,
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