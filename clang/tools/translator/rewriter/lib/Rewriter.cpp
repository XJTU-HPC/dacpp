#include <iostream>
#include <sstream>

#include "clang/Rewrite/Core/Rewriter.h"

#include "Rewriter.h"
#include "../../parser/include/Split.h"
#include "../../parser/include/Param.h"
#include "../include/dacInfo.h"
#include "../include/sub_template.h"


void dacppTranslator::Rewriter::setRewriter(clang::Rewriter* rewriter) {
    this->rewriter = rewriter;
}

void dacppTranslator::Rewriter::setDacppFile(DacppFile* dacppFile) {
    this->dacppFile = dacppFile;
}

void dacppTranslator::Rewriter::rewriteDac() {

    std::string code = "";

    // 添加头文件
    for(int i = 0; i < dacppFile->getNumHeaderFile(); i++) {
        code += "#include " + dacppFile->getHeaderFile(i)->getName() + "\n";
    }
    code += "\n";
    
    // 添加命名空间
    for(int i = 0; i < dacppFile->getNumNameSpace(); i++) {
        code += "using namespace " + dacppFile->getNameSpace(i)->getName() + ";\n";
    }
    code += "\n";
    
    // 添加数据关联表达式对应的划分结构和计算结构
    for(int i = 0; i < dacppFile->getNumExpression(); i++) {
        Expression* expr = dacppFile->getExpression(i);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        
        // 计算结构
        code += "void " + calc->getName() + "(";
        for(int count = 0; count < calc->getNumParams(); count++) {
            code += calc->getParam(count)->getBasicType() + "* " + calc->getParam(count)->getName();
            if(count != calc->getNumParams() - 1) {
                code += ", ";
            }
        }
        code += ") {\n";
        for(int count = 0; count < calc->getNumBody(); count++) {
            code += "    " + calc->getBody(i) + "\n";
        }
        code += "}\n";


        std::string dacShellName = shell->getName();
        
        std::string dacShellParams = "";
        for (int count = 0; count < shell->getNumParams(); count++) {
            Param* param = shell->getParam(count);
            dacShellParams += param->getType() + " " + param->getName();
            if(count != shell->getNumParams() - 1) { 
                dacShellParams += ", ";
            }
        }

        std::string opInit;
        for(int count = 0; count < shell->getNumSplits(); count++) {
            Split* split = shell->getSplit(count);
            if(split->type == "dacpp::RegularSplit") {
                RegularSplit* sp = static_cast<RegularSplit*>(split);
                opInit += CodeGen_RegularSliceInit(sp->getId(), std::to_string(sp->getSplitSize()), std::to_string(sp->getSplitStride()), std::to_string(sp->getSplitNumber()));
            }
            else if(split->type == "dacpp::IndexSplit") {
                IndexSplit* sp = static_cast<IndexSplit*>(split);
                opInit += CodeGen_IndexInit(sp->getId(), std::to_string(sp->getSplitNumber()));
            }
        }

        std::string dataRecon;
        for(int count = 0; count < shell->getNumShellParams(); count++) {
            ShellParam* shellParam = shell->getShellParam(count);
            std::string opPushBack = "";
            for(int count = 0; count < shellParam->getNumSplit(); count++) {
                Split* split = shellParam->getSplit(count);
                if(split->type == "dacpp::RegularSplit") {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    opPushBack += CodeGen_OpPushBack(shell->getName(), sp->getId(), std::to_string(sp->getDimIdx()), std::to_string(sp->getSplitNumber()));
                }
                else if(split->type == "dacpp::IndexSplit") {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    opPushBack += CodeGen_OpPushBack(shell->getName(), sp->getId(), std::to_string(sp->getDimIdx()), std::to_string(sp->getSplitNumber()));
                }
            }
            std::string dataOpsInit = CodeGen_DataOpsInit(shell->getName(), opPushBack);
            dataRecon += CodeGen_DataReconstruct(shellParam->getBasicType(), shellParam->getName(), std::to_string(100), dataOpsInit);
        }

        std::string deviceMemAlloc = "";
        for(int count = 0; count < shell->getNumShellParams(); count++) {
            ShellParam* shellParam = shell->getShellParam(count);
            deviceMemAlloc += CodeGen_DeviceMemAlloc(shellParam->getBasicType(), shellParam->getName(), std::to_string(100));
        }

        std::string H2DMemMove = "";
        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam* shellParam = shell->getShellParam(j);
            H2DMemMove += CodeGen_H2DMemMov(shellParam->getBasicType(), shellParam->getName(), std::to_string(100));
        }

        // 索引初始化
        Dac_Ops ops;
        for(int count = 0; count < shell->getNumSplits(); count++) {
            if(shell->getSplit(count)->type == "dacpp::Index") {
                dacppTranslator::IndexSplit* i = static_cast<dacppTranslator::IndexSplit*>(shell->getSplit(count));
                Index tmp = Index(i->getId());
                tmp.SetSplitSize(i->getSplitNumber());
                ops.push_back(tmp);
            } else if(shell->getSplit(count)->type == "dacpp::Index") {
                dacppTranslator::RegularSplit* r = static_cast<dacppTranslator::RegularSplit*>(shell->getSplit(count));
                RegularSlice tmp = RegularSlice(r->getId(), r->getSplitSize(), r->getSplitStride());
                tmp.SetSplitSize(r->getSplitNumber());
                ops.push_back(tmp);
            }
        }
	    std::string IndexInit = CodeGen_IndexInit(ops);

        // 嵌入计算
        Args args = Args();
        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam* shellParam = shell->getShellParam(j);
            Dac_Ops ops;
            for(int k = 0; k < shellParam->getNumSplit(); k++) {
                if(shellParam->getSplit(k)->getId() == "void") { continue; }
                Dac_Op op = Dac_Op(shell->getSplit(k)->getId(), shell->getSplit(k)->getDimIdx(), k);
                int length = 1;
                for(int idx = 0; idx < shellParam->getDim(); idx++) {
                    if(idx == k) continue;
                    length *= shellParam->getShape(idx);
                }
                op.setSplitLength(length);
                ops.push_back(op);
            }
            
            DacData temp = DacData("d_" + shellParam->getName(), shellParam->getDim(), ops);
            
            for(int dimIdx = 0; dimIdx < shellParam->getDim(); dimIdx++) {
                temp.setDimLength(dimIdx, shellParam->getShape(dimIdx));
            }
            
            args.push_back(temp);
        }
        
        std::string CalcEmbed = CodeGen_CalcEmbed(calc->getName(), args);

        // 内核执行
        std::string kernelExecute = CodeGen_KernelExecute("16", IndexInit, CalcEmbed);

        // 归约结果返回
        std::string reduction = "";

        // 归并结果返回
        std::string D2HMemMove = "";
        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam* shellParam = shell->getShellParam(j);
            D2HMemMove += CodeGen_D2HMemMov(shellParam->getName(), shellParam->getBasicType(), std::to_string(100), false);
        }

        // 内存释放
        std::string memFree = "";
        for(int j = 0; j < shell->getNumParams(); j++) {
            ShellParam* shellParam = shell->getShellParam(j);
            memFree += CodeGen_MemFree(shellParam->getName());
        }
        
        code += CodeGen_DAC2SYCL(dacShellName, dacShellParams, opInit, dataRecon, deviceMemAlloc, H2DMemMove, kernelExecute, reduction, D2HMemMove, memFree);
        code += "\n\n";
        rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
        rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
    }

    rewriter->InsertText(dacppFile->getMainFuncLoc()->getBeginLoc(), code);

}

/*
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
*/

void dacppTranslator::Rewriter::rewriteMain() {

}