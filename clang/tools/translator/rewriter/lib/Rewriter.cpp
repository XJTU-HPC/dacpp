#include <set>

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

        // 计算输入和输出划分数量
        int countIn = 1;
        int countOut = 1;
        std::set<std::string> setIn;
        std::set<std::string> setOut;
        for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
            ShellParam* shellParam = shell->getShellParam(shellParamIdx);
            if(shellParam->getRw() == true) {
                for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                    Split* split = shellParam->getSplit(splitIdx);
                    if(setOut.count(split->getId()) == 1) {
                        continue;
                    }
                    setOut.insert(split->getId());
                    if(split->type == "dacpp::RegularSplit") {
                        RegularSplit* sp = static_cast<RegularSplit*>(split);
                        countOut *= sp->getSplitNumber();
                    }
                    else if(split->type == "dacpp::IndexSplit") {
                        IndexSplit* sp = static_cast<IndexSplit*>(split);
                        countOut *= sp->getSplitNumber();
                    }
                }
            } else {
                for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                    Split* split = shellParam->getSplit(splitIdx);
                    if(setIn.count(split->getId()) == 1) {
                        continue;
                    }
                    setIn.insert(split->getId());
                    if(split->type == "dacpp::RegularSplit") {
                        RegularSplit* sp = static_cast<RegularSplit*>(split);
                        countIn *= sp->getSplitNumber();
                    }
                    else if(split->type == "dacpp::IndexSplit") {
                        IndexSplit* sp = static_cast<IndexSplit*>(split);
                        countIn *= sp->getSplitNumber();
                    }
                }
            }
        }

        // 计算数据重排后需要的设备内存空间
        int* mem = new int[shell->getNumShellParams()];
        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam* shellParam = shell->getShellParam(j);
            mem[j] = 1;
            int dim = shellParam->getDim();
            for(int k = 0; k < dim; k++) {
                Split* split = shellParam->getSplit(k);
                // 降维划分
                if(split->type.compare("dacpp::IndexSplit") == 0) {
                    mem[j] *= shellParam->getShape(k);
                }
                // 规则分区划分
                else if(split->type.compare("dacpp::RegularSplit") == 0) {
                    dacppTranslator::RegularSplit* regulerSplit = static_cast<dacppTranslator::RegularSplit*>(split);
                    mem[j] *= (shellParam->getShape(k) - regulerSplit->getSplitSize() + regulerSplit->getSplitStride()) / regulerSplit->getSplitStride() * regulerSplit->getSplitSize();
                }
                // 保形划分
                else {
                    mem[j] *= shellParam->getShape(k);
                }
            }
        }

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

        std::string opInit = "";
        for(int count = 0; count < shell->getNumSplits(); count++) {
            Split* split = shell->getSplit(count);
            if(split->type.compare("RegularSplit") == 0) {
                RegularSplit* sp = static_cast<RegularSplit*>(split);
                opInit += CodeGen_RegularSliceInit(sp->getId(), std::to_string(sp->getSplitSize()), std::to_string(sp->getSplitStride()), std::to_string(sp->getSplitNumber()));
            }
            else if(split->type.compare("IndexSplit") == 0) {
                IndexSplit* sp = static_cast<IndexSplit*>(split);
                opInit += CodeGen_IndexInit(sp->getId(), std::to_string(sp->getSplitNumber()));
            }
        }

        std::string dataRecon = "";
        for(int count = 0; count < shell->getNumShellParams(); count++) {
            ShellParam* shellParam = shell->getShellParam(count);
            std::string opPushBack = "";
            int len = mem[count];
            for(int count = 0; count < shellParam->getNumSplit(); count++) {
                Split* split = shellParam->getSplit(count);
                if(split->type == "dacpp::RegularSplit") {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    opPushBack += CodeGen_OpPushBack(shellParam->getName(), sp->getId(), std::to_string(sp->getDimIdx()), std::to_string(len / shellParam->getShape(count) * sp->getSplitSize()));
                    len = len / shellParam->getShape(count) * sp->getSplitSize();
                }
                else if(split->type == "dacpp::IndexSplit") {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    opPushBack += CodeGen_OpPushBack(shellParam->getName(), sp->getId(), std::to_string(sp->getDimIdx()), std::to_string(len / shellParam->getShape(count)));
                    len /= shellParam->getShape(count);
                }
            }
            std::string dataOpsInit = CodeGen_DataOpsInit(shellParam->getName(), opPushBack);
            dataRecon += CodeGen_DataReconstruct(shellParam->getBasicType(), shellParam->getName(), std::to_string(mem[count]), dataOpsInit);
        }

        std::string deviceMemAlloc = "";
        for(int count = 0; count < shell->getNumShellParams(); count++) {
            ShellParam* shellParam = shell->getShellParam(count);
            if (shellParam->getRw() == 1) {
                deviceMemAlloc += CodeGen_DeviceMemAlloc(shellParam->getBasicType(), shellParam->getName(), std::to_string(mem[count] * countIn / countOut));
            } else {
                deviceMemAlloc += CodeGen_DeviceMemAlloc(shellParam->getBasicType(), shellParam->getName(), std::to_string(mem[count]));
            }
        }

        std::string H2DMemMove = "";
        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam* shellParam = shell->getShellParam(j);
            if (shellParam->getRw() == 1) {
                H2DMemMove += CodeGen_H2DMemMov(shellParam->getBasicType(), shellParam->getName(), std::to_string(mem[j] * countIn / countOut));
            } else {
                H2DMemMove += CodeGen_H2DMemMov(shellParam->getBasicType(), shellParam->getName(), std::to_string(mem[j]));
            }
        }

        // 索引初始化
        Dac_Ops ops;
        for(int count = 0; count < shell->getNumSplits(); count++) {
            if(shell->getSplit(count)->type.compare("IndexSplit") == 0) {
                dacppTranslator::IndexSplit* i = static_cast<dacppTranslator::IndexSplit*>(shell->getSplit(count));
                Index tmp = Index(i->getId());
                tmp.SetSplitSize(i->getSplitNumber());
                ops.push_back(tmp);
            } else if(shell->getSplit(count)->type.compare("RegularSplit") == 0) {
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
        std::string kernelExecute = CodeGen_KernelExecute(std::to_string(countIn), IndexInit, CalcEmbed);

        // 归约结果返回
        std::string reduction = "";
        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam* shellParam = shell->getShellParam(j);
            if(shellParam->getRw() == 1) {
                std::string ReductionRule = "";
                reduction += CodeGen_Reduction(std::to_string(countOut), shellParam->getName(), shellParam->getBasicType(), ReductionRule);
            }
        }
        // 归并结果返回
        std::string D2HMemMove = "";
        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam* shellParam = shell->getShellParam(j);
            if(shellParam->getRw() == 1) {
                D2HMemMove += CodeGen_D2HMemMov(shellParam->getName(), shellParam->getBasicType(), std::to_string(mem[j] * countIn / countOut), false);
            } else {
                D2HMemMove += CodeGen_D2HMemMov(shellParam->getName(), shellParam->getBasicType(), std::to_string(mem[j]), false);
            }
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