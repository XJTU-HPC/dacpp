#include <iostream>
#include <sstream>

#include "clang/Rewrite/Core/Rewriter.h"

#include "Rewriter.h"
#include "/data/zjx/dacpp/clang/tools/translator/dac2sycl/dacppLib/include/dacInfo.h"
#include "/data/zjx/dacpp/clang/tools/translator/dac2sycl/dacppLib/include/sub_template.h"

void dacppTranslator::Rewriter::setRewriter(clang::Rewriter* rewriter) {
    this->rewriter = rewriter;
}

void dacppTranslator::Rewriter::setDacppFile(DacppFile* dacppFile) {
    this->dacppFile = dacppFile;
}

void dacppTranslator::Rewriter::rewriteDac() {
    std::stringstream sstream;

    std::string dacShellParams = "";
    std::string dataRecon = "";
    std::string deviceMemAlloc = "";
    std::string H2DMemMove = "";
    std::string kernelExecute = "";
    std::string reduction = "";
    std::string D2HMemMove = "";
    std::string memFree = "";

    // 添加头文件
    for(int i = 0; i < dacppFile->getNumHeaderFile(); i++) {
        sstream << "#include " << dacppFile->getHeaderFile(i)->getName() << "\n";
    }
    sstream << "\n";
    
    // 添加命名空间
    for(int i = 0; i < dacppFile->getNumNameSpace(); i++) {
        sstream << "using namespace " << dacppFile->getNameSpace(i)->getName() << ";\n";
    }
    sstream << "\n";
    
    // 添加数据关联表达式对应的划分结构和计算结构
    for(int i = 0; i < dacppFile->getNumExpression(); i++) {
        Expression* expr = dacppFile->getExpression(i);

        // 计算结构
        Calc* calc = expr->getCalc();
        sstream << "void " << calc->getName() << "(";
        for(int j = 0; j < calc->getNumParams(); j++) {
            sstream << calc->getParam(j)->getBasicType() << "* " << calc->getParam(j)->getName();
            if(j != calc->getNumParams() - 1) sstream << ", ";
        }
        sstream << ") {\n";
        for(int i = 0; i < calc->getNumBody(); i++) {
            sstream << calc->getBody(i) << "\n";
        }
        sstream << "}\n";
        
        // 划分结构
        Shell* shell = expr->getShell();
        sstream << "void " << shell->getName() << "(";
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << shell->getParam(j)->getType() << " " << shell->getParam(j)->getName();
            dacShellParams += shell->getParam(j)->getType() + " " + shell->getParam(j)->getName();
            if(j != shell->getNumParams() - 1) { 
                sstream << ", ";
                dacShellParams += ", ";
            }
        }
        sstream << ") {\n";
        
        // 创建任务队列并绑定设备
        sstream << "auto selector = gpu_selector_v;\nqueue q(selector);\n\n";
        
        // 数据重组
        sstream << "DataReconstructor<int> tool;\n";

        for(int j = 0; j < shell->getNumShellParams(); j++) {
            ShellParam* shellParam = shell->getShellParam(j);
            std::string indexInfo = "{"; 
            for(int k = 0; k < shellParam->getNumSplit(); k++) {
                if(shellParam->getSplit(k)->getId() == "void") { indexInfo += "false"; }
                else { indexInfo += "true"; }
                if(k != shellParam->getNumSplit() - 1) { indexInfo += ", "; }
                else { indexInfo += "}";}
            }
            sstream << CodeGen_DataReconstruct(shellParam->getBasicType(), shellParam->getName(), shellParam->getName() + ".getSize()", indexInfo);
            dataRecon += CodeGen_DataReconstruct(shellParam->getBasicType(), shellParam->getName(), shellParam->getName() + ".getSize()", indexInfo);
        }

        // 设备内存分配
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << CodeGen_DeviceMemAlloc(shell->getParam(j)->getBasicType(), shell->getParam(j)->getName(), shell->getParam(j)->getName() + ".getSize()");
            deviceMemAlloc += CodeGen_DeviceMemAlloc(shell->getParam(j)->getBasicType(), shell->getParam(j)->getName(), shell->getParam(j)->getName() + ".getSize()");
        }
        sstream << "\n";
        deviceMemAlloc += "\n";

        // 数据移动
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << CodeGen_H2DMemMov(shell->getParam(j)->getBasicType(),shell->getParam(j)->getName(), shell->getParam(j)->getName() + ".getSize()");
            H2DMemMove += CodeGen_H2DMemMov(shell->getParam(j)->getBasicType(),shell->getParam(j)->getName(), shell->getParam(j)->getName() + ".getSize()");
        }
        sstream << "\n";
        H2DMemMove += "\n";

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
        
        std::string CalcEmbed = CodeGen_CalcEmbed(shell->getExpr()->getCalc()->getName(), args);

        // 内核执行
        sstream << CodeGen_KernelExecute("16", IndexInit, CalcEmbed);
        kernelExecute += CodeGen_KernelExecute("16", IndexInit, CalcEmbed);

        // 归并结果返回
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << CodeGen_D2HMemMov(shell->getParam(j)->getName(), shell->getParam(j)->getBasicType(), shell->getParam(j)->getName() + ".getSize()", false);
            D2HMemMove += CodeGen_D2HMemMov(shell->getParam(j)->getName(), shell->getParam(j)->getBasicType(), shell->getParam(j)->getName() + ".getSize()", false);
        }
        sstream << "\n\n";
        D2HMemMove += "\n\n";

        for(int j = 0; j < shell->getNumShellParams(); j++) {
            if(shell->getShellParam(j)->getRw() != 1) continue;
            sstream << "    " + shell->getShellParam(j)->getName() + ".array2Tensor(r_" + shell->getShellParam(j)->getName() + ");\n"; 
        }

        // 内存释放
        for(int j = 0; j < shell->getNumParams(); j++) {
            sstream << CodeGen_MemFree(shell->getParam(j)->getName());
            memFree += CodeGen_MemFree(shell->getParam(j)->getName());
        }
        sstream << "\n}\n";
        memFree += "\n\n";
        
    }
    
    std::string strResult = sstream.str() + "\n";
    //std::string strRes = CodeGen_DAC2SYCL(shell->getName(), dacShellParams,
    //dataRecon, deviceMemAlloc, H2DMemMove, kernelExecute, reduction, D2HMemMove, memFree);
    
    rewriter->InsertText(dacppFile->getMainFuncLoc()->getBeginLoc(), strResult);
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