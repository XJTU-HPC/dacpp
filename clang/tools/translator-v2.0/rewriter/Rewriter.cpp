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