#include "clang/Rewrite/Core/Rewriter.h"
#include "Rewriter.h"
#include "Split.h"
#include "Param.h"
#include "dacInfo.h"
#include "sub_template.h"
#include "test.h"

#include <set>
#include <iostream>
#include <sstream>
#include <vector>

void dacppTranslator::Rewriter::setRewriter(clang::Rewriter* rewriter) {
    this->rewriter = rewriter;
}

void dacppTranslator::Rewriter::setDacppFile(DacppFile* dacppFile) {
    this->dacppFile = dacppFile;
}

std::string dacppTranslator::Rewriter::generateCalc(Calc* calc) {
    std::string code = "";
    for (int blockIdx = 0, idx = 0, exprIdx = 0; blockIdx < calc->blocks.size(); blockIdx++) {
        if (calc->blocks[blockIdx].compare("@Expression;") == 0) {
            code += generateCalc(calc->getExpr(exprIdx++)->getCalc());
            continue;
        }
        code += "void " + calc->getName() + "_" + std::to_string(idx++) + "(";
        for(int count = 0; count < calc->getNumParams(); count++) {
            code += calc->getParam(count)->getBasicType() + "* " + calc->getParam(count)->getName();
            if(count != calc->getNumParams() - 1) {
                code += ", ";
            }
        }
        code += ") {\n";
        code += calc->blocks[blockIdx] + "\n}\n";
    }
    return code;
}

/**
 * 获得数据关联计算表达式的所有算子（包含嵌套）
 * @param shapes 最顶层数据关联计算表达式dacpplist参数形状
 * @param splits 存放算子
 * @param expr 数据关联表达式
 */
void dacppTranslator::Rewriter::addSplit(std::vector<std::vector<int>>& shapes, std::vector<std::vector<Split*>>& splits, Expression* expr) {
    Shell* shell = expr->getShell();
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        int idx = 0;
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            if (split->type.compare("Split") != 0) {
                while(shapes[shellParamIdx][idx] == 1) {
                    idx++;
                }
                if (split->type.compare("IndexSplit") == 0) {
                    shapes[shellParamIdx][idx] = 1;
                }
            }
            if (split->type.compare("RegularSplit") == 0) {
                RegularSplit* rsp2 = new RegularSplit();
                RegularSplit* rsp1 = static_cast<RegularSplit*>(split);
                rsp2->type = "RegularSplit";
                rsp2->setId(rsp1->getId());
                rsp2->setSplitNumber(rsp1->getSplitNumber());
                rsp2->setSplitSize(rsp1->getSplitSize());
                rsp2->setSplitStride(rsp1->getSplitStride());
                rsp2->setDimIdx(idx++);
                splits[shellParamIdx].push_back(rsp2);
            }
            // 降维
            else if(split->type.compare("IndexSplit") == 0) {
                IndexSplit* isp2 = new IndexSplit();
                IndexSplit* isp1 = static_cast<IndexSplit*>(split);
                isp2->type = "IndexSplit";
                isp2->setId(isp1->getId());
                isp2->setSplitNumber(isp1->getSplitNumber());
                isp2->setDimIdx(idx++);
                splits[shellParamIdx].push_back(isp2);
            } else {
                Split* newSplit = new Split();
                newSplit->setDimIdx(idx++);
                splits[shellParamIdx].push_back(newSplit);
            }
        }
    }
    Calc* calc = expr->getCalc();
    if (calc->getNumExprs() != 0) {
        addSplit(shapes, splits, calc->getExpr(0));
    }
}






/**
 * 生成子数据关联表达式相关代码
 * 包括数据重排
 */
std::string dacppTranslator::Rewriter::generateChildExpr2(std::vector<std::string> ancestorParamNames,
                                                          std::vector<int> ancestorMem,
                                                          Dac_Ops fatherOps,
                                                          std::vector<Dac_Ops> fatherParamOps,
                                                          int fatherInputNumber,
                                                          int fatherOutputNumber,
                                                          Expression* expr) {
    Shell* shell = expr->getShell();
    int shellParamNumber = shell->getNumShellParams();

    // 数据重组
    std::string dataRecon = "";
    for (int shellParamIdx = 0; shellParamIdx < shellParamNumber; shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        std::string toolPushBack = "";
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            if (split->type.compare("Split") == 0) {
                continue;
            }
            toolPushBack += CodeGen_OpPushBack2Tool(ancestorParamNames[shellParamIdx],
                                                    split->getId(),
                                                    std::to_string(split->getDimIdx()),
                                                    std::to_string(split->getSplitNumber()));
        }
        dataRecon += CodeGen_DataReconstructOpPush(ancestorParamNames[shellParamIdx], toolPushBack);
    }

    // 主机数据移动至设备
    // data host->device
    std::string h2DMemMove = "";
    for (int shellParamIdx = 0; shellParamIdx < shellParamNumber; shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        h2DMemMove += CodeGen_H2DMemMov(shellParam->getBasicType(),
                                        ancestorParamNames[shellParamIdx], 
                                        std::to_string(ancestorMem[shellParamIdx]));
    }

    // 计算输入和输出划分数量
    int inputNumber = fatherInputNumber;
    int outputNumber = fatherOutputNumber;
    std::set<std::string> inputOpSet;
    std::set<std::string> outputOpSet;
    for (int shellParamIdx = 0; shellParamIdx < shellParamNumber; shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if(shellParam->getRw() == 1) {
            for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                Split* split = shellParam->getSplit(splitIdx);
                if(outputOpSet.count(split->getId()) == 1) {
                    continue;
                }
                outputOpSet.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    outputNumber *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    outputNumber *= sp->getSplitNumber();
                }
            }
        } else {
            for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                Split* split = shellParam->getSplit(splitIdx);
                if(inputOpSet.count(split->getId()) == 1) {
                    continue;
                }
                inputOpSet.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    inputNumber *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    inputNumber *= sp->getSplitNumber();
                }
            }
        }
    }





    // 数据偏移
    std::vector<std::vector<int>> offset(shell->getNumShellParams(), std::vector<int>());
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        
        // 首先计算ShellParam中数据的数量
        int count = 1;
        for (int dimIdx = 0; dimIdx < shellParam->getDim(); dimIdx++) {
            count *= shellParam->getShape(dimIdx);
        }
        // 根据算子计算数据的偏移
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* sp = shellParam->getSplit(splitIdx);
            // 规则分区
            if (sp->type.compare("RegularSplit") == 0) {
                RegularSplit* rsp = static_cast<RegularSplit*>(sp);
                count = count / shellParam->getShape(rsp->getDimIdx()) * rsp->getSplitSize();
                offset[shellParamIdx].push_back(count);
            }
            // 降维
            else if(sp->type.compare("IndexSplit") == 0) {
                IndexSplit* isp = static_cast<IndexSplit*>(sp);
                count = count / shellParam->getShape(isp->getDimIdx());
                offset[shellParamIdx].push_back(count);
            }
            // 保形
            else {
                offset[shellParamIdx].push_back(count);
            }
        }
    }

    // 计算输入和输出划分数量
    int in = 1;
    int out = 1;
    std::set<std::string> setIn;
    std::set<std::string> setOut;
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if(shellParam->getRw() == 1) {
            for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                Split* split = shellParam->getSplit(splitIdx);
                if(setOut.count(split->getId()) == 1) {
                    continue;
                }
                setOut.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    out *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    out *= sp->getSplitNumber();
                }
            }
        } else {
            for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                Split* split = shellParam->getSplit(splitIdx);
                if(setIn.count(split->getId()) == 1) {
                    continue;
                }
                setIn.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    in *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    in *= sp->getSplitNumber();
                }
            }
        }
    }

    // 计算数据重排需要分配的空间
    int* mem = new int[shell->getNumShellParams()];
    // 数据关联表达式shellParam的形状，上一步获取所有算子时修改过，恢复回来
    std::vector<std::vector<int>> shapesCopy(shell->getNumShellParams(), std::vector<int>());
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        mem[shellParamIdx] = 1;
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for (int shapeIdx = 0; shapeIdx < shellParam->getDim(); shapeIdx++) {
            shapesCopy[shellParamIdx].push_back(shellParam->getShape(shapeIdx));
            mem[shellParamIdx] *= shellParam->getShape(shapeIdx);
        }
    }
    for(int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for(int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            // 降维划分
            if(split->type.compare("IndexSplit") == 0) {
                shapesCopy[shellParamIdx][split->getDimIdx()] = 1;
            }
            // 规则分区划分
            else if(split->type.compare("RegularSplit") == 0) {
                dacppTranslator::RegularSplit* regularSplit = static_cast<dacppTranslator::RegularSplit*>(split);
                mem[shellParamIdx] *= shapesCopy[shellParamIdx][split->getDimIdx()] / ((shapesCopy[shellParamIdx][split->getDimIdx()] 
                                            - regularSplit->getSplitSize() + regularSplit->getSplitStride()) 
                                            / regularSplit->getSplitStride() * regularSplit->getSplitSize());
                shapesCopy[shellParamIdx][split->getDimIdx()] = regularSplit->getSplitSize();
            }
        }
    }
    for(int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        if(shell->getShellParam(shellParamIdx)->getRw() == 1) {
            mem[shellParamIdx] *= in / out;
        }
    }

    // 数据重排
    std::string dataRecon = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        std::string toolPushBack = "";
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            if (split->type.compare("Split") == 0) {
                continue;
            }
            toolPushBack += CodeGen_OpPushBack2Tool(ancestor->getShell()->getShellParam(shellParamIdx)->getName(), split->getId(),
                                                 std::to_string(split->getDimIdx()), std::to_string(offset[shellParamIdx][splitIdx]));
        }
        dataRecon += CodeGen_DataReconstructOpPush(ancestor->getShell()->getShellParam(shellParamIdx)->getName(), toolPushBack);
    }

    // 主机数据移动至设备
    // data host->device
    std::string h2DMemMove = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        h2DMemMove += CodeGen_H2DMemMov(shellParam->getBasicType(), ancestor->getShell()->getShellParam(shellParamIdx)->getName(), 
                                        std::to_string(fatherMem[shellParamIdx]));
    }

    // parallel_for中的索引
    std::string indexInit = "";
    for (int splitIdx = 0; splitIdx < shell->getNumSplits(); splitIdx++) {
        Split* split = shell->getSplit(splitIdx);
        if(split->type.compare("RegularSplit") == 0) {
            RegularSplit* sp = static_cast<RegularSplit*>(split);
            RegularSlice s = RegularSlice(sp->getId(), sp->getSplitSize(), sp->getSplitStride());
            s.SetSplitSize(sp->getSplitNumber());
            ops.push_back(s);
        }
        else if(split->type.compare("IndexSplit") == 0) {
            IndexSplit* sp = static_cast<IndexSplit*>(split);
            Index s = Index(sp->getId());
            s.SetSplitSize(sp->getSplitNumber());
            ops.push_back(s);
        }
    }
    indexInit = CodeGen_IndexInit(ops);

    // parallel_for中的计算
    Args args = Args();
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            if(split->type.compare("RegularSplit") == 0) {
                RegularSplit* sp = static_cast<RegularSplit*>(split);
                RegularSlice s = RegularSlice(sp->getId(), sp->getSplitSize(), sp->getSplitStride());
                s.SetSplitSize(sp->getSplitNumber());
                s.setDimId(sp->getDimIdx());
                s.setSplitLength(offset[shellParamIdx][splitIdx]);
                tensorsOps[shellParamIdx].push_back(s);
            }
            else if(split->type.compare("IndexSplit") == 0) {
                IndexSplit* sp = static_cast<IndexSplit*>(split);
                Index s = Index(sp->getId());
                s.SetSplitSize(sp->getSplitNumber());
                s.setDimId(sp->getDimIdx());
                s.setSplitLength(offset[shellParamIdx][splitIdx]);
                tensorsOps[shellParamIdx].push_back(s);
            }
        }
        DacData d_tensor = DacData("d_" + ancestor->getShell()->getShellParam(shellParamIdx)->getName(), shellParam->getDim(), tensorsOps[shellParamIdx]);
        args.push_back(d_tensor);
    }

    Calc* calc = expr->getCalc();

    std::string calcEmbed = CodeGen_CalcEmbed(calc->getName(), args);
	std::string kernelExecute = CodeGen_KernelExecute("64", indexInit, calcEmbed);

    // 规约
    std::string reduction = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if (shellParam->getRw() == 1) {
            reduction += CodeGen_Reduction_Span(std::to_string(fatherMem[shellParamIdx]), "2",
                                                "1", ancestor->getShell()->getShellParam(shellParamIdx)->getName(), shellParam->getBasicType(), "sycl::plus<>()");
        }
    }
    
    // data device->host
    std::string d2HMemMove = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if (shellParam->getRw() == 1) {
            d2HMemMove += CodeGen_D2HMemMov(ancestor->getShell()->getShellParam(shellParamIdx)->getName(), shellParam->getBasicType(),
                                            std::to_string(fatherMem[shellParamIdx]), false);
        }
    }

    std::string dac = CodeGen_DataAssocComp(dataRecon, h2DMemMove, kernelExecute, reduction, d2HMemMove);

    for (int exprIdx = 0; exprIdx < calc->getNumExprs(); exprIdx++) {
        dac += generateChildExpr(ancestor, calc->getExpr(exprIdx), ops, tensorsOps, fatherMem);
    }

    // 算子弹出
    std::string dataReconReverse = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        std::string toolPopBack = "";
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            toolPopBack += CodeGen_OpPopFromTool(ancestor->getShell()->getShellParam(shellParamIdx)->getName());
        }
        dataReconReverse += CodeGen_DataReconstructOpPop(ancestor->getShell()->getShellParam(shellParamIdx)->getName(), toolPopBack);
    }

    return dac + dataReconReverse;
}



/**
 * 生成子数据关联表达式相关代码
 * 包括数据重排
 */
std::string dacppTranslator::Rewriter::generateChildExpr(Expression* ancestor, Expression* expr, Dac_Ops& ops, std::vector<Dac_Ops>& tensorsOps,
                                                         int* fatherMem) {
    Shell* shell = expr->getShell();

    // 数据偏移
    std::vector<std::vector<int>> offset(shell->getNumShellParams(), std::vector<int>());
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        
        // 首先计算ShellParam中数据的数量
        int count = 1;
        for (int dimIdx = 0; dimIdx < shellParam->getDim(); dimIdx++) {
            count *= shellParam->getShape(dimIdx);
        }
        // 根据算子计算数据的偏移
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* sp = shellParam->getSplit(splitIdx);
            // 规则分区
            if (sp->type.compare("RegularSplit") == 0) {
                RegularSplit* rsp = static_cast<RegularSplit*>(sp);
                count = count / shellParam->getShape(rsp->getDimIdx()) * rsp->getSplitSize();
                offset[shellParamIdx].push_back(count);
            }
            // 降维
            else if(sp->type.compare("IndexSplit") == 0) {
                IndexSplit* isp = static_cast<IndexSplit*>(sp);
                count = count / shellParam->getShape(isp->getDimIdx());
                offset[shellParamIdx].push_back(count);
            }
            // 保形
            else {
                offset[shellParamIdx].push_back(count);
            }
        }
    }

    // 计算输入和输出划分数量
    int in = 1;
    int out = 1;
    std::set<std::string> setIn;
    std::set<std::string> setOut;
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if(shellParam->getRw() == 1) {
            for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                Split* split = shellParam->getSplit(splitIdx);
                if(setOut.count(split->getId()) == 1) {
                    continue;
                }
                setOut.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    out *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    out *= sp->getSplitNumber();
                }
            }
        } else {
            for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                Split* split = shellParam->getSplit(splitIdx);
                if(setIn.count(split->getId()) == 1) {
                    continue;
                }
                setIn.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    in *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    in *= sp->getSplitNumber();
                }
            }
        }
    }

    // 计算数据重排需要分配的空间
    int* mem = new int[shell->getNumShellParams()];
    // 数据关联表达式shellParam的形状，上一步获取所有算子时修改过，恢复回来
    std::vector<std::vector<int>> shapesCopy(shell->getNumShellParams(), std::vector<int>());
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        mem[shellParamIdx] = 1;
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for (int shapeIdx = 0; shapeIdx < shellParam->getDim(); shapeIdx++) {
            shapesCopy[shellParamIdx].push_back(shellParam->getShape(shapeIdx));
            mem[shellParamIdx] *= shellParam->getShape(shapeIdx);
        }
    }
    for(int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for(int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            // 降维划分
            if(split->type.compare("IndexSplit") == 0) {
                shapesCopy[shellParamIdx][split->getDimIdx()] = 1;
            }
            // 规则分区划分
            else if(split->type.compare("RegularSplit") == 0) {
                dacppTranslator::RegularSplit* regularSplit = static_cast<dacppTranslator::RegularSplit*>(split);
                mem[shellParamIdx] *= shapesCopy[shellParamIdx][split->getDimIdx()] / ((shapesCopy[shellParamIdx][split->getDimIdx()] 
                                            - regularSplit->getSplitSize() + regularSplit->getSplitStride()) 
                                            / regularSplit->getSplitStride() * regularSplit->getSplitSize());
                shapesCopy[shellParamIdx][split->getDimIdx()] = regularSplit->getSplitSize();
            }
        }
    }
    for(int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        if(shell->getShellParam(shellParamIdx)->getRw() == 1) {
            mem[shellParamIdx] *= in / out;
        }
    }

    // 数据重排
    std::string dataRecon = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        std::string toolPushBack = "";
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            if (split->type.compare("Split") == 0) {
                continue;
            }
            toolPushBack += CodeGen_OpPushBack2Tool(ancestor->getShell()->getShellParam(shellParamIdx)->getName(), split->getId(),
                                                 std::to_string(split->getDimIdx()), std::to_string(offset[shellParamIdx][splitIdx]));
        }
        dataRecon += CodeGen_DataReconstructOpPush(ancestor->getShell()->getShellParam(shellParamIdx)->getName(), toolPushBack);
    }

    // 主机数据移动至设备
    // data host->device
    std::string h2DMemMove = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        h2DMemMove += CodeGen_H2DMemMov(shellParam->getBasicType(), ancestor->getShell()->getShellParam(shellParamIdx)->getName(), 
                                        std::to_string(fatherMem[shellParamIdx]));
    }

    // parallel_for中的索引
    std::string indexInit = "";
    for (int splitIdx = 0; splitIdx < shell->getNumSplits(); splitIdx++) {
        Split* split = shell->getSplit(splitIdx);
        if(split->type.compare("RegularSplit") == 0) {
            RegularSplit* sp = static_cast<RegularSplit*>(split);
            RegularSlice s = RegularSlice(sp->getId(), sp->getSplitSize(), sp->getSplitStride());
            s.SetSplitSize(sp->getSplitNumber());
            ops.push_back(s);
        }
        else if(split->type.compare("IndexSplit") == 0) {
            IndexSplit* sp = static_cast<IndexSplit*>(split);
            Index s = Index(sp->getId());
            s.SetSplitSize(sp->getSplitNumber());
            ops.push_back(s);
        }
    }
    indexInit = CodeGen_IndexInit(ops);

    // parallel_for中的计算
    Args args = Args();
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            if(split->type.compare("RegularSplit") == 0) {
                RegularSplit* sp = static_cast<RegularSplit*>(split);
                RegularSlice s = RegularSlice(sp->getId(), sp->getSplitSize(), sp->getSplitStride());
                s.SetSplitSize(sp->getSplitNumber());
                s.setDimId(sp->getDimIdx());
                s.setSplitLength(offset[shellParamIdx][splitIdx]);
                tensorsOps[shellParamIdx].push_back(s);
            }
            else if(split->type.compare("IndexSplit") == 0) {
                IndexSplit* sp = static_cast<IndexSplit*>(split);
                Index s = Index(sp->getId());
                s.SetSplitSize(sp->getSplitNumber());
                s.setDimId(sp->getDimIdx());
                s.setSplitLength(offset[shellParamIdx][splitIdx]);
                tensorsOps[shellParamIdx].push_back(s);
            }
        }
        DacData d_tensor = DacData("d_" + ancestor->getShell()->getShellParam(shellParamIdx)->getName(), shellParam->getDim(), tensorsOps[shellParamIdx]);
        args.push_back(d_tensor);
    }

    Calc* calc = expr->getCalc();

    std::string calcEmbed = CodeGen_CalcEmbed(calc->getName(), args);
	std::string kernelExecute = CodeGen_KernelExecute("64", indexInit, calcEmbed);

    // 规约
    std::string reduction = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if (shellParam->getRw() == 1) {
            reduction += CodeGen_Reduction_Span(std::to_string(fatherMem[shellParamIdx]), "2",
                                                "1", ancestor->getShell()->getShellParam(shellParamIdx)->getName(), shellParam->getBasicType(), "sycl::plus<>()");
        }
    }
    
    // data device->host
    std::string d2HMemMove = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if (shellParam->getRw() == 1) {
            d2HMemMove += CodeGen_D2HMemMov(ancestor->getShell()->getShellParam(shellParamIdx)->getName(), shellParam->getBasicType(),
                                            std::to_string(fatherMem[shellParamIdx]), false);
        }
    }

    std::string dac = CodeGen_DataAssocComp(dataRecon, h2DMemMove, kernelExecute, reduction, d2HMemMove);

    for (int exprIdx = 0; exprIdx < calc->getNumExprs(); exprIdx++) {
        dac += generateChildExpr(ancestor, calc->getExpr(exprIdx), ops, tensorsOps, fatherMem);
    }

    // 算子弹出
    std::string dataReconReverse = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        std::string toolPopBack = "";
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            toolPopBack += CodeGen_OpPopFromTool(ancestor->getShell()->getShellParam(shellParamIdx)->getName());
        }
        dataReconReverse += CodeGen_DataReconstructOpPop(ancestor->getShell()->getShellParam(shellParamIdx)->getName(), toolPopBack);
    }

    return dac + dataReconReverse;
}

/**
 * 生成sycl函数的代码
 * @param expr 数据关联计算表达式
 */
std::string dacppTranslator::Rewriter::generateSyclFunc(Expression* expr) {
    Shell* shell = expr->getShell();

    // sycl函数名
    std::string dacShellName = shell->getName();
    
    // sycl函数参数列表
    std::string dacShellParams = "";
    for (int count = 0; count < shell->getNumParams(); count++) {
        Param* param = shell->getParam(count);
        dacShellParams += param->getType() + " " + param->getName();
        if(count != shell->getNumParams() - 1) { 
            dacShellParams += ", ";
        }
    }

    // 设备上需要分配的空间
    int* deviceMem = new int[shell->getNumShellParams()];

    // 存放顶级数据关联计算表达式到最底层数据关联计算表达式中的所有算子
    std::vector<std::vector<Split*>> splits(shell->getNumShellParams(), std::vector<Split*>());
    // 顶级数据关联表达式shellParam的形状
    std::vector<std::vector<int>> shapes(shell->getNumShellParams(), std::vector<int>());
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for (int shapeIdx = 0; shapeIdx < shellParam->getDim(); shapeIdx++) {
            shapes[shellParamIdx].push_back(shellParam->getShape(shapeIdx));
        }
    }
    addSplit(shapes, splits, expr);

    // 计算最终的输入和输出划分数量
    int allIn = 1;
    int allOut = 1;
    std::set<std::string> setAllIn;
    std::set<std::string> setAllOut;
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        if(shell->getShellParam(shellParamIdx)->getRw() == 1) {
            for (int splitIdx = 0; splitIdx < splits[shellParamIdx].size(); splitIdx++) {
                Split* split = splits[shellParamIdx][splitIdx];
                if(setAllOut.count(split->getId()) == 1) {
                    continue;
                }
                setAllOut.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    allOut *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    allOut *= sp->getSplitNumber();
                }
            }
        } else {
            for (int splitIdx = 0; splitIdx < splits[shellParamIdx].size(); splitIdx++) {
                Split* split = splits[shellParamIdx][splitIdx];
                if(setAllIn.count(split->getId()) == 1) {
                    continue;
                }
                setAllIn.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    allIn *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    allIn *= sp->getSplitNumber();
                }
            }
        }
    }

    // 计算设备上需要分配的空间
    // 顶级数据关联表达式shellParam的形状，上一步获取所有算子时修改过，恢复回来
    std::vector<std::vector<int>> shapesCopy(shell->getNumShellParams(), std::vector<int>());
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        deviceMem[shellParamIdx] = 1;
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for (int shapeIdx = 0; shapeIdx < shellParam->getDim(); shapeIdx++) {
            shapesCopy[shellParamIdx].push_back(shellParam->getShape(shapeIdx));
            deviceMem[shellParamIdx] *= shellParam->getShape(shapeIdx);
        }
    }
    for(int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for(int splitIdx = 0; splitIdx < splits[shellParamIdx].size(); splitIdx++) {
            Split* split = splits[shellParamIdx][splitIdx];
            // 降维划分
            if(split->type.compare("IndexSplit") == 0) {
                shapesCopy[shellParamIdx][split->getDimIdx()] = 1;
            }
            // 规则分区划分
            else if(split->type.compare("RegularSplit") == 0) {
                dacppTranslator::RegularSplit* regularSplit = static_cast<dacppTranslator::RegularSplit*>(split);
                deviceMem[shellParamIdx] *= shapesCopy[shellParamIdx][split->getDimIdx()] / ((shapesCopy[shellParamIdx][split->getDimIdx()] 
                                            - regularSplit->getSplitSize() + regularSplit->getSplitStride()) 
                                            / regularSplit->getSplitStride() * regularSplit->getSplitSize());
                shapesCopy[shellParamIdx][split->getDimIdx()] = regularSplit->getSplitSize();
            }
        }
    }
    for(int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        if(shell->getShellParam(shellParamIdx)->getRw() == 1) {
            deviceMem[shellParamIdx] *= allIn / allOut;
        }
    }

    // 设备内存分配
    std::string deviceMemAlloc = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        deviceMemAlloc += CodeGen_DeviceMemAlloc(shellParam->getBasicType(), shellParam->getName(), 
                                                 std::to_string(deviceMem[shellParamIdx]));
    }

    // 算子初始化
    std::string opInit = "";
    std::set<std::string> splitSet;
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        for (int splitIdx = 0; splitIdx < splits[shellParamIdx].size(); splitIdx++) {
            Split* split = splits[shellParamIdx][splitIdx];
            if(splitSet.count(split->getId()) == 1) {
                continue;
            }
            splitSet.insert(split->getId());
            if(split->type.compare("RegularSplit") == 0) {
                RegularSplit* sp = static_cast<RegularSplit*>(split);
                opInit += CodeGen_RegularSliceInit(sp->getId(), std::to_string(sp->getSplitSize()), 
                                                   std::to_string(sp->getSplitStride()), std::to_string(sp->getSplitNumber()));
            }
            else if(split->type.compare("IndexSplit") == 0) {
                IndexSplit* sp = static_cast<IndexSplit*>(split);
                opInit += CodeGen_IndexInit(sp->getId(), std::to_string(sp->getSplitNumber()));
            }
        }
    }

    // 数据偏移
    std::vector<std::vector<int>> offset(shell->getNumShellParams(), std::vector<int>());
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        // 首先计算ShellParam中数据的数量
        int count = 1;
        for (int dimIdx = 0; dimIdx < shellParam->getDim(); dimIdx++) {
            count *= shellParam->getShape(dimIdx);
        }
        // 根据算子计算数据的偏移
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* sp = shellParam->getSplit(splitIdx);
            // 规则分区
            if (sp->type.compare("RegularSplit") == 0) {
                RegularSplit* rsp = static_cast<RegularSplit*>(sp);
                count = count / shellParam->getShape(rsp->getDimIdx()) * rsp->getSplitSize();
                offset[shellParamIdx].push_back(count);
            }
            // 降维
            else if(sp->type.compare("IndexSplit") == 0) {
                IndexSplit* isp = static_cast<IndexSplit*>(sp);
                count = count / shellParam->getShape(isp->getDimIdx());
                offset[shellParamIdx].push_back(count);
            }
            // 保形
            else {
                offset[shellParamIdx].push_back(count);
            }
        }
    }

    // 计算顶级数据关联表达式输入和输出划分数量
    int in = 1;
    int out = 1;
    std::set<std::string> setIn;
    std::set<std::string> setOut;
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if(shellParam->getRw() == 1) {
            for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                Split* split = shellParam->getSplit(splitIdx);
                if(setOut.count(split->getId()) == 1) {
                    continue;
                }
                setOut.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    out *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    out *= sp->getSplitNumber();
                }
            }
        } else {
            for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
                Split* split = shellParam->getSplit(splitIdx);
                if(setIn.count(split->getId()) == 1) {
                    continue;
                }
                setIn.insert(split->getId());
                if(split->type.compare("RegularSplit") == 0) {
                    RegularSplit* sp = static_cast<RegularSplit*>(split);
                    in *= sp->getSplitNumber();
                }
                else if(split->type.compare("IndexSplit") == 0) {
                    IndexSplit* sp = static_cast<IndexSplit*>(split);
                    in *= sp->getSplitNumber();
                }
            }
        }
    }

    // 设备规约内存分配
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if (shellParam->getRw() == 1) {
            deviceMemAlloc += CodeGen_DeviceMemAllocReduction(shellParam->getBasicType(), shellParam->getName(), 
                                                              std::to_string(deviceMem[shellParamIdx] / allIn * allOut * in / out));
        }
    }

    // 计算数据重排需要分配的空间
    int* mem = new int[shell->getNumShellParams()];
    // 数据关联表达式shellParam的形状，上一步获取所有算子时修改过，恢复回来
    std::vector<std::vector<int>> shapesCopyAgain(shell->getNumShellParams(), std::vector<int>());
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        mem[shellParamIdx] = 1;
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for (int shapeIdx = 0; shapeIdx < shellParam->getDim(); shapeIdx++) {
            shapesCopyAgain[shellParamIdx].push_back(shellParam->getShape(shapeIdx));
            mem[shellParamIdx] *= shellParam->getShape(shapeIdx);
        }
    }
    for(int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        for(int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            // 降维划分
            if(split->type.compare("IndexSplit") == 0) {
                shapesCopyAgain[shellParamIdx][split->getDimIdx()] = 1;
            }
            // 规则分区划分
            else if(split->type.compare("RegularSplit") == 0) {
                dacppTranslator::RegularSplit* regularSplit = static_cast<dacppTranslator::RegularSplit*>(split);
                mem[shellParamIdx] *= shapesCopyAgain[shellParamIdx][split->getDimIdx()] / ((shapesCopyAgain[shellParamIdx][split->getDimIdx()] 
                                            - regularSplit->getSplitSize() + regularSplit->getSplitStride()) 
                                            / regularSplit->getSplitStride() * regularSplit->getSplitSize());
                shapesCopyAgain[shellParamIdx][split->getDimIdx()] = regularSplit->getSplitSize();
            }
        }
    }
    for(int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        if(shell->getShellParam(shellParamIdx)->getRw() == 1) {
            mem[shellParamIdx] *= in / out;
        }
    }

    // 数据重排
    std::string dataRecon = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        std::string opPushBack = "";
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            if (split->type.compare("Split") == 0) {
                continue;
            }
            opPushBack += CodeGen_OpPushBack2Ops(shellParam->getName(), split->getId(),
                                                 std::to_string(split->getDimIdx()), std::to_string(offset[shellParamIdx][splitIdx]));
        }
        std::string dataOpsInit = CodeGen_DataOpsInit(shellParam->getName(), opPushBack);
        dataRecon += CodeGen_DataReconstruct(shellParam->getBasicType(), shellParam->getName(),
                                             std::to_string(deviceMem[shellParamIdx]), dataOpsInit);
    }

    // 主机数据移动至设备
    // data host->device
    std::string h2DMemMove = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        h2DMemMove += CodeGen_H2DMemMov(shellParam->getBasicType(), shellParam->getName(), 
                                        std::to_string(deviceMem[shellParamIdx]));
    }

    // parallel_for中的索引
    std::string indexInit = "";
    Dac_Ops ops;
    for (int splitIdx = 0; splitIdx < shell->getNumSplits(); splitIdx++) {
        Split* split = shell->getSplit(splitIdx);
        if(split->type.compare("RegularSplit") == 0) {
            RegularSplit* sp = static_cast<RegularSplit*>(split);
            RegularSlice s = RegularSlice(sp->getId(), sp->getSplitSize(), sp->getSplitStride());
            s.SetSplitSize(sp->getSplitNumber());
            ops.push_back(s);
        }
        else if(split->type.compare("IndexSplit") == 0) {
            IndexSplit* sp = static_cast<IndexSplit*>(split);
            Index s = Index(sp->getId());
            s.SetSplitSize(sp->getSplitNumber());
            ops.push_back(s);
        }
    }
    indexInit = CodeGen_IndexInit(ops);

    // // 判断是否需要添加算子
    //     bool exop = 0;
    //     if(countIn > countOut)
    //         exop = 1;
    //     Dac_Ops ExOps;
        
    //     //添加算子
    //     if(exop){
    //         int CountExOp = 0;
    //         std::set<std::string> setOut;
    //         for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
    //             ShellParam* shellParam = shell->getShellParam(shellParamIdx);
    //             if(shellParam->getRw() == 1) {
    //                 for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
    //                     Split* split = shellParam->getSplit(splitIdx);
    //                     if(setOut.count(split->getId()) == 1) {
    //                         continue;
    //                     }
    //                     setOut.insert(split->getId());
    //                 }
    //             }
    //         }
    //         for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
    //             ShellParam* shellParam = shell->getShellParam(shellParamIdx);
    //             if(shellParam->getRw() == 0) {
    //                 for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
    //                     Split* split = shellParam->getSplit(splitIdx);
    //                     if(setOut.count(split->getId()) == 1) {
    //                         continue;
    //                     }
    //                     Dac_Op op = Dac_Op(split->getId(), split->getDimIdx(), splitIdx);//要改的
    //                     /*通过 算子名称，算子划分数，算子作用的维度 创建算子. Dac_Op(std::string Name,int SplitSize,int dimId);*/
    //                     op.setSplitLength(countOut);
    //                     ExOps.push_back(op);
    //                     setOut.insert(split->getId());
    //                     CountExOp++;
    //                 }
    //             }
    //         }
    //     }

    // parallel_for中的计算
    Args args = Args();
    std::vector<Dac_Ops> tensorsOps(shell->getNumShellParams());
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        Dac_Ops tensorOps;
        for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
            Split* split = shellParam->getSplit(splitIdx);
            if(split->type.compare("RegularSplit") == 0) {
                RegularSplit* sp = static_cast<RegularSplit*>(split);
                RegularSlice s = RegularSlice(sp->getId(), sp->getSplitSize(), sp->getSplitStride());
                s.SetSplitSize(sp->getSplitNumber());
                s.setDimId(sp->getDimIdx());
                s.setSplitLength(offset[shellParamIdx][splitIdx]);
                tensorOps.push_back(s);
                tensorsOps.push_back(tensorOps);
            }
            else if(split->type.compare("IndexSplit") == 0) {
                IndexSplit* sp = static_cast<IndexSplit*>(split);
                Index s = Index(sp->getId());
                s.SetSplitSize(sp->getSplitNumber());
                s.setDimId(sp->getDimIdx());
                s.setSplitLength(offset[shellParamIdx][splitIdx]);
                tensorOps.push_back(s);
                tensorsOps.push_back(tensorOps);
            }
        }
        DacData d_tensor = DacData("d_" + shellParam->getName(), shellParam->getDim(), tensorOps);
        args.push_back(d_tensor);
    }

    Calc* calc = expr->getCalc();

    std::string calcEmbed = CodeGen_CalcEmbed(calc->getName(), args);
	std::string kernelExecute = CodeGen_KernelExecute(std::to_string(in), indexInit, calcEmbed);

    // 规约
    std::string reduction = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if (shellParam->getRw() == 1) {
            reduction += CodeGen_Reduction_Span(std::to_string(mem[shellParamIdx] / in * out), std::to_string(in / out), 
                                                "4", shellParam->getName(), shellParam->getBasicType(), "sycl::plus<>()");
        }
    }
    
    // data device->host
    std::string d2HMemMove = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if (shellParam->getRw() == 1) {
            d2HMemMove += CodeGen_D2HMemMov(shellParam->getName(), shellParam->getBasicType(), std::to_string(mem[shellParamIdx] / in * out), false);
        }
    }

    std::string dac = CodeGen_DataAssocComp(dataRecon, h2DMemMove, kernelExecute, reduction, d2HMemMove);

    for (int exprIdx = 0; exprIdx < calc->getNumExprs(); exprIdx++) {
        dac += generateChildExpr(expr, calc->getExpr(exprIdx), ops, tensorsOps, mem);
    }

    std::string reduction2 = "";
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        if (shellParam->getRw() == 1) {
            reduction2 += CodeGen_Reduction_Span(std::to_string(mem[shellParamIdx] / in * out), std::to_string(in / out), 
                                                "4", shellParam->getName(), shellParam->getBasicType(), "sycl::plus<>()");
        }
    }

    // 设备内存释放
    std::string memFree = reduction2;
    for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
        ShellParam* shellParam = shell->getShellParam(shellParamIdx);
        memFree += CodeGen_MemFree(shellParam->getName());
    }

    std::string syclFunc = CodeGen_DAC2SYCL(dacShellName, dacShellParams, deviceMemAlloc,
                                       opInit, dac, memFree);
    
    return syclFunc;
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

        code += generateCalc(calc);

        code += generateSyclFunc(expr) + "\n\n";

        // 删除原来的shell和calc
        rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
        rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());

        // 插入新生成的sycl函数
        rewriter->InsertText(dacppFile->getMainFuncLoc()->getBeginLoc(), code);
    }
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
    /*
    for (int exprCount = 0; exprCount < dacppFile->getNumExpression(); exprCount++) {
        Expression* expr = dacppFile->getExpression(exprCount);
        Shell* shell = expr->getShell();
        std::string code = "";
        code += shell->getName() + "(";
        for(int paramCount = 0; paramCount < shell->getNumParams(); paramCount++) {
            code += shell->getParam(paramCount)->getName();
            if(paramCount != shell->getNumParams() - 1) {
                code += ", ";
            }
        }
        code += ");";
        expr->getDacExpr()->dump();
        rewriter->InsertText(expr->getDacExpr()->getBeginLoc(), code);   
    }
    */
}