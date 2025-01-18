// #include <set>
// #include <iostream>
// #include <sstream>
// #include "clang/Rewrite/Core/Rewriter.h"
// #include "Rewriter.h"
// #include "Split.h"
// #include "Param.h"
// #include "dacInfo.h"
// // #include "sub_template.h"
// #include "universal_template.h"
// #include "usm_template.h"
// // #include "buffer_template.h"


// void dacppTranslator::Rewriter::rewriteDac_usm() {

//     std::string code = "";

//     // 添加头文件
//     for(int i = 0; i < dacppFile->getNumHeaderFile(); i++) {
//         code += "#include " + dacppFile->getHeaderFile(i)->getName() + "\n";
//     }
//     code += "\n";
    
//     // 添加命名空间
//     for(int i = 0; i < dacppFile->getNumNameSpace(); i++) {
//         code += "using namespace " + dacppFile->getNameSpace(i)->getName() + ";\n";
//     }
//     code += "\n";
    
//     // 添加数据关联表达式对应的划分结构和计算结构
//     for(int i = 0; i < dacppFile->getNumExpression(); i++) {
//         Expression* expr = dacppFile->getExpression(i);
//         Shell* shell = expr->getShell();
//         Calc* calc = expr->getCalc();

//         std::vector<BINDINFO> info;
//         shell->GetBindInfo(&info);//获取binding信息

//         int countIn = 1;
//         int countOut = 1;
//         std::string set;
//         std::set<std::string> setIn;
//         std::set<std::string> setOut;
//         for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
//             ShellParam* shellParam = shell->getShellParam(shellParamIdx);
//             if(shellParam->getRw() == 1) {
//                 for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
//                     Split* split = shellParam->getSplit(splitIdx);
//                     for(int i = 0; i < info.size(); i++){
//                         if(shell->search_symbol(info[i].v)->getId() == split->getId())
//                             set = std::to_string(info[i].icls);
//                     }
//                     if(setOut.count(set) == 1 ) {//不重复计算同一个集合的算子
//                         continue;
//                     }
//                     setOut.insert(set);   
//                     if(split->type.compare("RegularSplit") == 0) {
//                         RegularSplit* sp = static_cast<RegularSplit*>(split);
//                         countOut *= sp->getSplitNumber();
//                     }
//                     else if(split->type.compare("IndexSplit") == 0) {
//                         IndexSplit* sp = static_cast<IndexSplit*>(split);
//                         countOut *= sp->getSplitNumber();
//                     }
//                 }
//             } else {
//                 for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
//                     Split* split = shellParam->getSplit(splitIdx);
//                     for(int i = 0; i < info.size(); i++){
//                         if(shell->search_symbol(info[i].v)->getId() == split->getId())
//                             set = std::to_string(info[i].icls);
//                     }
//                     if(setIn.count(set) == 1) {
//                         continue;
//                     }
//                     setIn.insert(set);
//                     if(split->type.compare("RegularSplit") == 0) {
//                         RegularSplit* sp = static_cast<RegularSplit*>(split);
//                         // std::cout << sp->getSplitNumber() <<std::endl;
//                         countIn *= sp->getSplitNumber();
//                     }
//                     else if(split->type.compare("IndexSplit") == 0) {
//                         IndexSplit* sp = static_cast<IndexSplit*>(split);
//                         // std::cout << sp->getSplitNumber() <<std::endl;
//                         countIn *= sp->getSplitNumber();
//                     }
//                 }
//             }
//         }
//         // std::cout << "countIn" <<countIn << std::endl; 
//         // std::cout << "countOut" <<countOut << std::endl; 
//         // 判断是否需要添加算子
//         bool exop = 0;
//         if(countIn > countOut)
//             exop = 1;
//         Dac_Ops ExOps;
//         Dac_Ops difop;//存储不同命的算子
//         int difsize = 0;//不同名的算子数量
//         //添加算子
//         if(exop){
//             int CountExOp = 0;
//             std::set<std::string> setOut;
//             for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
//                 ShellParam* shellParam = shell->getShellParam(shellParamIdx);
//                 if(shellParam->getRw() == 1) {
//                     for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
//                         Split* split = shellParam->getSplit(splitIdx);
//                         for(int i = 0; i < info.size(); i++){
//                             if(shell->search_symbol(info[i].v)->getId() == split->getId())
//                                 set = std::to_string(info[i].icls);//寻找改节点的划分信息
//                         }
//                         if(setOut.count(set) == 1) {//同一个划分的节点视为同名
//                             continue;
//                         }
//                         Dac_Op op = Dac_Op(split->getId(), split->getDimIdx(), splitIdx);
//                         difop.push_back(op);//存储算子列表
//                         difsize++;
//                         setOut.insert(set);
//                     }
//                 }
//             }
//             for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
//                 ShellParam* shellParam = shell->getShellParam(shellParamIdx);
//                 if(shellParam->getRw() == 0) {
//                     for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
//                         Split* split = shellParam->getSplit(splitIdx);
//                         for(int i = 0; i < info.size(); i++){
//                             if(shell->search_symbol(info[i].v)->getId() == split->getId())
//                                 set = std::to_string(info[i].icls);
//                         }
//                         if(setOut.count(set) == 1) {
//                             continue;
//                         }
//                         Dac_Op op = Dac_Op(split->getId(), split->getDimIdx(), splitIdx);// 中间的getDimIdx应该是有问题的，但是着一步并不需要考虑算子的划分信息
//                         /*通过 算子名称，算子划分数，算子作用的维度 创建算子. Dac_Op(std::string Name,int SplitSize,int dimId);*/
//                         op.setSplitLength(countOut);
//                         ExOps.push_back(op);
//                         difop.push_back(op);//存储算子列表
//                         difsize++;
//                         setOut.insert(set);
//                         CountExOp++;
//                     }
//                 }
//             }
//         }

//         std::vector<std::vector<int>> offset(shell->getNumShellParams(), std::vector<int>());
//         for (int shellParamIdx = 0; shellParamIdx < shell->getNumShellParams(); shellParamIdx++) {
//             ShellParam* shellParam = shell->getShellParam(shellParamIdx);
//             // 首先计算ShellParam中数据的数量
//             int count = 1;
//             for (int dim = 0; dim < shellParam->getDim(); dim++) {
//                 count *= shellParam->getShape(dim);
//             }
//             for (int splitIdx = 0; splitIdx < shellParam->getNumSplit(); splitIdx++) {
//                 Split* sp = shellParam->getSplit(splitIdx);
//                 // 规则分区
//                 if (sp->type.compare("RegularSplit") == 0) {
//                     RegularSplit* rsp = static_cast<RegularSplit*>(sp);
//                     count = count / shellParam->getShape(rsp->getDimIdx()) * rsp->getSplitSize();
//                     offset[shellParamIdx].push_back(count);
//                 }
//                 // 降维
//                 else if(sp->type.compare("IndexSplit") == 0) {
//                     IndexSplit* isp = static_cast<IndexSplit*>(sp);
//                     count = count / shellParam->getShape(isp->getDimIdx());
//                     offset[shellParamIdx].push_back(count);
//                 }
//                 // 保形
//                 else {
//                     offset[shellParamIdx].push_back(count);
//                 }
//             }
//         }


//         // 计算数据重排后需要的设备内存空间
//         int* mem = new int[shell->getNumShellParams()];
//         for(int j = 0; j < shell->getNumShellParams(); j++) {
//             ShellParam* shellParam = shell->getShellParam(j);
//             mem[j] = 1;
//             int dim = shellParam->getNumSplit();
//             for(int k = 0; k < dim; k++) {
//                 Split* split = shellParam->getSplit(k);
//                 // 降维划分
//                 if(split->type.compare("IndexSplit") == 0) {
//                     mem[j] *= shellParam->getShape(k);
//                 }
//                 // 规则分区划分
//                 else if(split->type.compare("RegularSplit") == 0) {
//                     dacppTranslator::RegularSplit* regulerSplit = static_cast<dacppTranslator::RegularSplit*>(split);
//                     mem[j] *= (shellParam->getShape(k) - regulerSplit->getSplitSize() + regulerSplit->getSplitStride()) / regulerSplit->getSplitStride() * regulerSplit->getSplitSize();
//                 }
//                 // 保形划分
//                 else {
//                     mem[j] *= shellParam->getShape(k);
//                 }
//             }
//         }

//         // 计算结构
//         code += "void " + calc->getName() + "(";
//         for(int count = 0; count < calc->getNumParams(); count++) {
//             code += calc->getParam(count)->getBasicType() + "* " + calc->getParam(count)->getName();
//             if(count != calc->getNumParams() - 1) {
//                 code += ", ";
//             }
//         }
//         code += ") \n";
//         for(int count = 0; count < calc->getNumBody(); count++) {
            
//             code += calc->getBody(count) + "\n";
//         }


//         std::string dacShellName = shell->getName();
        
//         std::string dacShellParams = "";
//         for (int count = 0; count < shell->getNumParams(); count++) {
//             Param* param = shell->getParam(count);
//             dacShellParams += param->getType() + " " + param->getName();
//             if(count != shell->getNumParams() - 1) { 
//                 dacShellParams += ", ";
//             }
//         }

//         // 算子初始化
//         std::string opInit = "";
//         for(int count = 0; count < shell->getNumSplits(); count++) {
//             Split* split = shell->getSplit(count);
//             if(split->type.compare("RegularSplit") == 0) {
//                 RegularSplit* sp = static_cast<RegularSplit*>(split);
//                 opInit += UNIVERSAL_TEMPLATE::CodeGen_RegularSliceInit(sp->getId(), std::to_string(sp->getSplitSize()), std::to_string(sp->getSplitStride()), std::to_string(sp->getSplitNumber()));
//             }
//             else if(split->type.compare("IndexSplit") == 0) {
//                 IndexSplit* sp = static_cast<IndexSplit*>(split);
//                 opInit += UNIVERSAL_TEMPLATE::CodeGen_IndexInit(sp->getId(), std::to_string(sp->getSplitNumber()));
//             }
//         }

//         std::string dataRecon = "";
//         for(int count = 0; count < shell->getNumShellParams(); count++) {
//             ShellParam* shellParam = shell->getShellParam(count);
//             std::string opPushBack = "";
//             int len = mem[count];
//             for(int count = 0; count < shellParam->getNumSplit(); count++) {
//                 Split* split = shellParam->getSplit(count);
//                 if(split->type == "dacpp::RegularSplit") {
//                     RegularSplit* sp = static_cast<RegularSplit*>(split);
//                     opPushBack += UNIVERSAL_TEMPLATE::CodeGen_OpPushBack(shellParam->getName(), sp->getId(), std::to_string(sp->getDimIdx()), std::to_string(len / shellParam->getShape(count) * sp->getSplitSize()));
//                     len = len / shellParam->getShape(count) * sp->getSplitSize();
//                 }
//                 else if(split->type == "IndexSplit") {
//                     IndexSplit* sp = static_cast<IndexSplit*>(split);
//                     opPushBack += UNIVERSAL_TEMPLATE::CodeGen_OpPushBack(shellParam->getName(), sp->getId(), std::to_string(sp->getDimIdx()), std::to_string(len / shellParam->getShape(count)));
//                     len /= shellParam->getShape(count);
//                 }
//             }
//             std::string dataOpsInit = UNIVERSAL_TEMPLATE::CodeGen_DataOpsInit(shellParam->getName(), opPushBack);
//             dataRecon += UNIVERSAL_TEMPLATE::CodeGen_DataReconstruct(shellParam->getBasicType(), shellParam->getName(), std::to_string(mem[count]), dataOpsInit);
//         }

//         // 设备内存分配
//         std::string deviceMemAlloc = "";
//         for(int count = 0; count < shell->getNumShellParams(); count++) {
//             ShellParam* shellParam = shell->getShellParam(count);
//             if (shellParam->getRw() == 1) {
//                 deviceMemAlloc += USM_TEMPLATE::CodeGen_DeviceMemAlloc(shellParam->getBasicType(), shellParam->getName(), std::to_string(mem[count] * countIn / countOut));
//             } else {
//                 deviceMemAlloc += USM_TEMPLATE::CodeGen_DeviceMemAlloc(shellParam->getBasicType(), shellParam->getName(), std::to_string(mem[count]));
//             }
//         }

//         std::string H2DMemMove = "";
//         for(int j = 0; j < shell->getNumShellParams(); j++) {
//             ShellParam* shellParam = shell->getShellParam(j);
//             if (shellParam->getRw() == 0) {
//                 H2DMemMove += USM_TEMPLATE::CodeGen_H2DMemMov(shellParam->getBasicType(), shellParam->getName(), std::to_string(mem[j]));
//             }
//         }
//         // 索引初始化
//         Dac_Ops ops;
//         std::vector<std::string> sets;  // 存储每个节点所属的连通分量的ID
//         std::vector<std::string> bindoffset;  // 存储每个节点相对于其连通分量代表节点的偏移
//         int componentID = 1;                               // 连通分量的编号
//         // std::string s;                                     // 用于存储 GetBindInfo 的偏移量字符串

//         for(int i = 0; i < info.size(); i++){
//             // std::cout << shell->search_symbol(info[i].v)->getId() << std::endl;
//             // std::cout << info[i].icls << std::endl;
//             if(info[i].offset.empty())
//                 info[i].offset = "0"; 
//             // std::cout << info[i].offset << std::endl;
//         }
//         for(int i = 0; i < info.size(); i++){
//             if(shell->search_symbol(info[i].v)->type.compare("IndexSplit") == 0) {
//                 dacppTranslator::IndexSplit* index = static_cast<dacppTranslator::IndexSplit*>(shell->search_symbol(info[i].v));
//                 Index tmp = Index(index->getId());
//                 tmp.SetSplitSize(index->getSplitNumber());
//                 tmp.setDimId(index->getDimIdx());
//                 ops.push_back(tmp);
//                 sets.push_back("id"+std::to_string(info[i].icls));
//                 bindoffset.push_back(info[i].offset);
//             }else if(shell->search_symbol(info[i].v)->type.compare("RegularSplit") == 0) {
//                 dacppTranslator::RegularSplit* r = static_cast<dacppTranslator::RegularSplit*>(shell->search_symbol(info[i].v));
//                 RegularSlice tmp = RegularSlice(r->getId(), r->getSplitSize(), r->getSplitStride());
//                 tmp.SetSplitSize(r->getSplitNumber());
//                 tmp.setDimId(r->getDimIdx());
//                 ops.push_back(tmp);
//                 sets.push_back("id"+std::to_string(info[i].icls));
//                 bindoffset.push_back(info[i].offset);
//             }
//         }
//         std::string IndexInit = UNIVERSAL_TEMPLATE::CodeGen_IndexInit(ops,sets,bindoffset);

//         // 嵌入计算
//         Args args = Args();
//         for(int j = 0; j < shell->getNumShellParams(); j++) {
//             ShellParam* shellParam = shell->getShellParam(j);
//             Dac_Ops ops;
//             for(int k = 0; k < shellParam->getNumSplit(); k++) {
//                 if(shellParam->getSplit(k)->getId() == "void") { continue;}
//                 Dac_Op op = Dac_Op(shellParam->getSplit(k)->getId(), shellParam->getSplit(k)->getDimIdx(), k);
//                 if(shellParam->getRw() == 1 && exop == 1){
//                     op.setSplitLength(offset[j][k] * countIn / countOut);
//                 }else{
//                     op.setSplitLength(offset[j][k]);
//                 }

//                 ops.push_back(op);
//             }
//             if(exop == 1 && shellParam->getRw() == 1 ){
//                 for(int i = 0; i < ExOps.size; i++)
//                 ops.push_back(ExOps[i]);
//             }
                        
//             DacData temp = DacData("d_" + shellParam->getName(), shellParam->getDim(), ops);
            
//             for(int dimIdx = 0; dimIdx < shellParam->getDim(); dimIdx++) {
//                 temp.setDimLength(dimIdx, shellParam->getShape(dimIdx));
//             }
            
//             args.push_back(temp);
//         }
        
//         std::string CalcEmbed = UNIVERSAL_TEMPLATE::CodeGen_CalcEmbed(calc->getName(), args);

//         // 内核执行
//         std::string kernelExecute = USM_TEMPLATE::CodeGen_KernelExecute(std::to_string(countIn), IndexInit, CalcEmbed);

//         // 归约结果返回
//         std::string reduction = "";
//         for(int j = 0; j < shell->getNumShellParams(); j++) {
//             ShellParam* shellParam = shell->getShellParam(j);
//             if(shellParam->getRw() == 1 && countIn != countOut) {
//                 std::string ReductionRule = "sycl::plus<>()";
//                 reduction += USM_TEMPLATE::CodeGen_Reduction_Span(std::to_string(mem[j]), std::to_string(countIn / countOut),
//                                                     std::to_string(countIn), shellParam->getName(), 
//                                                     shellParam->getBasicType(), ReductionRule);
//             }
//         }
//         // 归并结果返回
//         std::string D2HMemMove = "";
//         for(int j = 0; j < shell->getNumShellParams(); j++) {
//             ShellParam* shellParam = shell->getShellParam(j);
//             if(shellParam->getRw() == 1) {
//                 D2HMemMove += USM_TEMPLATE::CodeGen_D2HMemMov(shellParam->getName(), shellParam->getBasicType(), std::to_string(mem[j]), false);
//             }
//         }

//         // 内存释放
//         std::string memFree = "";
//         for(int j = 0; j < shell->getNumParams(); j++) {
//             ShellParam* shellParam = shell->getShellParam(j);
//             memFree += USM_TEMPLATE::CodeGen_MemFree(shellParam->getName());
//         }
        
//         code += USM_TEMPLATE::CodeGen_DAC2SYCL(dacShellName, dacShellParams, opInit, dataRecon, deviceMemAlloc, H2DMemMove, kernelExecute, reduction, D2HMemMove, memFree);
//         code += "\n\n";
//         rewriter->RemoveText(shell->getShellLoc()->getSourceRange());
//         rewriter->RemoveText(calc->getCalcLoc()->getSourceRange());
//     }

//     rewriter->InsertText(dacppFile->getMainFuncLoc()->getBeginLoc(), code);

// }

