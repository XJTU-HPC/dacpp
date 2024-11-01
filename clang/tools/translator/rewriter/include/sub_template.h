#ifndef SUB_TEMPLATE_H
#define SUB_TEMPLATE_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"dacInfo.h"

/*
	文本替换。
	将 text 中的 find 换成 replace。
*/
void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);
/*
	文本替换。
	对 templ 做 replacements 中的文本替换。
*/
std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

std::string CodeGen_DAC2SYCL(std::string dacShellName,std::string dacShellParams,std::string opInit,std::string dataRecon,
	std::string deviceMemAlloc,std::string H2DMemMove,std::string kernelExecute,std::string reduction,std::string D2HMemMove,std::string memFree);

std::string CodeGen_RegularSliceInit(std::string opName,std::string size,std::string stride,std::string splitSize);

std::string CodeGen_IndexInit(std::string opName,std::string splitSize);

std::string CodeGen_OpPushBack(std::string name, std::string opName, std::string dimId, std::string splitLength);

std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack);

std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit);

std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size);

std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size);

std::string CodeGen_KernelExecute(std::string SplitSize,std::string IndexInit,std::string CalcEmbed);

std::string CodeGen_IndexInit(Dac_Ops ops);

std::string CodeGen_CalcEmbed(std::string Name,Args args);

std::string CodeGen_Reduction(std::string SplitSize,std::string Name,std::string Type,std::string ReductionRule);

std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction);

std::string CodeGen_MemFree(std::string Name);

#endif