#ifndef USM_TEMPLATE_H
#define USM_TEMPLATE_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"dacInfo.h"

namespace USM_TEMPLATE {
    extern const char *DEVICE_MEM_ALLOC_Template;
    extern const char *H2D_MEM_MOV_Template;
    extern const char *KERNEL_EXECUTE_Template;
    extern const char *REDUCTION_Template;
    extern const char *REDUCTION_Template_Span;
    extern const char *D2H_MEM_MOV_1_Template;
    extern const char *D2H_MEM_MOV_2_Template;
    extern const char *MEM_FREE_Template;

    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);

    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

    std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size);
    //finish
    std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size);
    //finish
    std::string CodeGen_KernelExecute(std::string SplitSize,std::string IndexInit,std::string CalcEmbed);

    std::string CodeGen_Reduction(std::string SplitSize,std::string Name,std::string Type,std::string ReductionRule);

    std::string CodeGen_Reduction_Span(std::string ARRAY_SIZE,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule);

    std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction);

    std::string CodeGen_MemFree(std::string Name);
}

#endif