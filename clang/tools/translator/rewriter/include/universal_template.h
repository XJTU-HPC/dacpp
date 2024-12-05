#ifndef UNIVERSITY_TEMPLATE_H
#define UNIVERSITY_TEMPLATE_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"dacInfo.h"

namespace UNIVERSAL_TEMPLATE{

    extern const char *OP_REGULAR_SLICE_INIT_Template;
    extern const char *OP_INDEX_INIT_Template;
    extern const char *OP_PUSH_BACK_Template;
    extern const char *DATA_OPS_INIT_Template;
    extern const char *DATA_RECON_Template;
    extern const char *INDEX_INIT_Template;
    extern const char *CALC_EMBED_Template;

    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);
    
    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

    std::string CodeGen_RegularSliceInit(std::string opName,std::string size,std::string stride,std::string splitSize);

    std::string CodeGen_IndexInit(std::string opName,std::string splitSize);

    std::string CodeGen_OpPushBack(std::string name, std::string opName, std::string dimId, std::string splitLength);

    std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack);

    std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit);

    std::string CodeGen_IndexInit(Dac_Ops ops);

    std::string CodeGen_CalcEmbed(std::string Name,Args args);
}

#endif