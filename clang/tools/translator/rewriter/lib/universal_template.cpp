#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"universal_template.h"

namespace UNIVERSAL_TEMPLATE{

void replaceTextInString(std::string& text, 
    const std::string &find, 
    const std::string &replace){
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos){
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}
std::string templateString(std::string templ, 
    std::vector<std::pair<std::string, std::string>> replacements){
	for(auto &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}

const char *OP_REGULAR_SLICE_INIT_Template = R"~~~(
    // 规则分区算子初始化
    RegularSlice {{OP_NAME}} = RegularSlice("{{OP_NAME}}", {{SIZE}}, {{STRIDE}});
    {{OP_NAME}}.SetSplitSize({{SPLIT_SIZE}});)~~~";

std::string CodeGen_RegularSliceInit(std::string opName,std::string size,std::string stride,std::string splitSize){
    return templateString(OP_REGULAR_SLICE_INIT_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{SIZE}}",       size},
		{"{{STRIDE}}",     stride},
		{"{{SPLIT_SIZE}}", splitSize}
	});
}

const char *OP_INDEX_INIT_Template = R"~~~(
    // 降维算子初始化
    Index {{OP_NAME}} = Index("{{OP_NAME}}");
    {{OP_NAME}}.SetSplitSize({{SPLIT_SIZE}});)~~~";

std::string CodeGen_IndexInit(std::string opName,std::string splitSize){
    return templateString(OP_INDEX_INIT_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{SPLIT_SIZE}}", splitSize}
	});
}

const char *OP_PUSH_BACK_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.setSplitLength({{SPLIT_LENGTH}});
    {{NAME}}_ops.push_back({{OP_NAME}});)~~~";

std::string CodeGen_OpPushBack(std::string name, std::string opName, std::string dimId, std::string splitLength){
    return templateString(OP_PUSH_BACK_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{NAME}}",       name},
		{"{{DIM_ID}}",     dimId},
		{"{{SPLIT_LENGTH}}", splitLength}
	});
}

const char *DATA_OPS_INIT_Template = R"~~~(
    // 数据算子组初始化
    Dac_Ops {{NAME}}_ops;
    {{OP_PUSH_BACK}})~~~";

std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack){
    return templateString(DATA_OPS_INIT_Template,
	{
		{"{{NAME}}",       name},
		{"{{OP_PUSH_BACK}}",    opPushBack},
	});
}

const char *DATA_RECON_Template = R"~~~(
    // 数据重组
    DataReconstructor<{{TYPE}}> {{NAME}}_tool;
    {{TYPE}}* r_{{NAME}}=({{TYPE}}*)malloc(sizeof({{TYPE}})*{{SIZE}});
    {{DATA_OPS_INIT}}
    {{NAME}}_tool.init({{NAME}},{{NAME}}_ops);
    {{NAME}}_tool.Reconstruct(r_{{NAME}});)~~~";

std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit){
    return templateString(DATA_RECON_Template,
	{
		{"{{TYPE}}",       type},
		{"{{NAME}}",       name},
		{"{{SIZE}}",       size},
		{"{{DATA_OPS_INIT}}", dataOpsInit}
	});
}

const char *INDEX_INIT_Template = R"~~~(
            const auto {{NAME}}={{EXPRESSION}};)~~~";

std::string CodeGen_IndexInit(Dac_Ops ops){
	int len = ops.size;
	for(int i=0;i<len;i++){
		std::string sub_expression = "item_id";
		for(int j=i+1;j<len;j++){
			sub_expression = sub_expression + "/" + std::to_string(ops[j].split_size);
		}
		sub_expression = sub_expression + "%" + std::to_string(ops[i].split_size);
		ops[i].setExp(sub_expression);
	}

	std::string expression = "";
	for(int i=0;i<len;i++){
		expression = expression + templateString(INDEX_INIT_Template,
		{
			{"{{NAME}}", ops[i].name},
			{"{{EXPRESSION}}", ops[i].getExp()}
		});
	}

	return expression;
}
const char *CALC_EMBED_Template = R"~~~(
            {{DAC_CALC_NAME}}{{DAC_CALC_ARGS}})~~~";

std::string CodeGen_CalcEmbed(std::string Name,Args args){
	std::string DacCalcArgs = "(";
	int len = args.size;
	for(int i=0;i<len;i++){
		std::string IndexComb="(";
		for(int j=0;j<args[i].ops.size;j++){
			IndexComb+= args[i].ops[j].name + "*" + std::to_string(args[i].ops[j].split_length);
			if(j!=args[i].ops.size-1) IndexComb+="+";
		}
		IndexComb+=")";
		DacCalcArgs+=args[i].name + "+" + IndexComb;
		if(i==len-1){
			DacCalcArgs+=");";
		}
		else{
			DacCalcArgs+=",";
		}
	}
	return templateString(CALC_EMBED_Template,
	{
		{"{{DAC_CALC_NAME}}",    Name},
		{"{{DAC_CALC_ARGS}}",    DacCalcArgs}
	});
}

}