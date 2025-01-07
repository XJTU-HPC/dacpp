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

std::string CodeGen_IndexInit(Dac_Ops ops,std::vector<std::string> sets,std::vector<std::string> offsets)//sets表示每个算子属于的集合的名字 offsets表示每个算子相对于集合的偏移量
{ 
    std::set<std::string> sets_map;//用于辅助找到不同的集合的个数
    std::vector<std::string> sets_order;//记录了不同的集合出现的顺序，储存集合的名字： idx idy idz
    std::vector<int> sets_split;//记录了不同集合对应的划分数，与集合名相对应： idx的划分数 idy的划分数 idz的划分数 
    for (int i = 0; i < sets.size(); ++i) 
    {
        if (sets_map.find(sets[i]) == sets_map.end())//如果容器里没有
        {
            sets_map.insert(sets[i]);//将集合插入容器
            sets_order.push_back(sets[i]);//将集合放入到集合的数组中
            sets_split.push_back(ops[i].split_size);//将集合对应的划分数放入数组中
        }
    }
    
    int sets_size = sets_map.size();//得到各类集合总个数
    std::unordered_map<std::string,std::string> sets_sub_expression;//<集合的名称，集合对应的索引表达式>

    for(int i = 0;i < sets_size; i++)//有几个集合就循环几次
    {
		std::string sub_expression = "item_id";
		for(int j = i + 1;j < sets_size;j ++){
			sub_expression = sub_expression + "/" + std::to_string(sets_split[j]);
		}
		//sub_expression = sub_expression + "%" + std::to_string(sets_split[i]);//取模操作应该在偏移之后
        sets_sub_expression[sets_order[i]] = sub_expression;//将子表达式和集合的名字进行关联
	}

    //下面根据偏移量来计算各个算子对应的索引
    int len = ops.size;
    for(int i = 0;i < len;i ++)
    {
        std::string index_expression = "(";
        index_expression = index_expression + sets_sub_expression[sets[i]];//得到集合的索引
        index_expression = index_expression + "+" + "(" + offsets[i] + ")" + "+" + std::to_string(ops[i].split_size) + ")";//加上偏移量和划分数 防止出现负数
		index_expression = index_expression + "%" + std::to_string(ops[i].split_size);//偏移之后再取模
        ops[i].setExp(index_expression);
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
	std::vector<std::string> index_expression_vector;
	for(int i=0;i<len;i++){
		std::string sub_expression = "item_id";
		for(int j=i+1;j<len;j++){
			sub_expression = sub_expression + "/" + std::to_string(ops[j].split_size);
		}
		sub_expression = sub_expression + "%" + std::to_string(ops[i].split_size);
		//ops[i].setExp(sub_expression);
		index_expression_vector.push_back(sub_expression);
	}

	std::string expression = "";
	for(int i=0;i<len;i++){
		std::string opsname = ops[i].name;
		std::string index_i_expression = index_expression_vector[i];		
		expression = expression + templateString(INDEX_INIT_Template,
		{
			{"{{NAME}}", opsname + "_"},
			{"{{EXPRESSION}}", index_i_expression}
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
			std::string opsname = args[i].ops[j].name;
			//IndexComb+= args[i].ops[j].name + "*" + std::to_string(args[i].ops[j].split_length);
			IndexComb+= opsname + "*" + std::to_string(args[i].ops[j].split_length);
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