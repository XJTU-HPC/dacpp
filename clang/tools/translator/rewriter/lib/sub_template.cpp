#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"sub_template.h"

/*
	文本替换。
	将 text 中的 find 换成 replace。
*/
void replaceTextInString(std::string& text, 
const std::string &find, 
const std::string &replace){
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos){
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}
/*
	文本替换。
	对 templ 做 replacements 中的文本替换。
*/
std::string templateString(std::string templ, 
std::vector<std::pair<std::string, std::string>> replacements){
	for(auto &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}

const char *DATA_ASSOC_COMP_Template = R"~~~(
    {{DATA_RECON}}
    {{H2D_MEM_MOV}}
	{{KERNEL_EXECUTE}}
	{{REDUCTION}}
	{{D2H_MEM_MOV}}
)~~~";

std::string CodeGen_DataAssocComp(std::string dataRecon, std::string H2DMemMove, std::string kernelExecute, std::string reduction, std::string D2HMemMove){
    return templateString(DATA_ASSOC_COMP_Template,
	{
        {"{{DATA_RECON}}",        dataRecon},
        {"{{H2D_MEM_MOV}}",       H2DMemMove},
        {"{{KERNEL_EXECUTE}}",    kernelExecute},
		{"{{REDUCTION}}",         reduction},
        {"{{D2H_MEM_MOV}}",       D2HMemMove}
	});
}

const char *DAC2SYCL_Template_1 = R"~~~(
// 生成函数调用
void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 设备内存分配
    {{DEVICE_MEM_ALLOC}}
    // 算子初始化
    {{OP_INIT}}
    // 数据关联计算
    {{DATA_ASSOC_COMP}}
    // 内存释放
    {{MEM_FREE}}
})~~~";

std::string CodeGen_DAC2SYCL(std::string dacShellName, std::string dacShellParams, std::string deviceMemAlloc, std::string opInit, std::string dataAssocComp, std::string memFree){
    return templateString(DAC2SYCL_Template_1,
	{
		{"{{DAC_SHELL_NAME}}",    dacShellName},
		{"{{DAC_SHELL_PARAMS}}",  dacShellParams},
		{"{{OP_INIT}}",           opInit},
		{"{{DEVICE_MEM_ALLOC}}",  deviceMemAlloc},
		{"{{DATA_ASSOC_COMP}}",   dataAssocComp},
        {"{{MEM_FREE}}",          memFree}
	});
}

// aborted
const char *DAC2SYCL_Template = R"~~~(
// 生成函数调用
void {{DAC_SHELL_NAME}}({{DAC_SHELL_PARAMS}}) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    {{OP_INIT}}
    // 数据重组
    {{DATA_RECON}}
    // 设备内存分配
    {{DEVICE_MEM_ALLOC}}
    // 数据移动
    {{H2D_MEM_MOV}}   
    // 内核执行
    {{KERNEL_EXECUTE}}    
    // 归约
    {{REDUCTION}}
    // 返回计算结果
    {{D2H_MEM_MOV}}
    // 内存释放
    {{MEM_FREE}}
})~~~";

// aborted
std::string CodeGen_DAC2SYCL(std::string dacShellName,std::string dacShellParams,std::string opInit,std::string dataRecon,
	std::string deviceMemAlloc,std::string H2DMemMove,std::string kernelExecute,std::string reduction,std::string D2HMemMove,std::string memFree){
    return templateString(DAC2SYCL_Template,
	{
		{"{{DAC_SHELL_NAME}}",    dacShellName},
		{"{{DAC_SHELL_PARAMS}}",  dacShellParams},
		{"{{OP_INIT}}",           opInit},
        {"{{DATA_RECON}}",        dataRecon},
		{"{{DEVICE_MEM_ALLOC}}",  deviceMemAlloc},
        {"{{H2D_MEM_MOV}}",       H2DMemMove},
        {"{{KERNEL_EXECUTE}}",    kernelExecute},
		{"{{REDUCTION}}",         reduction},
        {"{{D2H_MEM_MOV}}",       D2HMemMove},
        {"{{MEM_FREE}}",          memFree}
	});
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

const char *OP_PUSH_BACK2OPS_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.setSplitLength({{SPLIT_LENGTH}});
    {{NAME}}_ops.push_back({{OP_NAME}});)~~~";

std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId, std::string splitLength){
    return templateString(OP_PUSH_BACK2OPS_Template,
	{
		{"{{OP_NAME}}",    opName},
		{"{{NAME}}",       name},
		{"{{DIM_ID}}",     dimId},
		{"{{SPLIT_LENGTH}}", splitLength}
	});
}

const char *OP_PUSH_BACK2TOOL_Template = R"~~~(
    {{OP_NAME}}.setDimId({{DIM_ID}});
    {{OP_NAME}}.setSplitLength({{SPLIT_LENGTH}});
    {{NAME}}_tool.push_back({{OP_NAME}});)~~~";

std::string CodeGen_OpPushBack2Tool(std::string name, std::string opName, std::string dimId, std::string splitLength){
    return templateString(OP_PUSH_BACK2TOOL_Template,
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
    {{OP_PUSH_BACK2OPS}})~~~";

std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack2Ops){
    return templateString(DATA_OPS_INIT_Template,
	{
		{"{{NAME}}",       name},
		{"{{OP_PUSH_BACK2OPS}}",    opPushBack2Ops},
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

const char *DATA_RECON_OP_PUSH_Template = R"~~~(
    // 数据重组
    {{OP_PUSH_BACK2TOOL}}
    {{NAME}}_tool.Reconstruct(r_{{NAME}});)~~~";

std::string CodeGen_DataReconstructOpPush(std::string name,std::string opPushBack2Tool){
    return templateString(DATA_RECON_OP_PUSH_Template,
	{
		{"{{NAME}}",       name},
		{"{{OP_PUSH_BACK2TOOL}}", opPushBack2Tool}
	});
}

const char *OP_POP_FROM_TOOL_Template = R"~~~(
    {{NAME}}_tool.pop_back();)~~~";
std::string CodeGen_OpPopFromTool(std::string name){
    return templateString(OP_POP_FROM_TOOL_Template,
	{
		{"{{NAME}}", name}
	});
}

const char *DATA_RECON_OP_POP_Template = R"~~~(
    // 数据重组
    {{OP_POP_FROM_TOOL}}
    {{NAME}}_tool.Reconstruct(r_{{NAME}});)~~~";

std::string CodeGen_DataReconstructOpPop(std::string name,std::string opPopFromTool){
    return templateString(DATA_RECON_OP_POP_Template,
	{
		{"{{NAME}}",       name},
		{"{{OP_POP_FROM_TOOL}}", opPopFromTool}
	});
}

const char *DEVICE_MEM_ALLOC_Template = R"~~~(
    // 设备内存分配
    {{TYPE}} *d_{{NAME}}=malloc_device<{{TYPE}}>({{SIZE}},q);)~~~";

std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size){
    return templateString(DEVICE_MEM_ALLOC_Template,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

const char *DEVICE_MEM_ALLOC_REDUCTION_Template = R"~~~(
    // 归约设备内存分配
    {{TYPE}} *reduction_{{NAME}} = malloc_device<{{TYPE}}>({{SIZE}},q);)~~~";

std::string CodeGen_DeviceMemAllocReduction(std::string  type,std::string name,std::string size){
	return templateString(DEVICE_MEM_ALLOC_REDUCTION_Template,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

const char *H2D_MEM_MOV_Template = R"~~~(
    // 数据移动
    q.memcpy(d_{{NAME}},r_{{NAME}},{{SIZE}}*sizeof({{TYPE}})).wait();)~~~";

std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size){
    return templateString(H2D_MEM_MOV_Template,
	{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

const char *KERNEL_EXECUTE_Template = R"~~~(
    //工作项划分
    sycl::range<3> local(1, 1, {{SPLIT_SIZE}});
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			{{INDEX_INIT}}
            // 嵌入计算
			{{CALC_EMBED}}
        });
    }).wait();
    
)~~~";

std::string CodeGen_KernelExecute(std::string SplitSize,std::string IndexInit,std::string CalcEmbed){
    return templateString(KERNEL_EXECUTE_Template,
	{
		{"{{SPLIT_SIZE}}",    SplitSize},
		{"{{INDEX_INIT}}",    IndexInit},
		{"{{CALC_EMBED}}",    CalcEmbed}
	});
}

const char *INDEX_INIT_Template = R"~~~(
            const auto {{NAME}}={{EXPRESSION}};)~~~";

//aborted
std::string CodeGen_IndexInit(Dac_Ops ops)
{
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
		if(IndexComb == "()")
		{
			DacCalcArgs+=args[i].name;
		}
		else{
			DacCalcArgs+=args[i].name + "+" + IndexComb;
		}		
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

// aborted
const char *REDUCTION_Template = R"~~~(
    // 归约
    //使用内核函数进行归约
    q.submit([&](handler &h) {
    	h.parallel_for(range<1>({{SPLIT_SIZE}}),reduction(reduction_{{NAME}}, {{REDUCTION_RULE}},property::reduction::initialize_to_identity()),[=](id<1> i,auto &reducer) {
            	reducer.combine(d_{{NAME}}[i]);
     	});
 }).wait();
)~~~";

// aborted
std::string CodeGen_Reduction(std::string SplitSize,std::string Name,std::string Type,std::string ReductionRule) {
    return templateString(REDUCTION_Template,
	{
		{"{{SPLIT_SIZE}}",       SplitSize},
		{"{{TYPE}}",             Type},
		{"{{NAME}}",             Name},
		{"{{REDUCTION_RULE}}",   ReductionRule}
	});
}

const char *REDUCTION_Template_Span = R"~~~(
    // 归约
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>({{SPLIT_SIZE}} * {{SPAN_SIZE}}),
        reduction(span<{{TYPE}},{{SPAN_SIZE}}>(reduction_{{NAME}},{{SPAN_SIZE}}), 
        {{REDUCTION_RULE}},
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            reducer[i % {{SPLIT_LENGTH}} + i/({{SPLIT_LENGTH}}*{{SPLIT_SIZE}})*{{SPLIT_LENGTH}}].combine(d_{{NAME}}[i]);
     	});
 }).wait();
    q.memcpy(d_{{NAME}},reduction_{{NAME}}, {{SPAN_SIZE}}*sizeof({{TYPE}})).wait();
)~~~";

std::string CodeGen_Reduction_Span(std::string SpanSize,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule) {
    return templateString(REDUCTION_Template_Span,
	{
        {"{{SPAN_SIZE}}",        SpanSize},   
		{"{{SPLIT_SIZE}}",       SplitSize},
		{"{{SPLIT_LENGTH}}",     SplitLength},
		{"{{TYPE}}",             Type},
		{"{{NAME}}",             Name},
		{"{{REDUCTION_RULE}}",   ReductionRule}
	});
}

const char *D2H_MEM_MOV_1_Template = R"~~~(
    // 归并结果返回
    q.memcpy(r_{{NAME}}, d_{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();
    {{NAME}} = {{NAME}}_tool.UpdateData(r_{{NAME}});)~~~";

const char *D2H_MEM_MOV_2_Template = R"~~~(
    // 归约结果返回
    q.memcpy(r_{{NAME}},d__{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();
    {{NAME}} = {{NAME}}_tool.UpdateData(r_{{NAME}});)~~~";

std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction){
    if(isReduction){
		return templateString(D2H_MEM_MOV_2_Template,
		{
			{"{{TYPE}}",            Type},
			{"{{NAME}}",            Name},
			{"{{SIZE}}",            Size}
		});
	}
	else{
		return templateString(D2H_MEM_MOV_1_Template,
		{
			{"{{TYPE}}",            Type},
			{"{{NAME}}",            Name},
			{"{{SIZE}}",            Size}
		});
	}
}

const char *MEM_FREE_Template = R"~~~(
    sycl::free(d_{{NAME}}, q);)~~~";

std::string CodeGen_MemFree(std::string Name){
    return templateString(MEM_FREE_Template,
	{
		{"{{NAME}}",            Name}
	});
}

// int main(){
// 	std::cout<<"******************dac2sycl CodeGen test******************\n\n";
// 	Dac_Op i = Dac_Op("i",3,0);
// 	Dac_Op j = Dac_Op("j",3,0);
	
// 	Dac_Ops ops;
// 	ops.push_back(i);
// 	ops.push_back(j);
// 	std::string IndexInit = CodeGen_IndexInit(ops);

// 	i.setSplitLength(1);
// 	Dac_Ops vecA_ops;
// 	vecA_ops.push_back(i);
// 	DacData d_vecA = DacData("d_vecA",1,vecA_ops);
// 	d_vecA.setDimLength(0,3);

// 	j.setSplitLength(1);
// 	Dac_Ops vecB_ops;
// 	vecB_ops.push_back(j);
// 	DacData d_vecB = DacData("d_vecB",1,vecB_ops);
// 	d_vecB.setDimLength(0,3);

// 	Dac_Ops dotProduct_ops;
// 	i.setSplitLength(3);
// 	j.setSplitLength(3);
// 	j.setDimId(1);
// 	dotProduct_ops.push_back(i);
// 	dotProduct_ops.push_back(j);
// 	DacData d_dotProduct = DacData("d_dotProduct",2,dotProduct_ops);
// 	d_dotProduct.setDimLength(0,3);
// 	d_dotProduct.setDimLength(1,3);

// 	Args args = Args();
// 	args.push_back(d_vecA);
// 	args.push_back(d_vecB);
// 	args.push_back(d_dotProduct);
// 	std::string CalcEmbed = CodeGen_CalcEmbed("mat_mul",args);

// 	std::string KernelExecute = CodeGen_KernelExecute("9",IndexInit,CalcEmbed);

// 	std::cout<<KernelExecute;
// 	return 0;
// }