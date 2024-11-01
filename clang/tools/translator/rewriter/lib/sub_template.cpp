#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"sub_template.h"

/*
	文本替换。
	将 text 中的 find 换成 replace。
*/
void replaceTextInString(std::string& text, const std::string &find, const std::string &replace){
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
std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements){
	for(std::pair<std::string, std::string> &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}

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
		// for(int j=0;j<args[i].ops.size;j++){
		// 	IndexComb+= args[i].ops[j].name+"*";
		// }
		for(int j=0;j<args[i].ops.size;j++){
			IndexComb+= args[i].ops[j].name + "*" + std::to_string(args[i].ops[j].split_length);
			// for(int k=j+1;k<args[i].ops.size;k++){
			// 	IndexComb+="*"+std::to_string(args[i].getDimlength(args[i].ops[k].dimId));
			// }
			// 
			if(j==args[i].ops.size-1) IndexComb+=")";
			else IndexComb+="+";
		}
		// DacCalcArgs+=args[i].name + "+" + IndexComb + std::to_string(args[i].split_length);
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

const char *REDUCTION_Template = R"~~~(
    // 归约
    {{TYPE}} *reduction_{{NAME}} = malloc_device<{{TYPE}}>(1,q);
    q.submit([&](handler &h) {
    	h.parallel_for(range<1>({{SPLIT_SIZE}}),reduction(reduction_{{NAME}}, {{REDUCTION_RULE}}),[=](id<1> i,auto &reducer) {
            	reducer.combine(d_{{NAME}}[i]);
     	});
 }).wait();
)~~~";

std::string CodeGen_Reduction(std::string SplitSize,std::string Name,std::string Type,std::string ReductionRule){
    return templateString(REDUCTION_Template,
	{
		{"{{SPLIT_SIZE}}",      SplitSize},
		{"{{TYPE}}",            Type},
		{"{{NAME}}",            Name},
		{"{{ReductionRule}}",   ReductionRule}
	});
}

const char *D2H_MEM_MOV_1_Template = R"~~~(
    // 归并结果返回
    q.memcpy(r_{{NAME}}, d_{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();)~~~";

const char *D2H_MEM_MOV_2_Template = R"~~~(
    // 归约结果返回
    q.memcpy(r_{{NAME}},reduction_{{NAME}}, sizeof({{TYPE}})).wait();)~~~";

std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction){
    if(isReduction){
		return templateString(D2H_MEM_MOV_2_Template,
		{
			{"{{TYPE}}",            Type},
			{"{{NAME}}",            Name}
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