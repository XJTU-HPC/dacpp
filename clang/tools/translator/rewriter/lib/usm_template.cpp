#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "usm_template.h"
#include <string>
#include <vector>
#include <set>
#include <unordered_map>

namespace USM_TEMPLATE {



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

const char *DEVICE_MEM_ALLOC_Template = R"~~~(
    // 设备内存分配
    {{TYPE}} *d_{{NAME}}=malloc_device<{{TYPE}}>({{SIZE}},q);)~~~";

std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size){
    return templateString(DEVICE_MEM_ALLOC_Template,{
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

const char *REDUCTION_Template = R"~~~(
    // 归约
    {{TYPE}} *reduction_{{NAME}} = malloc_device<{{TYPE}}>(1,q);
    //使用内核函数进行归约
    q.submit([&](handler &h) {
    	h.parallel_for(range<1>({{SPLIT_SIZE}}),reduction(reduction_{{NAME}}, {{REDUCTION_RULE}},property::reduction::initialize_to_identity()),[=](id<1> i,auto &reducer) {
            	reducer.combine(d_{{NAME}}[i]);
     	});
 }).wait();
)~~~";

std::string CodeGen_Reduction(std::string SplitSize,std::string Name,std::string Type,std::string ReductionRule) {
    return templateString(REDUCTION_Template,
	{
		{"{{SPLIT_SIZE}}",       SplitSize},
		{"{{TYPE}}",             Type},
		{"{{NAME}}",             Name},
		{"{{REDUCTION_RULE}}",    ReductionRule}
	});
}

const char *REDUCTION_Template_Span = R"~~~(
    // 归约
    {{TYPE}} *reduction_{{NAME}} = malloc_device<{{TYPE}}>({{ARRAY_SIZE}},q); //存归约结果 归约结果存在长度为ARRAY_SIZE的数组中
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>({{SPLIT_SIZE}} * {{ARRAY_SIZE}}),
        reduction(span<{{TYPE}},{{ARRAY_SIZE}}>(reduction_{{NAME}},{{ARRAY_SIZE}}), 
        {{REDUCTION_RULE}},
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            	reducer[i % {{SPLIT_LENGTH}} + i/({{SPLIT_LENGTH}}*{{SPLIT_SIZE}})*{{SPLIT_LENGTH}}].combine(d_{{NAME}}[i]);
     	});
 }).wait();
)~~~";

std::string CodeGen_Reduction_Span(std::string ARRAY_SIZE,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule) {
    return templateString(REDUCTION_Template_Span,
	{
        {"{{ARRAY_SIZE}}",       ARRAY_SIZE},   
		{"{{SPLIT_SIZE}}",       SplitSize},
		{"{{SPLIT_LENGTH}}",     SplitLength},
		{"{{TYPE}}",             Type},
		{"{{NAME}}",             Name},
		{"{{REDUCTION_RULE}}",    ReductionRule}
	});
}

const char *D2H_MEM_MOV_1_Template = R"~~~(
    // 归并结果返回
    q.memcpy(r_{{NAME}}, d_{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();)~~~";

const char *D2H_MEM_MOV_2_Template = R"~~~(
    // 归约结果返回
    q.memcpy(r_{{NAME}},reduction_{{NAME}}, {{SIZE}}*sizeof({{TYPE}})).wait();)~~~";

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


}