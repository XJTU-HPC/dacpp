#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "buffer_template.h"

namespace BUFFER_TEMPLATE {



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

std::string BUFFER_ACCESSOR_LIST = "";
std::string ACCESSOR_POINTER_LIST = "";
const char *BUFFER_ACCESSOR_Template = R"~~~(
        accessor acc_{{NAME}}{b_{{NAME}}, h};)~~~";
const char *ACCESSOR_POINTER_Template = R"~~~(
            auto* d_{{NAME}} = acc_{{NAME}}.get_multi_ptr<access::decorated::no>().get())~~~";

const char *DEVICE_MEM_ALLOC_Template = R"~~~(
    // Buffer设备内存分配
    buffer <{{TYPE}}> b_{{NAME}}{{{SIZE}}};)~~~";

std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size){
    BUFFER_ACCESSOR_LIST += templateString(BUFFER_ACCESSOR_Template,{
        {"{{NAME}}", name}
    });
    ACCESSOR_POINTER_LIST += templateString(ACCESSOR_POINTER_Template,{
        {"{{NAME}}", name}
    });
    return templateString(DEVICE_MEM_ALLOC_Template,{
		{"{{TYPE}}", type},
		{"{{NAME}}", name},
		{"{{SIZE}}", size}
	});
}

const char *H2D_MEM_MOV_Template = R"~~~(
    { //Buffer主机到设备传输数据
        host_accessor temp_accessor{b_{{NAME}}};
        for(int i = 0; i < {{SIZE}}; i++){
            temp_accessor[i] = r_{{NAME}}[i];
        }
    }
)~~~";

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
    {{ACCESSOR_LIST}}
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			{{INDEX_INIT}}
            // 获得accessor指针
            {{ACCESSOR_POINTER_LIST}}
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
		{"{{CALC_EMBED}}",    CalcEmbed},
        {"{{ACCESSOR_LIST}}",   BUFFER_ACCESSOR_LIST},
        {"{{ACCESSOR_POINTER_LIST}}",   ACCESSOR_POINTER_LIST}
	});
}

const char *REDUCTION_Template = R"~~~(
    // 归约
    buffer <{{TYPE}}> b_reduction_{{NAME}} {1};
    //使用内核函数进行归约
    q.submit([&](handler &h) {
        accessor d_{{NAME}}{b_{{NAME}}, h};
        h.parallel_for(range<1>({{SPLIT_SIZE}}),reduction(b_reduction_{{NAME}}, h, 0, {{REDUCTION_RULE}},property::reduction::initialize_to_identity()),[=](id<1> i,auto &reducer) {
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
        accessor d_{{NAME}}{b_{{NAME}}, h};
    	h.parallel_for(
            range<1>({{SPLIT_SIZE}} * {{ARRAY_SIZE}}),
            reduction(span<{{TYPE}},{{ARRAY_SIZE}}>(reduction_{{NAME}},{{ARRAY_SIZE}}), 
            {{REDUCTION_RULE}},
            property::reduction::initialize_to_identity()),
            [=](id<1> i,auto &reducer) {
            	reducer[i % {{SPLIT_LENGTH}} + i/({{SPLIT_LENGTH}}*{{SPLIT_SIZE}})*{{SPLIT_LENGTH}}].combine(d_{{NAME}}[i]);
     	});
    }).wait();
    buffer <{{TYPE}}> b_reduction_{{NAME}} {{{ARRAY_SIZE}}};
    {
        {{TYPE}}* temp_{{NAME}} = ({{TYPE}}*)malloc(sizeof({{TYPE}})*{{ARRAY_SIZE}});
        q.memcpy(temp_{{NAME}}, reduction_{{NAME}}, sizeof({{TYPE}})*{{ARRAY_SIZE}}).wait();
        host_accessor temp_accessor{b_reduction_{{NAME}}};
        for(int i = 0; i < ARRAY_SIZE; i++)
            temp_accessor[i] = temp_{{NAME}}[i];
        free(temp_{{NAME}});
    }
    free(reduction_{{NAME}}, q);
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
    {
        host_accessor temp_accessor{b_{{NAME}}};
        for(int i = 0; i < {{SIZE}}; i++){
            r_{{NAME}}[i] = temp_accessor[i];
        }
    }
)~~~";

const char *D2H_MEM_MOV_2_Template = R"~~~(
    // 归约结果返回
    {
        host_accessor temp_accessor{b_reduction_{{NAME}}};
        for(int i = 0; i < {{SIZE}}; i++){
            r_{{NAME}}[i] = temp_accessor[i];
        }
    }
)~~~";

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

}