#include "../dacppLib/include/Slice.h"
#include "../dacppLib/include/Tensor.hpp"

namespace dacpp {
typedef std::vector<std::any> list;
}

using dacpp::Tensor;





#include <sycl/sycl.hpp>
#include "DataReconstructor.cpp"

using namespace sycl;

void multiplication(int* mat1, int* mat2, int* mat3) 
{
}


// 生成函数调用
void split(const Tensor<int> & matA, const Tensor<int> & matB, Tensor<int> & matC) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    
    // 规则分区算子初始化
    RegularSlice s1 = RegularSlice("s1", 2, 2);
    s1.SetSplitSize(2);
    // 规则分区算子初始化
    RegularSlice s2 = RegularSlice("s2", 2, 2);
    s2.SetSplitSize(2);
    // 数据重组
    
    // 数据重组
    DataReconstructor<int> matA_tool;
    int* r_matA=(int*)malloc(sizeof(int)*16);
    
    // 数据算子组初始化
    Dac_Ops matA_ops;
    
    matA_tool.init(matA,matA_ops);
    matA_tool.Reconstruct(r_matA);
    // 数据重组
    DataReconstructor<int> matB_tool;
    int* r_matB=(int*)malloc(sizeof(int)*16);
    
    // 数据算子组初始化
    Dac_Ops matB_ops;
    
    matB_tool.init(matB,matB_ops);
    matB_tool.Reconstruct(r_matB);
    // 数据重组
    DataReconstructor<int> matC_tool;
    int* r_matC=(int*)malloc(sizeof(int)*16);
    
    // 数据算子组初始化
    Dac_Ops matC_ops;
    
    matC_tool.init(matC,matC_ops);
    matC_tool.Reconstruct(r_matC);
    // 设备内存分配
    
    // 设备内存分配
    int *d_matA=malloc_device<int>(16,q);
    // 设备内存分配
    int *d_matB=malloc_device<int>(16,q);
    // 设备内存分配
    int *d_matC=malloc_device<int>(16,q);
    // 数据移动
    
    // 数据移动
    q.memcpy(d_matA,r_matA,16*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_matB,r_matB,16*sizeof(int)).wait();   
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 4);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto s1=item_id/2%2;
            const auto s2=item_id%2;
            // 嵌入计算
			
            multiplication(d_matA+(s1*8),d_matB+(s2*21978),d_matC+(s1*8+s2*4));
        });
    }).wait();
    
    
    // 归约
    
    // 归约
    int *reduction_matC = malloc_device<int>(16,q); //存归约结果 归约结果存在长度为ARRAY_SIZE的数组中
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(1 * 16),
        reduction(span<int,16>(reduction_matC,16), 
        {{REDUCTION_RULE}},
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            	reducer[i % 4 + i/(4*1)*4].combine(d_matC[i]);
     	});
 }).wait();

    // 返回计算结果
    
    // 归并结果返回
    q.memcpy(r_matC, d_matC, 16*sizeof(int)).wait();
    // 内存释放
    
    sycl::free(d_matA, q);
    sycl::free(d_matB, q);
    sycl::free(d_matC, q);
}

int main() {
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeA{4, 4};
    Tensor<int> matA(dataA, shapeA);

    std::vector<int> dataB{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeB{4, 4};
    Tensor<int> matB(dataB, shapeB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeC{4, 4};
    Tensor<int> matC(dataC, shapeC);

    split(matA, matB, matC) <-> multiplication;

    return 0;
}