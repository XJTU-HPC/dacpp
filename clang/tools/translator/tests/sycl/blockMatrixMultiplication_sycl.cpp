#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include <iostream>

using namespace sycl;
using dacpp::Tensor;

void multiplication(int* mat1, int* mat2, int* mat3) {
    for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
            mat3[i * 2 + j] += mat1[i * 2 + k] * mat2[k * 2 + j];
        }
    }
}

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
    // 规则分区算子初始化
    RegularSlice s3 = RegularSlice("s3", 2, 2);
    s3.SetSplitSize(2);
    // 数据重组
    
    // 数据重组
    DataReconstructor<int> matA_tool;
    int* r_matA=(int*)malloc(sizeof(int)*16);
    
    // 数据算子组初始化
    Dac_Ops matA_ops;
    
    s1.setDimId(0);
    s1.setSplitLength(8);
    matA_ops.push_back(s1);
    s3.setDimId(1);
    s3.setSplitLength(4);
    matA_ops.push_back(s3);
    matA_tool.init(matA,matA_ops);
    matA_tool.Reconstruct(r_matA);
    // 数据重组
    DataReconstructor<int> matB_tool;
    int* r_matB=(int*)malloc(sizeof(int)*16);
    
    // 数据算子组初始化
    Dac_Ops matB_ops;
    
    s3.setDimId(0);
    s3.setSplitLength(8);
    matB_ops.push_back(s3);
    s2.setDimId(1);
    s2.setSplitLength(4);
    matB_ops.push_back(s2);
    matB_tool.init(matB,matB_ops);
    matB_tool.Reconstruct(r_matB);
    // 数据重组
    DataReconstructor<int> matC_tool;
    int* r_matC=(int*)malloc(sizeof(int)*16);
    
    // 数据算子组初始化
    Dac_Ops matC_ops;
    
    s1.setDimId(0);
    s1.setSplitLength(8);
    matC_ops.push_back(s1);
    s2.setDimId(1);
    s2.setSplitLength(4);
    matC_ops.push_back(s2);
    matC_tool.init(matC,matC_ops);
    matC_tool.Reconstruct(r_matC);
    // 设备内存分配
    
    // 设备内存分配
    int *d_matA=malloc_device<int>(16,q);
    // 设备内存分配
    int *d_matB=malloc_device<int>(16,q);
    // 设备内存分配
    int *d_matC=malloc_device<int>(32,q);
    // 数据移动
    
    // 数据移动
    q.memcpy(d_matA,r_matA,16*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_matB,r_matB,16*sizeof(int)).wait();   
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 8);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto s1=item_id/2/2%2;
            const auto s2=item_id/2%2;
            const auto s3=item_id%2;
            // 嵌入计算
			
            multiplication(d_matA+(s1*8+s3*4),d_matB+(s3*8+s2*4),d_matC+(s1*16+s2*8+s3*4));
        });
    }).wait();
    
    std::cout <<  "归约前计算结果:\n";
    q.memcpy(r_matC, d_matC, 32*sizeof(int)).wait();
    for(int i=0;i<32;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<std::endl;

    // 归约
    
    // 归约
    int *reduction_matC = malloc_device<int>(16,q); //存归约结果 归约结果存在长度为ARRAY_SIZE的数组中
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(2 * 16),
        reduction(span<int,16>(reduction_matC,16), 
        sycl::plus<>(),
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            	reducer[i % 4 + i/(4*2)*4].combine(d_matC[i]);
     	});
 }).wait();

    // 返回计算结果
   int *res = malloc_shared<int>(16,q);
    q.memcpy(res,reduction_matC, 16*sizeof(int)).wait();
    std::cout <<  "归约后计算结果(未归并):\n";
    for(int i=0;i<16;i++) {
        std::cout << res[i] << " ";
        if(i%4==3) std::cout<<"\n";
    }
    
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

    split(matA, matB, matC);

    return 0;
}