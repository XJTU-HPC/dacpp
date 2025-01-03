#include <vector>

#include "/data/zjx/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"

using dacpp::Tensor;

namespace dacpp {
typedef std::vector<std::any> list;
}





#include <sycl/sycl.hpp>
#include "/data/qinian/ice/dacpp/clang/tools/translator/dpcppLib/include/DataReconstructor.old.h"

using namespace sycl;

void matMul(int* vecA, int* vecB, int* dotProduct) 
{
    for (int i = 0; i < 5; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}


// 生成函数调用
void matMulSplit(const Tensor<int> & matA, const Tensor<int> & matB, Tensor<int> & matC) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    
    // 降维算子初始化
    Index i = Index("i");
    i.SetSplitSize(4);
    // 降维算子初始化
    Index j = Index("j");
    j.SetSplitSize(4);
    // 数据重组
    
    // 数据重组
    DataReconstructor<int> matA_tool;
    int* r_matA=(int*)malloc(sizeof(int)*20);
    
    // 数据算子组初始化
    Dac_Ops matA_ops;
    
    i.setDimId(0);
    i.setSplitLength(5);
    matA_ops.push_back(i);
    matA_tool.init(matA,matA_ops);
    matA_tool.Reconstruct(r_matA);
    // 数据重组
    DataReconstructor<int> matB_tool;
    int* r_matB=(int*)malloc(sizeof(int)*20);
    
    // 数据算子组初始化
    Dac_Ops matB_ops;
    
    j.setDimId(1);
    j.setSplitLength(5);
    matB_ops.push_back(j);
    matB_tool.init(matB,matB_ops);
    matB_tool.Reconstruct(r_matB);
    // 数据重组
    DataReconstructor<int> matC_tool;
    int* r_matC=(int*)malloc(sizeof(int)*16);
    
    // 数据算子组初始化
    Dac_Ops matC_ops;
    
    i.setDimId(0);
    i.setSplitLength(4);
    matC_ops.push_back(i);
    j.setDimId(1);
    j.setSplitLength(1);
    matC_ops.push_back(j);
    matC_tool.init(matC,matC_ops);
    matC_tool.Reconstruct(r_matC);
    // 设备内存分配
    
    // 设备内存分配
    int *d_matA=malloc_device<int>(20,q);
    // 设备内存分配
    int *d_matB=malloc_device<int>(20,q);
    // 设备内存分配
    int *d_matC=malloc_device<int>(16,q);
    
    // for(int i=0;i<20;i++) std::cout<<r_matA[i]<<" ";
    // std::cout<<std::endl;

    // for(int i=0;i<20;i++) std::cout<<r_matB[i]<<" ";
    // std::cout<<std::endl;
    
    // 数据移动
    
    // 数据移动
    q.memcpy(d_matA,r_matA,20*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_matB,r_matB,20*sizeof(int)).wait();   
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 16);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i=(item_id/4+(0)+4)%4;
            const auto j=(item_id+(0)+4)%4;
            // 嵌入计算
			
            matMul(d_matA+(i*5),d_matB+(j*5),d_matC+(i*4+j*1));
        });
    }).wait();
    
    
    // 归约
    
    // 返回计算结果
    
    // 归并结果返回
    q.memcpy(r_matC, d_matC, 16*sizeof(int)).wait();
    // for(int i=0;i<16;i++) std::cout<<r_matC[i]<<" ";
    // std::cout<<std::endl;
    matC = matC_tool.UpdateData(r_matC);
    // 内存释放
    
    sycl::free(d_matA, q);
    sycl::free(d_matB, q);
    sycl::free(d_matC, q);
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now(); // 开始时间测量

    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> shapeA{4, 5};
    Tensor<int> matA(dataA, shapeA);

    std::vector<int> dataB{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> shapeB{5, 4};
    Tensor<int> matB(dataB, shapeB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeC{4, 4};
    Tensor<int> matC(dataC, shapeC);

    matMulSplit(matA, matB, matC);
    matC.print();
    auto end_time = std::chrono::high_resolution_clock::now(); // 结束时间测量
    std::chrono::duration<double> duration = end_time - start_time; // 计算持续时间
    std::cout << "总执行时间: " << duration.count() << " 秒" << std::endl; // 输出执行时间
    return 0;
}