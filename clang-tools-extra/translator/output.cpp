#include <vector>

#include "../dacppLib/Slice.h"
#include "../dacppLib/Tensor.hpp"

using dacpp::Tensor;

namespace dacpp {

typedef std::vector<std::any> list;

}





#include <sycl/sycl.hpp>
#include "../DataReconstructor.cpp"

using namespace dacpp;
using namespace sycl;

void matMul(int* vecA, int* vecB, int* dotProduct)
{
    for (int i = 0; i < 5; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}

void matMulSplit(const Tensor<int> & matA, const Tensor<int> & matB, Tensor<int> & matC) {
    auto selector = gpu_selector_v;
   queue q(selector);

  DataReconstructor<int> tool;

    // 数据重组
	int* r_matA=(int*)malloc(sizeof(int)*matA.getSize());
	std::vector<bool> isIndex_matA{true, false};
	tool.init(matA,isIndex_matA);
	tool.Reconstruct(r_matA);
    // 数据重组
	int* r_matB=(int*)malloc(sizeof(int)*matB.getSize());
	std::vector<bool> isIndex_matB{false, true};
	tool.init(matB,isIndex_matB);
	tool.Reconstruct(r_matB);
    // 数据重组
	int* r_matC=(int*)malloc(sizeof(int)*matC.getSize());
	std::vector<bool> isIndex_matC{true, true};
	tool.init(matC,isIndex_matC);
	tool.Reconstruct(r_matC);
    // 设备内存分配
    int *d_matA=malloc_device<int>(matA.getSize(),q);
    // 设备内存分配
    int *d_matB=malloc_device<int>(matB.getSize(),q);
    // 设备内存分配
    int *d_matC=malloc_device<int>(matC.getSize(),q);

    // 数据移动
    q.memcpy(d_matA,r_matA,matA.getSize()*sizeof(int)).wait();
    

    // 数据移动
    q.memcpy(d_matB,r_matB,matB.getSize()*sizeof(int)).wait();
    

    // 数据移动
    q.memcpy(d_matC,r_matC,matC.getSize()*sizeof(int)).wait();
    


    //工作项划分
    sycl::range<3> local(1, 1, 16);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i=item_id/4%4;
            const auto j=item_id%4;
            // 嵌入计算
			
            matMul(d_matA+(i)*5,d_matB+(j)*5,d_matC+(i*4+j)*1);
        });
    }).wait();
    

    // 归并结果返回
    q.memcpy(r_matA, d_matA, matA.getSize()*sizeof(int)).wait();
    // 归并结果返回
    q.memcpy(r_matB, d_matB, matB.getSize()*sizeof(int)).wait();
    // 归并结果返回
    q.memcpy(r_matC, d_matC, matC.getSize()*sizeof(int)).wait();

    matC.array2Tensor(r_matC);

    sycl::free(d_matA, q);
    sycl::free(d_matB, q);
    sycl::free(d_matC, q);
}

int main() {
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
    return 0;
}