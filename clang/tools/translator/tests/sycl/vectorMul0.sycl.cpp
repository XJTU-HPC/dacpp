#include <sycl/sycl.hpp>
using namespace sycl;
#include <vector>
#include<iostream>
#include "DataReconstructor.h"

void debug(int x) {
    std::cout << "ok" << x << std::endl;
}
void vector_mul(int *vecA, int *vecB, int *vecC, int *vecD) {
    vecD[0] = vecA[0] * vecB[0] * vecC[0];
}
void vectorMulSplit(dacpp::Tensor<int,1> &vecA, dacpp::Tensor<int,1> &vecB, dacpp::Tensor<int,1> &vecC, dacpp::Tensor<int,1> &vecD)
{
    auto selector = gpu_selector_v;
    queue q(selector);

    DataInfo info_vecA;
    info_vecA.dim = vecA.getDim();
    for(int i = 0; i < info_vecA.dim; i++) info_vecA.dimLength.push_back(vecA.getShape(i));

    DataInfo info_vecB;
    info_vecB.dim = vecB.getDim();
    for(int i = 0; i < info_vecB.dim; i++) info_vecB.dimLength.push_back(vecB.getShape(i));

    DataInfo info_vecC;
    info_vecC.dim = vecC.getDim();
    for(int i = 0; i < info_vecC.dim; i++) info_vecC.dimLength.push_back(vecC.getShape(i));

    DataInfo info_vecD;
    info_vecD.dim = vecD.getDim();
    for(int i = 0; i < info_vecD.dim; i++) info_vecD.dimLength.push_back(vecD.getShape(i));

    Index i = Index("i");
    i.SetSplitSize(3);
    Index j = Index("j");
    j.SetSplitSize(3);

    DataReconstructor<int> vecA_tool;
    int* r_vecA=(int*)malloc(sizeof(int)*3);
    Dac_Ops vecA_ops;
    i.setDimId(0);
    i.setSplitLength(1);
    vecA_ops.push_back(i);
    vecA_tool.init(info_vecA,vecA_ops);
    vecA_tool.Reconstruct(r_vecA,vecA);

    std::cout <<  "vecA重组结果:\n";
    for(int i=0;i<3;i++){
        std::cout<<r_vecA[i]<<" ";
    }
    std::cout<<std::endl;

    DataReconstructor<int> vecB_tool;
    int* r_vecB=(int*)malloc(sizeof(int)*3);
    Dac_Ops vecB_ops;
    i.setDimId(0);
    i.setSplitLength(1);
    vecB_ops.push_back(i);
    vecB_tool.init(info_vecB,vecB_ops);
    vecB_tool.Reconstruct(r_vecB,vecB);

    std::cout <<  "vecB重组结果:\n";
    for(int i=0;i<3;i++){
        std::cout<<r_vecB[i]<<" ";
    }
    std::cout<<std::endl;

    DataReconstructor<int> vecC_tool;
    int* r_vecC=(int*)malloc(sizeof(int)*3);
    Dac_Ops vecC_ops;
    j.setDimId(0);
    j.setSplitLength(1);
    vecC_ops.push_back(j);
    vecC_tool.init(info_vecC,vecC_ops);
    vecC_tool.Reconstruct(r_vecC,vecC);

    std::cout <<  "vecC重组结果:\n";
    for(int i=0;i<3;i++){
        std::cout<<r_vecC[i]<<" ";
    }
    std::cout<<std::endl;

    DataReconstructor<int> vecD_tool;
    int* r_vecD=(int*)malloc(sizeof(int)*3*3);
    Dac_Ops vecD_ops;
    j.setDimId(0);
    j.setSplitLength(1);
    vecD_ops.push_back(j);
    vecD_tool.init(info_vecD,vecD_ops);
    vecD_tool.Reconstruct(r_vecD,vecD);

    std::cout <<  "vecD重组结果:\n";
    for(int i=0;i<3;i++){
        std::cout<<r_vecD[i]<<" ";
    }
    std::cout<<std::endl;

    int *d_vecA=malloc_device<int>(3,q);
    int *d_vecB=malloc_device<int>(3,q);
    int *d_vecC=malloc_device<int>(3,q);
    int *d_vecD=malloc_device<int>(3*3,q);

    q.memcpy(d_vecA,r_vecA,3*sizeof(int)).wait();
    q.memcpy(d_vecB,r_vecB,3*sizeof(int)).wait();
    q.memcpy(d_vecC,r_vecC,3*sizeof(int)).wait();
    // q.memcpy(d_vecD,r_vecD,3*sizeof(int)).wait();

    sycl::range<3> local(1, 1, 9);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
            const auto i=item_id/3%3;
            const auto j=item_id%3;

            // 嵌入计算
            vector_mul(d_vecA+(i*1),d_vecB+(i*1),d_vecC+(j*1),d_vecD+(j*3+i*1));
        });
    }).wait();

    // debug(4);
    std::cout<<std::endl;
    std::cout <<  "归约前计算结果:\n";
    q.memcpy(r_vecD, d_vecD, 9*sizeof(int)).wait();
    for(int i=0;i<9;i++) std::cout<<r_vecD[i]<<" ";
    std::cout<<std::endl;

    // 归约
    int *reduction_vecD = malloc_device<int>(3,q);
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(3 * 3),
        reduction(span<int,3>(reduction_vecD,3), 
        sycl::plus<>(),
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            	reducer[i % 1+i/(3*1)*1].combine(d_vecD[i]);
     	});
    }).wait();

    int *res = malloc_shared<int>(3,q);
    q.memcpy(res,reduction_vecD, 3*sizeof(int)).wait();
    std::cout <<  "归约后计算结果(未归并):\n";
    for(int i=0;i<3;i++) {
        std::cout << res[i] << " ";
    }
    std::cout<<"\n";

    vecD_tool.UpdateData(res,vecD);
    std::cout <<  "归约后计算结果(归并):\n";
    vecD.print();
}
int main() {
    std::vector<int> dataA{1,2,3};
    dacpp::Tensor<int,1> vecA(dataA);
    std::vector<int> dataB{1,2,3};
    dacpp::Tensor<int,1> vecB(dataB);
    std::vector<int> dataC{1,2,3};
    dacpp::Tensor<int,1> vecC(dataC);
    std::vector<int> dataD{0,0,0};
    dacpp::Tensor<int,1> vecD(dataD);

    vectorMulSplit(vecA,vecB,vecC,vecD);
    // std::cout <<  "归约后计算结果(归并):\n";
    //vecD.print();
    return 0;
}