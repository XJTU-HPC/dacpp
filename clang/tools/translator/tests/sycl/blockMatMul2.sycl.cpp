#include <sycl/sycl.hpp>
using namespace sycl;
#include <vector>
#include<iostream>
#include "DataReconstructor.h"

void debug(int x) {
    std::cout << "ok" << x << std::endl;
}
void calc1_0(int* matA, int* matB, int* matC) {
    int a = 0;
    
}

void calc1_1(int* matA, int* matB, int* matC) {
    int b = 0;
}

void calc2_0(int* vecA, int* vecB, int* dotProduct) {
    int c = 0;
}

void calc2_1(int* vecA, int* vecB, int* dotProduct) {
    int d = 0;
}

void calc3(int* a, int* b, int* c) {
    c[0] = a[0] * b[0];
}
void blockMatMulSplit(dacpp::Tensor<int> &matA, dacpp::Tensor<int> &matB, dacpp::Tensor<int> &matC)
{
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);

    // 设备内存分配
    int *d_matA=malloc_device<int>(16,q);
    int *d_matB=malloc_device<int>(16,q);
    int *d_matC=malloc_device<int>(16*2*2,q);
    // 归约设备内存分配
    int *reduction_matC = malloc_device<int>(16*2,q);

    // 算子初始化
    RegularSlice si = RegularSlice("si", 2, 2);
	si.SetSplitSize(2);
	RegularSlice sj = RegularSlice("sj", 2, 2);
	sj.SetSplitSize(2);
    RegularSlice sk = RegularSlice("sk", 2, 2);
	sk.SetSplitSize(2);

    // 数据划分重组
    DataReconstructor<int> matA_tool;
    int* r_matA=(int*)malloc(sizeof(int)*16);
    Dac_Ops matA_ops;
    si.setDimId(0);
    si.setSplitLength(8);
    matA_ops.push_back(si);
    sk.setDimId(1);
    sk.setSplitLength(4);
    matA_ops.push_back(sk);
    matA_tool.init(matA,matA_ops);
    matA_tool.Reconstruct(r_matA);

    /*
    std::cout <<  "matA重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matA[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    */

    DataReconstructor<int> matB_tool;
    int* r_matB=(int*)malloc(sizeof(int)*16);
    Dac_Ops matB_ops;
    sk.setDimId(0);
    sk.setSplitLength(8);
    matB_ops.push_back(sk);
    sj.setDimId(1);
    sj.setSplitLength(4);
    matB_ops.push_back(sj);
    matB_tool.init(matB,matB_ops);
    matB_tool.Reconstruct(r_matB);

    /*
    std::cout <<  "matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    */

    DataReconstructor<int> matC_tool;
    int* r_matC=(int*)malloc(sizeof(int)*16*2*2);
    Dac_Ops matC_ops;
    si.setDimId(0);
    si.setSplitLength(8);
    matC_ops.push_back(si);
    sj.setDimId(1);
    sj.setSplitLength(4);
    matC_ops.push_back(sj);
    matC_tool.init(matC,matC_ops);
    matC_tool.Reconstruct(r_matC);

    /*
    std::cout <<  "matC重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matC[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    */

    // 如果嵌入计算的函数不存在，则不需要下面的数据移动、内核执行、归并归约
    // 数据移动
    q.memcpy(d_matA,r_matA,16*sizeof(int)).wait();
    q.memcpy(d_matB,r_matB,16*sizeof(int)).wait();
    q.memcpy(d_matC,r_matC,16*2*2*sizeof(int)).wait();

    // 内核执行
    sycl::range<3> local(1, 1, 8);
    sycl::range<3> global(1, 1, 1);
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
            const auto si=item_id/2/2%2;
            const auto sj=item_id/2%2;
            const auto sk=item_id%2;

            // 嵌入计算
            calc1_0(d_matA+(si*8+sk*4),d_matB+(sk*8+sj*4),d_matC+(si*16+sj*8+sk*4));
        });
    }).wait();

    std::cout<<std::endl;
    std::cout <<  "归约前计算结果:\n";
    q.memcpy(r_matC, d_matC, 32*sizeof(int)).wait();
    for(int i=0;i<32;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<std::endl;

    // 归约归并
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(8 * 4),
        reduction(span<int,16>(reduction_matC,16), 
        sycl::plus<>(),
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            	reducer[i % 4+i/8*4].combine(d_matC[i]);
     	});
    }).wait();
    q.memcpy(r_matC,reduction_matC, 16*sizeof(int)).wait();
    std::cout <<  "归约后计算结果(未归并):\n";
    for(int i=0;i<16;i++) {
        std::cout << r_matC[i] << " ";
        if(i%4==3) std::cout<<"\n";
    }
    matC = matC_tool.UpdateData(r_matC);
    std::cout <<  "归约后计算结果(归并):\n";
    matC.print();

    // 算子初始化
    Index i = Index("i");
	i.SetSplitSize(2);
	Index j = Index("j");
	j.SetSplitSize(2);
    
    // 数据划分重组
    i.setDimId(0);
    i.setSplitLength(2);
    matA_tool.push_back(i);
    matA_tool.Reconstruct(r_matA);

    std::cout <<  "matA重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matA[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    j.setDimId(1);
    j.setSplitLength(2);
    matB_tool.push_back(j);
    matB_tool.Reconstruct(r_matB);

    std::cout <<  "matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    i.setDimId(0);
    i.setSplitLength(2);
     matC_tool.push_back(i);
    j.setDimId(1);
    j.setSplitLength(1);
    matC_tool.push_back(j);
    matC_tool.Reconstruct(r_matC);

    std::cout <<  "matC重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matC[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    // 如果嵌入计算的函数不存在，则不需要下面的数据移动、内核执行、归并归约
    // 数据移动
    // q.memcpy(d_matA,r_matA,16*sizeof(int)).wait();
    // q.memcpy(d_matB,r_matB,16*sizeof(int)).wait();
    // q.memcpy(d_matC,r_matC,32*sizeof(int)).wait();

    // // 内核执行
    // sycl::range<3> local(1, 1, 32);
    // sycl::range<3> global(1, 1, 1);
    // q.submit([&](handler &h) {
    //     h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
    //         const auto item_id = item.get_local_id(2);
    //         // 索引初始化
    //         const auto si=item_id/2/2/2/2%2;
    //         const auto sj=item_id/2/2/2%2;
    //         const auto sk=item_id/2/2%2;
    //         const auto i=item_id/2%2;
    //         const auto j=item_id%2;

    //         // 嵌入计算
    //         calc2_0(d_matA+(si*8+sk*4+i*2),d_matB+(sk*8+sj*4+j*2),d_matC+(si*16+sj*8+sk*4+i*2+j*1));
    //     });
    // }).wait();

    // std::cout<<std::endl;
    // std::cout <<  "归约前计算结果:\n";
    // q.memcpy(r_matC, d_matC, 32*sizeof(int)).wait();
    // for(int i=0;i<32;i++) std::cout<<r_matC[i]<<" ";
    // std::cout<<std::endl;

    // // 归约归并
    // int *reduction_matC = malloc_device<int>(16,q);
    // q.submit([&](handler &h) {
    // 	h.parallel_for(
    //     range<1>(8 * 4),
    //     reduction(span<int,16>(reduction_matC,16), 
    //     sycl::plus<>(),
    //     property::reduction::initialize_to_identity()),
    //     [=](id<1> i,auto &reducer) {
    //         	reducer[i % 4+i/8*4].combine(d_matC[i]);
    //  	});
    // }).wait();
    // q.memcpy(r_matC,reduction_matC, 16*sizeof(int)).wait();
    // std::cout <<  "归约后计算结果(未归并):\n";
    // for(int i=0;i<16;i++) {
    //     std::cout << r_matC[i] << " ";
    //     if(i%4==3) std::cout<<"\n";
    // }
    // matC = matC_tool.UpdateData(r_matC);
    // std::cout <<  "归约后计算结果(归并):\n";
    // matC.print();

    // 算子初始化
    Index k = Index("k");
	k.SetSplitSize(2);

    // 数据划分重组
    k.setDimId(0);
    k.setSplitLength(1);
    matA_tool.push_back(k);
    matA_tool.Reconstruct(r_matA);

    // std::cout <<  "matA重组结果:\n";
    // for(int i=0;i<4;i++){
    //     for(int j=0;j<4;j++){
    //         std::cout<<r_matA[i*4+j*1]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    k.setDimId(1);
    k.setSplitLength(1);
    matB_tool.push_back(k);
    matB_tool.Reconstruct(r_matB);

    std::cout <<  "matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    // matC 保形不重组
    std::cout <<  "matC重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matC[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    // 如果嵌入计算的函数不存在，则不需要下面的数据移动、内核执行、归并归约
    // 数据移动
    q.memcpy(d_matA,r_matA,16*sizeof(int)).wait();
    q.memcpy(d_matB,r_matB,16*sizeof(int)).wait();
    q.memcpy(d_matC,r_matC,32*sizeof(int)).wait();

    // 内核执行
    sycl::range<3> local1(1, 1, 64);
    sycl::range<3> global1(1, 1, 1);
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global1 * local1, local1),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
            const auto si=item_id/2/2/2/2/2%2;
            const auto sj=item_id/2/2/2/2%2;
            const auto sk=item_id/2/2/2%2;
            const auto i=item_id/2/2%2;
            const auto j=item_id/2%2;
            const auto k=item_id%2;

            // 嵌入计算
            calc3(d_matA+(si*8+sk*4+i*2+k*1),d_matB+(sk*8+sj*4+j*2+k*1),d_matC+(si*32+sj*16+sk*8+i*4+j*2+k*1));
        });
    }).wait();

    std::cout<<std::endl;
    std::cout <<  "归约前计算结果:\n";
    q.memcpy(r_matC, d_matC, 64*sizeof(int)).wait();
    for(int i=0;i<64;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<std::endl;

    // 归约归并
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(2 * 32),
        reduction(span<int,32>(reduction_matC,32), 
        sycl::plus<>(),
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            	reducer[i % 1+i/(2*1)*1].combine(d_matC[i]);
     	});
    }).wait();
    q.memcpy(d_matC,reduction_matC, 32*sizeof(int)).wait();

    std::cout <<  "第一次归约后计算结果:\n";
    q.memcpy(r_matC, d_matC, 32*sizeof(int)).wait();
    for(int i=0;i<32;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<std::endl;

    // 归约归并
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(2 * 16),
        reduction(span<int,16>(reduction_matC,16), 
        sycl::plus<>(),
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            	reducer[i % 4+i/(2*4)*4].combine(d_matC[i]);
     	});
    }).wait();
    q.memcpy(d_matC,reduction_matC, 16*sizeof(int)).wait();
    
    q.memcpy(r_matC,d_matC, 16*sizeof(int)).wait();
    std::cout <<  "两次归约后计算结果(未归并):\n";
    for(int i=0;i<16;i++) {
        std::cout << r_matC[i] << " ";
        if(i%4==3) std::cout<<"\n";
    }
    matC = matC_tool.UpdateData(r_matC);
    std::cout <<  "归约后计算结果(归并):\n";
    matC.print();


}
int main() {
    std::vector<int> dataA{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    std::vector<int> shapeA{4,4};
    dacpp::Tensor<int> matA(dataA,shapeA);
    std::vector<int> dataB{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    std::vector<int> shapeB{4,4};
    dacpp::Tensor<int> matB(dataB,shapeB);
    std::vector<int> dataC{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    std::vector<int> shapeC{4,4};
    dacpp::Tensor<int> matC(dataC,shapeC);

    blockMatMulSplit(matA,matB,matC);
    
    return 0;
}