#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

void debug(int x) {
    std::cout << "ok" << x << std::endl;
}
void block_mat_mul(int *matA, int *matB, int *matC) {
    for (int i=0;i<2;i++){
        for (int j=0;j<2;j++){
            for (int k=0;k<2;k++){
                int a=matA[i*2+k];
                int b=matB[k*2+j];
                matC[i*2+j] += a*b;
            }
        }
    } 
}
int main() {
    auto selector = sycl::gpu_selector_v;
    queue q(selector);
    
    debug(1);
    int r_matA[16]={1,2,1,2,3,4,3,4,1,2,1,2,3,4,3,4};
    int r_matB[16]={1,2,1,2,3,4,3,4,1,2,1,2,3,4,3,4};
    int r_matC[32]={0};
    debug(2);
    // 设备内存分配
    int *d_matA=malloc_device<int>(16,q);
    // 设备内存分配
    int *d_matB=malloc_device<int>(16,q);
    // 设备内存分配
    int *d_matC=malloc_device<int>(16*2,q);
    debug(3);
    // 数据移动
    q.memcpy(d_matA,r_matA,16*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_matB,r_matB,16*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_matC,r_matC,16*2*sizeof(int)).wait();
    debug(4);
    //工作项划分
    sycl::range<3> local(1, 1, 8);
    sycl::range<3> global(1, 1, 1);
    
    // 队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化

            const auto si=item_id/2/2%2;
            const auto sj=item_id/2%2;
            const auto sk=item_id%2;
            // 嵌入计算
            block_mat_mul(&d_matA[(si*8+sk*4)],&d_matB[(sk*8+sj*4)],&d_matC[(si*16+sj*8+sk*4)]);
        });
    }).wait();
    
    debug(5);
    
    q.memcpy(r_matC, d_matC, 32*sizeof(int)).wait();
    for(int i=0;i<32;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<std::endl;
    
    return 0;
}