#include <iostream>
#include <vector>
#include <cmath>
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Slice.h"
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"

// 定义矩阵大小
const int N = 100; // 可以修改 N 的值来改变矩阵大小
const int max_iter = 100;
const float tolerance = 1e-6;
using dacpp::Tensor;

namespace dacpp {
    typedef std::vector<std::any> list;
}






#include <sycl/sycl.hpp>
#include "DataReconstructor.h"

using namespace sycl;

void jacobi(float* a, float* b, float* x, float* x_new, int* num) 
{
    float sigma = 0;
    for (int i = 0; i < N; ++i) {
        if (i != num[0]) {
            sigma += a[i] * x[i];
        }
    }
    x_new[0] = (b[0] - sigma) / a[num[0]];
}


// 生成函数调用
void jacobiShell(const Tensor<float> & A, const Tensor<float> & b, const Tensor<float> & x, Tensor<float> & x_new, const Tensor<int> & nums) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.SetSplitSize(100);
    // 数据重组
    
    // 数据重组
    DataReconstructor<float> A_tool;
    float* r_A=(float*)malloc(sizeof(float)*10000);
    
    // 数据算子组初始化
    Dac_Ops A_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(100);
    A_ops.push_back(idx1);
    A_tool.init(A,A_ops);
    A_tool.Reconstruct(r_A);
    // 数据重组
    DataReconstructor<float> b_tool;
    float* r_b=(float*)malloc(sizeof(float)*100);
    
    // 数据算子组初始化
    Dac_Ops b_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(1);
    b_ops.push_back(idx1);
    b_tool.init(b,b_ops);
    b_tool.Reconstruct(r_b);
    // 数据重组
    DataReconstructor<float> x_tool;
    float* r_x=(float*)malloc(sizeof(float)*100);
    
    // 数据算子组初始化
    Dac_Ops x_ops;
    
    x_tool.init(x,x_ops);
    x_tool.Reconstruct(r_x);
    // 数据重组
    DataReconstructor<float> x_new_tool;
    float* r_x_new=(float*)malloc(sizeof(float)*100);
    
    // 数据算子组初始化
    Dac_Ops x_new_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(1);
    x_new_ops.push_back(idx1);
    x_new_tool.init(x_new,x_new_ops);
    x_new_tool.Reconstruct(r_x_new);
    // 数据重组
    DataReconstructor<int> nums_tool;
    int* r_nums=(int*)malloc(sizeof(int)*100);
    
    // 数据算子组初始化
    Dac_Ops nums_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(1);
    nums_ops.push_back(idx1);
    nums_tool.init(nums,nums_ops);
    nums_tool.Reconstruct(r_nums);
    // 设备内存分配
    
    // 设备内存分配
    float *d_A=malloc_device<float>(10000,q);
    // 设备内存分配
    float *d_b=malloc_device<float>(100,q);
    // 设备内存分配
    float *d_x=malloc_device<float>(100,q);
    // 设备内存分配
    float *d_x_new=malloc_device<float>(100,q);
    // 设备内存分配
    int *d_nums=malloc_device<int>(100,q);
    // 数据移动
    
    // 数据移动
    q.memcpy(d_A,r_A,10000*sizeof(float)).wait();
    // 数据移动
    q.memcpy(d_b,r_b,100*sizeof(float)).wait();
    // 数据移动
    q.memcpy(d_x,r_x,100*sizeof(float)).wait();
    // 数据移动
    q.memcpy(d_nums,r_nums,100*sizeof(int)).wait();   
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 100);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto idx1=(item_id+(0)+100)%100;
            // 嵌入计算
			
            jacobi(d_A+(idx1*100),d_b+(idx1*1),d_x,d_x_new+(idx1*1),d_nums+(idx1*1));
        });
    }).wait();
    
    
    // 归约
    
    // 返回计算结果
    
    // 归并结果返回
    q.memcpy(r_x_new, d_x_new, 100*sizeof(float)).wait();
    x_new = x_new_tool.UpdateData(r_x_new);
    // 内存释放
    
    sycl::free(d_A, q);
    sycl::free(d_b, q);
    sycl::free(d_x, q);
    sycl::free(d_x_new, q);
    sycl::free(d_nums, q);
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now(); // 开始时间测量

    // 初始化系数矩阵 A 和向量 b
    std::vector<float> mat_A(N * N, 0.0f);
    std::vector<float> vec_b(N, 0.0f);
    std::vector<float> vec_x(N, 0.0f);     // 初始解
    std::vector<float> vec_x_new(N, 0.0f); // 更新后的解

    // 自动初始化 A 和 b
    for (int i = 0; i < N; ++i) {
        mat_A[i * N + i] = 4.0f; // 对角线元素，确保对角占优

        if (i > 0) {
            mat_A[i * N + i - 1] = -1.0f; // 下三角元素
        }
        if (i < N - 1) {
            mat_A[i * N + i + 1] = -1.0f; // 上三角元素
        }

        vec_b[i] = 1.0f; // 初始化向量 b，可根据需要修改
    }

    std::vector<int> A_shape = {100, 100};
    std::vector<int> b_shape = {100};
    std::vector<int> x_shape = {100};
    std::vector<int> x_new_shape = {100};
    Tensor<float> A(mat_A, A_shape);
    Tensor<float> b(vec_b, b_shape);
    Tensor<float> x(vec_x, x_shape);
    Tensor<float> x_new(vec_x_new, x_new_shape);
    
    bool converged = false;
    int iter = 0;
    std::vector<int> nums(100);
    // 使用 std::iota 填充 nums，值从 0 开始
    for(int i = 0;i < N;  i++){
        nums[i] = i;
    }
    std::vector<int> nums_shape = {100};
    Tensor<int> tensor_nums(nums, nums_shape);
    float* data = new float[1 * 100];
    float* data2 = new float[1 * 100];

    while (!converged && iter < max_iter) {
        jacobiShell(A, b, x, x_new, tensor_nums);
        
        x.tensor2Array(data);
        x_new.tensor2Array(data2);

        float max_error = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error, std::fabs(data2[i] - data[i]));
        }

        if (max_error < tolerance) {
            converged = true;
        }

        // 更新 x
        for(int i = 0;i<N;i++){
            float* data = new float[1];
            x_new[i].tensor2Array(data);
            x[i].array2Tensor(data);    
        }
        //x=x_new;

        ++iter;
    }


    // 输出结果
    std::cout << "迭代次数: " << iter << std::endl;
    std::cout << "解向量 x:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << data2[i] << " ";
    }
    std::cout << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now(); // 结束时间测量
    std::chrono::duration<double> duration = end_time - start_time; // 计算持续时间
    std::cout << "总执行时间: " << duration.count() << " 秒" << std::endl; // 输出执行时间
    return 0;
}
