#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <any>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <queue>
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Slice.h"
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"

using dacpp::Tensor;

namespace dacpp {
    typedef std::vector<std::any> list;
}

double phi(double x) {
    return x*x*x+x;
}

double alpha(double t) {
    return 0.0;
}

double beta(double t) {
    return 1.0+exp(t);
}

double f(double x, double t) {
    return x*exp(t)-6*x;
}

double exact(double x, double t) {
    return x*(x*x+exp(t));
}

//同样的问题，划分时，一个待计算数据和三个计算数据，一共四个数据要划分到一起





#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
//改，改成.h

using namespace sycl;

void pde(double* u_kin, double* u_kout, double* r) 
{
    u_kout[0] = r[0] * u_kin[0] + (1 - 2 * r[0]) * u_kin[1] + r[0] * u_kin[2];
}


// 生成函数调用
void PDE(dacpp::Tensor<int> u_kin, dacpp::Tensor<int> u_kout, dacpp::Tensor<int> r) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.SetSplitSize(0);
    // 规则分区算子初始化
    RegularSlice S1 = RegularSlice("S1", 3, 1);
    S1.SetSplitSize(4);
    // 数据重组
    
    // 数据重组
    DataReconstructor<int> u_kin_tool;
    int* r_u_kin=(int*)malloc(sizeof(int)*6);
    
    // 数据算子组初始化
    Dac_Ops u_kin_ops;
    
    u_kin_tool.init(u_kin,u_kin_ops);
    u_kin_tool.Reconstruct(r_u_kin);
    // 数据重组
    DataReconstructor<int> u_kout_tool;
    int* r_u_kout=(int*)malloc(sizeof(int)*4);
    
    // 数据算子组初始化
    Dac_Ops u_kout_ops;
    
    u_kout_tool.init(u_kout,u_kout_ops);
    u_kout_tool.Reconstruct(r_u_kout);
    // 数据重组
    DataReconstructor<int> r_tool;
    int* r_r=(int*)malloc(sizeof(int)*1);
    
    // 数据算子组初始化
    Dac_Ops r_ops;
    
    r_tool.init(r,r_ops);
    r_tool.Reconstruct(r_r);
    // 设备内存分配
    
    // 设备内存分配,改，int -> double
    double *d_u_kin=malloc_device<double>(303,q);
    // 设备内存分配
    double *d_u_kout=malloc_device<double>(3,q);
    // 设备内存分配
    double *d_r=malloc_device<double>(1,q);
    // 数据移动
       
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 4);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto idx1=item_id%4;
            const auto S1=item_id%4;
            // 嵌入计算
			
            pde(d_u_kin+(idx1*303),d_u_kout+(idx1*303),d_r);
            //改，删去d_r删去+（）
        });
    }).wait();
    
    
    // 归约
    
    // 归约
    int reduction_u_kin[1212] = {0};
    //增加reduction_u_kin的初始化
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(0 * 1212),
        reduction(span<int,1212>(reduction_u_kin,1212), 
        sycl::plus<>(),
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            reducer[i % 1 + i/(1*0)*1].combine(d_u_kin[i]);
     	});
 }).wait();
	q.memcpy(d_u_kin,reduction_u_kin, 1212*sizeof(int)).wait();

    // 归约
    int reduction_u_kout[606] = {0};
    //增加reduction_u_kin的初始化
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(0 * 12),
        reduction(span<int,12>(reduction_u_kout,12), 
        sycl::plus<>(),
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            reducer[i % 1 + i/(1*0)*1].combine(d_u_kout[i]);
     	});
 }).wait();
	q.memcpy(d_u_kout,reduction_u_kout, 12*sizeof(int)).wait();

    // 归约
    int reduction_r[6] = {0};
    //增加reduction_u_kin的初始化
    q.submit([&](handler &h) {
    	h.parallel_for(
        range<1>(0 * 6),
        reduction(span<int,6>(reduction_r,6), 
        sycl::plus<>(),
        property::reduction::initialize_to_identity()),
        [=](id<1> i,auto &reducer) {
            reducer[i % 1 + i/(1*0)*1].combine(d_r[i]);
     	});
 }).wait();
	q.memcpy(d_r,reduction_r, 6*sizeof(int)).wait();

    // 返回计算结果
    
    // 归并结果返回
    q.memcpy(r_u_kin, d_u_kin, 1212*sizeof(int)).wait();
	u_kin = u_kin_tool.UpdateData(r_u_kin);
    // 归并结果返回
    q.memcpy(r_u_kout, d_u_kout, 12*sizeof(int)).wait();
	u_kout = u_kout_tool.UpdateData(r_u_kout);
    // 归并结果返回
    q.memcpy(r_r, d_r, 6*sizeof(int)).wait();
	r = r_tool.UpdateData(r_r);
    // 内存释放
    
    sycl::free(d_u_kin, q);
    sycl::free(d_u_kout, q);
    sycl::free(d_r, q);
}

int main() {
    int n = 100; //时间域n等分
    int m = 5; //空间域m等分
    int r = 1;
    double a = 1.0;
    double h = 1.0 / m; //空间步长
    double tau = 1.0 / n; //时间步长
    double *x,*t,**u;
    
    //r=a*tau/(h*h);  //网比
    //printf("r=%.4f.\n",r);
    

    x = (double*)malloc(sizeof(double)*(m+1));
    for (int i=0;i<=m;i++) {
        x[i]=i*h;
    }
    t = (double*)malloc(sizeof(double)*(n+1));
    for (int i = 0; i <= n; i++) {
        t[i]=i*tau;
    }
    u = (double**)malloc(sizeof(double*)*(m+1));
    for (int i=0;i<=m;i++) {
        u[i]=(double*)malloc(sizeof(double)*(n+1));
    }
    for (int i = 0; i <= m; i++)
        u[i][0]=phi(x[i]);
    for (int i = 1; i <= n; i++) {
        u[0][i]=alpha(t[i]);
        u[m][i]=beta(t[i]);
    }
    
    // Flatten the 2D u array into a 1D vector for Tensor creation
    std::vector<int> u_flat;
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            u_flat.push_back(static_cast<int>(u[i][j]));  // Cast if needed
        }
    }

    // Define the shape of the tensor (rows, columns) and create the Tensor
    std::vector<int> shape = {6, 101};
    Tensor<int> u_tensor(u_flat, shape);
    
    for (int k = 0; k < n-1; k++) {
        //输入数据是6个点，3个一组分为四组，输出数据四个点，降维
        //这里再输入之前要把输出那一行初始化为各长度为4的Tensor
        std::vector<int> middle_points;
        for (int i = 1; i <= 4; i++) {
          middle_points.push_back(static_cast<int>(u[k+1][i]));
        }
        std::vector<int> shape2 = {1, 4};
        Tensor<int> middle_tensor(middle_points, shape);
        
        PDE(u_tensor[{k}][{}], middle_tensor[{}],r);
        //改，删去生成的<->pde
        //计算完毕后，替换第1到4个点
        for (int i = 1; i <= 4; i++) {
            u_tensor[{k+1}][i] = middle_tensor[i];
        }

    }

    // 每个位置需要下，左下，右下，三个位置的元素，串行中从下往上，从左往右遍历计算
    // 那么每一行的元素计算是互不相关的，可以并行执行，所有的行从下往上串行执行

    int j = int(0.2 / tau);
    int number = int(0.4 / h);
    for (int k = j; k <= n; k = k + j) {
        printf("(x,t)=(%.1f,%.1f), y=%f, exact=%f, err=%.4e.\n",x[number],t[k],u[number][k],exact(x[number],t[k]),std::fabs(u[number][k]-exact(x[number],t[k])));//改，显式指定std::fabs
    }

    free(x);
    free(t);
    for (int i = 0; i <= m; i++) {
        free(u[i]);
    }
    free(u);

    return 0;
}