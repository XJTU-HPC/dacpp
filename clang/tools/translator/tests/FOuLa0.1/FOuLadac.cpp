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

shell dacpp::list PDE(dacpp::Tensor<int> u_kin, dacpp::Tensor<int> u_kout,dacpp::Tensor<int> r) {
    dacpp::Index idx1("idx1");
    dacpp::RegularSplit S1("S1",3,1);
    dacpp::list dataList{u_kin[{S1}][{}], u_kout[{idx1}][{}],r[{}]};
    return dataList;
}

calc void pde(int u_kin[], int  u_kout,double r) {
    u_kout = r * u_kin[0] + (1 - 2 * r) * u_kin[1] + r * u_kin[2];
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
        PDE(u_tensor[{k}][{}], u_tensor[{k+1}][{}],r) <-> pde;
    }

    // 每个位置需要下，左下，右下，三个位置的元素，串行中从下往上，从左往右遍历计算
    // 那么每一行的元素计算是互不相关的，可以并行执行，所有的行从下往上串行执行

    int j = int(0.2 / tau);
    int number = int(0.4 / h);
    for (int k = j; k <= n; k = k + j) {
        printf("(x,t)=(%.1f,%.1f), y=%f, exact=%f, err=%.4e.\n",x[number],t[k],u[number][k],exact(x[number],t[k]),fabs(u[number][k]-exact(x[number],t[k])));
    }

    free(x);
    free(t);
    for (int i = 0; i <= m; i++) {
        free(u[i]);
    }
    free(u);

    return 0;
}