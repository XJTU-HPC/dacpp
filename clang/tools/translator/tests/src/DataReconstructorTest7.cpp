#include <iostream>
#include <vector>
#include <cmath>
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Slice.h"
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"

using dacpp::Tensor;
using namespace std;

// 网格参数
const int NX = 8;    // x方向网格数量
const int NY = 8;    // y方向网格数量
const float Lx = 10.0f; // x方向长度
const float Ly = 10.0f; // y方向长度
const float c = 1.0f;   // 波速
const int TIME_STEPS = 1; // 时间步数
// 网格步长
const float dx = Lx / (NX - 1);
const float dy = Ly / (NY - 1);

// CFL条件
// const float dt = 0.5 * std::fmin(dx, dy) / c; // 满足稳定性条件

#include "DataReconstructor.h"

// 生成函数调用
void waveEqShell(const dacpp::Tensor<float> & matCur, const dacpp::Tensor<float> & matPrev, dacpp::Tensor<float> & matNext) { 
    // 算子初始化
    
    // 规则分区算子初始化
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.SetSplitSize(6);
    // 规则分区算子初始化
    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.SetSplitSize(6);
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.SetSplitSize(6);
    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.SetSplitSize(6);
    // 数据重组
    
    // 数据重组
    DataReconstructor<float> matCur_tool;
    float* r_matCur=(float*)malloc(sizeof(float)*324);
    
    // 数据算子组初始化
    Dac_Ops matCur_ops;
    sp1.setDimId(0);
    sp1.setSplitLength(24);
    matCur_ops.push_back(sp1);
    sp2.setDimId(1);
    sp2.setSplitLength(9);
    matCur_ops.push_back(sp2);
    
    matCur_tool.init(matCur,matCur_ops);
    matCur_tool.Reconstruct(r_matCur);
    std::cout<<"重组后的 r_matCur:\n";
    for(int i = 0; i < 36; i++) {
        for (size_t j = 0; j < 9; j++){
            std::cout<<r_matCur[i*9+j]<<" ";
        }
        std::cout<<"\n";
    }
    // for(int i = 0; i < 6; i++) {
    //     for (size_t j = 0; j < 24; j++){
    //         std::cout<<r_matCur[i*24+j]<<" ";
    //     }
    //     std::cout<<"\n";
    // }


    // 数据重组
    DataReconstructor<float> matPrev_tool;
    float* r_matPrev=(float*)malloc(sizeof(float)*36);

    // 数据算子组初始化
    Dac_Ops matPrev_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(6);
    matPrev_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(1);
    matPrev_ops.push_back(idx2);
    matPrev_tool.init(matPrev,matPrev_ops);
    matPrev_tool.Reconstruct(r_matPrev);

    std::cout<<"重组后的 r_matPrev:\n";
    for(int i = 0; i < 6; i++) {
        for (size_t j = 0; j < 6; j++){
            std::cout<<r_matPrev[i*6+j]<<" ";
        }
        std::cout<<"\n";
    }

    // 数据重组
    DataReconstructor<float> matNext_tool;
    float* r_matNext=(float*)malloc(sizeof(float)*36);
    
    // 数据算子组初始化
    Dac_Ops matNext_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(6);
    matNext_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(1);
    matNext_ops.push_back(idx2);
    matNext_tool.init(matNext,matNext_ops);
    matNext_tool.Reconstruct(r_matNext);
}

int main() {

    
    // 初始化波场
    vector<float> u_prev(NX * NY, 0.0f); // 前一步
    vector<float> u_curr(NX * NY, 0.0f);  // 当前步
    vector<float> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如一个高斯脉冲
    int cx = NX / 2;
    int cy = NY / 2;
    float sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            float x = i * dx;
            float y = j * dy;
            u_prev[i*NX+j] = i*NX+j;
        }
    }


    std::vector<int> shape_u_curr = {8, 8};
    Tensor<float> u_curr_tensor(u_curr, shape_u_curr);
    std::cout<<"init u_curr:\n";
    u_curr_tensor.print();
    
    std::vector<float> u_prev_middle_points;
    for (int i = 1; i <= 6; i++) {
        for (int j = 1; j <= 6; j++) {  
            u_prev_middle_points.push_back(u_prev[i*NY+j]);  
        }    
    }
    std::vector<int> shape2= {6, 6};
    Tensor<float> u_prev_middle_tensor(u_prev_middle_points, shape2);
    std::cout<<"init u_prev:\n";
    u_prev_middle_tensor.print();

    std::vector<float> u_next_middle_points;
    for (int i = 1; i <= 6; i++) {
        for (int j = 1; j <= 6; j++) {  
            u_next_middle_points.push_back(u_next[i*NY+j]);  
        }
    }
    std::vector<int> shape1= {6, 6};
    Tensor<float> u_next_middle_tensor(u_next_middle_points, shape1);
    std::cout<<"init u_next:\n";
    u_next_middle_tensor.print();

    for(int i = 0;i < TIME_STEPS; i++) {

        std::cout << "step " << i << ": " << std::endl;        
        std::cout << "input:\n";
        std::cout<<"u_curr_tensor:\n";
        u_curr_tensor.print();
        std::cout<<"u_prev_middle_tensor:\n";
        u_prev_middle_tensor.print();
        waveEqShell(u_curr_tensor, u_prev_middle_tensor, u_next_middle_tensor);
    }

    
    return 0;
}
