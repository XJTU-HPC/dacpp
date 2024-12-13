#include <iostream>
#include <vector>
#include <cmath>
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Slice.h"
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"

using dacpp::Tensor;
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}
// 网格参数
const int NX = 8;    // x方向网格数量
const int NY = 8;    // y方向网格数量
const float Lx = 10.0f; // x方向长度
const float Ly = 10.0f; // y方向长度
const float c = 1.0f;   // 波速
const int TIME_STEPS = 1; // 时间步数
// 网格步长
float dx = Lx / (NX - 1);
float dy = Ly / (NY - 1);

// CFL条件
float dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件

shell dacpp::list waveEqShell(const dacpp::Tensor<float>& matCur, const dacpp::Tensor<float>& matPrev, dacpp::Tensor<float>& matNext) {
    dacpp::RegularSplit sp1("S1",3, 1), sp2("S2",3, 1);
    dacpp::Index idx1("idx1"), idx2("idx2");
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matCur[{sp1}][{sp2}], matPrev[{idx1}][{idx2}], matNext[{idx1}][{idx2}]};
    return dataList;
}

calc void waveEq(float* cur, float* prev, float* next) {
    next[0] =2.0f*cur[1 *3  + 1] - prev[0] + c*c*dt*dt*((cur[2*3+1] -2.0f*cur[1*3+1]+cur[0*3+1]/(dx*dx))+(cur[1*3+2] -2.0f*cur[1*3+1]+cur[1*3+0]/(dy*dy)));
}

int main() {
    // 网格步长
    float dx = Lx / (NX - 1);
    float dy = Ly / (NY - 1);
    
    // CFL条件
    float dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件
    
    // 初始化波场
    vector<float> u_prev(NX * NY, 0.0f); // 前一步
    vector<float> u_curr;  // 当前步
    vector<float> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如一个高斯脉冲
    int cx = NX / 2;
    int cy = NY / 2;
    float sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            float x = i * dx;
            float y = j * dy;
            u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
        }
    }

    for(int i = 0; i < 1; i++){
        for(int j = 0; j < NY; j++){
            std::cout << u_prev[i * NX + j] << " " ;
        }
        std::cout << std::endl;
    }

  

    std::vector<int> shape_u_curr = {8, 8};
    Tensor<float> u_curr_tensor(u_curr, shape_u_curr);


    
    for(int i = 0;i < TIME_STEPS; i++) {
        std::vector<float> u_prev_middle_points;
        for (int i = 1; i <= 6; i++) {
            std::vector<float> row;
            for (int j = 1; j <= 6; j++) {  
                u_prev_middle_points.push_back(static_cast<float>(u_prev[i*NY+j]));  
            }
            
        }
        std::vector<int> shape2= {6, 6};
        Tensor<float> u_prev_middle_tensor(u_prev_middle_points, shape2);
        //u_next取点
        std::vector<float> u_next_middle_points;
        for (int i = 1; i <= 6; i++) {
            std::vector<float> row;
            for (int j = 1; j <= 6; j++) {  
                u_next_middle_points.push_back(static_cast<float>(u_next[i*NY+j]));  
            }
            
        }
        std::vector<int> shape1= {6, 6};
        Tensor<float> u_next_middle_tensor(u_next_middle_points, shape1);


        
        waveEqShell(u_curr_tensor, u_prev_middle_tensor, u_next_middle_tensor) <-> waveEq;

        u_next_middle_tensor.print();

        for (int i = 1; i <= 30; i++) {
            for(int j = 1; j <=30; j++){
                float* data = new float[1];
                u_curr_tensor[i][j].tensor2Array(data);
                u_prev_middle_tensor[i-1][j-1].array2Tensor(data);
            }
        }

        for (int i = 1; i <= 30; i++) {
            for(int j = 1; j <=30; j++){
                float* data = new float[1];
                u_next_middle_tensor[i-1][j-1].tensor2Array(data);
                u_curr_tensor[i][j].array2Tensor(data);
            }
        }

        // 处理边界条件（绝热边界：导数为零）


        for (int i = 0; i < NX; ++i) {
            float* data = new float{0};
            u_curr_tensor[i][0].array2Tensor(data);             
            u_curr_tensor[i][NY-1].array2Tensor(data);
        }
        for (int j = 0; j < NY; ++j) {
            float* data = new float{0};
            u_curr_tensor[0][j].array2Tensor(data);              // 顶部边界
            u_curr_tensor[NX - 1][j].array2Tensor(data);
             // 底部边界
        }

        
    }
    u_curr_tensor.print();


    
    // 输出最终结果的某些值作为示例
    //cout << "Final wave state at center: " << u_curr[(NX/2)*NY + (NY/2)] << "\n";
    
    return 0;
}
