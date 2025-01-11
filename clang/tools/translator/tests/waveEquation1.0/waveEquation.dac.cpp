#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
using namespace std;

namespace dacpp {
    typedef std::vector<std::any> list;
}
// 网格参数
const int NX = 8;    // x方向网格数量
const int NY = 8;    // y方向网格数量
const double Lx = 10.0f; // x方向长度
const double Ly = 10.0f; // y方向长度
const double c = 1.0f;   // 波速
const int TIME_STEPS = 1000; // 时间步数
// 网格步长
const double dx = Lx / (NX - 1);
const double dy = Ly / (NY - 1);

// CFL条件
const double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件

shell dacpp::list waveEqShell(const dacpp::Tensor<double, 2>& matCur, 
                                const dacpp::Tensor<double, 2>& matPrev, 
                                dacpp::Tensor<double, 2>& matNext) {
    dacpp::split sp1(3, 1), sp2(3, 1);
    dacpp::index idx1, idx2;
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matCur[sp1][sp2], matPrev[idx1][idx2], matNext[idx1][idx2]};
    return dataList;
}

calc void waveEq(dacpp::Tensor<double, 2>& cur, dacpp::Tensor<double, 1>& prev, dacpp::Tensor<double, 1>& next) {
    double dt = 0.5f * std::fmin(dx, dy) / c; // 满足稳定性条件
    //next[0] =2.0f*cur[1*3+1] - prev[0] + c*c*dt*dt*((cur[2*3+1] -2.0f*cur[1*3+1]+cur[0*3+1]/(dx*dx))+(cur[1*3+2] -2.0f*cur[1*3+1]+cur[1*3+0]/(dy*dy)));
    double u_xx = (cur[2][1] - 2.0f * cur[1][1] + cur[0][1])/ (dx * dx);
    double u_yy = (cur[1][2] - 2.0f * cur[1][1] + cur[1][0])/ (dy * dy);
    //next[0] = 2.0f * cur[1 * 3 + 1] - prev[0] + c * c * dt * dt * ((cur[2 * 3 + 1] - 2.0f * cur[1 * 3 + 1] + cur[0 * 3 + 1] / (dx * dx)) + (cur[1 * 3 + 2] - 2.0f * cur[1 * 3 + 1] + cur[1 * 3 + 0] / (dy * dy)));
    // next[0] = cur[1 * 3 + 1] + 1;
    next[0]=2.0f*cur[1][1]-prev[0]+(c * c)*dt*dt*(u_xx+u_yy);
}

int main() {

    
    // 初始化波场
    vector<double> u_prev(NX * NY, 0.0f); // 前一步
    vector<double> u_curr(NX * NY, 0.0f);  // 当前步
    vector<double> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如一个高斯脉冲
    int cx = NX / 2;
    int cy = NY / 2;
    double sigma = 0.5f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            double x = i * dx;
            double y = j * dy;
            u_prev[i*NX+j] = std::exp(-((x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2)) / (2 * sigma * sigma));
        }
    }

    for(int i = 0; i < 1; i++){
        for(int j = 0; j < NY; j++){
            std::cout << u_prev[i * NX + j] << " " ;
        }
        std::cout << std::endl;
    }

  

    //std::vector<int> shape_u_curr = {8, 8};
    dacpp::Tensor<double, 2> u_curr_tensor({8, 8}, u_curr);

    std::vector<double> u_prev_middle_points;
    for (int i = 1; i <= 6; i++) {
        std::vector<double> row;
        for (int j = 1; j <= 6; j++) {  
            u_prev_middle_points.push_back(static_cast<double>(u_prev[i*NY+j]));  
        }
        
    }
    //std::vector<int> shape2= {6, 6};
    dacpp::Tensor<double, 2> u_prev_middle_tensor({6, 6}, u_prev_middle_points);
    //u_next取点
    std::vector<double> u_next_middle_points;
    for (int i = 1; i <= 6; i++) {
        std::vector<double> row;
        for (int j = 1; j <= 6; j++) {  
            u_next_middle_points.push_back(static_cast<double>(u_next[i*NY+j]));  
        }
        
    }
    
    for(int i = 0;i < TIME_STEPS; i++) {
        
        //std::vector<int> shape1= {6, 6};
        dacpp::Tensor<double, 2> u_next_middle_tensor({6, 6}, u_next_middle_points);


        
        waveEqShell(u_curr_tensor, u_prev_middle_tensor, u_next_middle_tensor) <-> waveEq;

       // u_next_middle_tensor.print();

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                u_prev_middle_tensor[i-1][j-1]=u_curr_tensor[i][j];
            }
        }

        for (int i = 1; i <= NX-2; i++) {
            for(int j = 1; j <=NY-2; j++){
                u_curr_tensor[i][j]=u_next_middle_tensor[i-1][j-1];
            }
        }

        // 处理边界条件（绝热边界：导数为零）


        for (int i = 0; i < NX; ++i) {       
            u_curr_tensor[i][NY-1]=0;
            u_curr_tensor[i][0]=0;
        }
        for (int j = 0; j < NY; ++j) {
            u_curr_tensor[NX - 1][j]=0;
            u_curr_tensor[0][j]=0;
             // 底部边界
        }

        
    }
    u_curr_tensor.print();


    
    // 输出最终结果的某些值作为示例
    //cout << "Final wave state at center: " << u_curr[(NX/2)*NY + (NY/2)] << "\n";
    
    return 0;
}
