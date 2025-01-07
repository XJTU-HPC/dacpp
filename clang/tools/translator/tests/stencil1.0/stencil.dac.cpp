#include <iostream>
#include <vector>
#include <cmath>
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Slice.h"
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"


using namespace std;
using dacpp::Tensor;
namespace dacpp {
    typedef std::vector<std::any> list;
}

// 网格参数
const int NX = 32;           // x方向网格数量
const int NY = 32;           // y方向网格数量
const float Lx = 10.0f;       // x方向长度
const float Ly = 10.0f;       // y方向长度
const float alpha = 0.01f;    // 热扩散系数
const int TIME_STEPS = 1000;  // 时间步数
// 空间步长
const float dx = Lx / (NX - 1);
const float dy = Ly / (NY - 1);

// 稳定性条件
const float dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
const float delta_t = 0.4f * dt_stability; // 选择一个更严格的时间步长以确保稳定性

shell dacpp::list stencilShell(const dacpp::Tensor<float>& matIn, Tensor<float>& matOut) {
    dacpp::RegularSplit sp1("sp1",3, 1), sp2("sp2",3, 1);
    dacpp::Index idx1("idx1"), idx2("idx2");
    binding(sp1, idx1);
    binding(sp2, idx2);
    dacpp::list dataList{matIn[{sp1}][{sp2}], matOut[{idx1}][{idx2}]};
    return dataList;
}

calc void stencil(float* mat, float* out) {
    out[0] = mat[1 * 3 + 1] + alpha *delta_t * (((mat[2 * 3 + 1] - 2.0f * mat[1 * 3 + 1] + mat[0 * 3 + 1]) / (dx * dx))+ ((mat[1 * 3 + 2] - 2.0f * mat[1 * 3 + 1] + mat[1 * 3 + 0]) / (dy * dy)));                
}

int main() {


    cout << "Grid size: " << NX << "x" << NY << "\n";
    cout << "dx = " << dx << ", dy = " << dy << ", delta_t = " << delta_t << "\n";

    // 初始化温度场
    vector<float> u_prev(NX * NY, 0.0f); // 前一步（在热传导方程中，只有当前步和上一步）
    vector<float> u_curr(NX * NY, 0.0f);  // 当前步
    vector<float> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如，中心有一个高斯分布的热源
    int cx = NX / 2;
    int cy = NY / 2;
    float sigma = 1.0f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            float x = i * dx;
            float y = j * dy;
            // 高斯分布
            u_curr[i * NY + j] = std::exp(-((x - Lx/2.0f)*(x - Lx/2.0f) + (y - Ly/2.0f)*(y - Ly/2.0f)) / (2.0f * sigma * sigma));
        }
    }

    std::vector<int> shape = {32, 32};
    Tensor<float> u_curr_tensor(u_curr, shape);

    for(int i=0;i<TIME_STEPS;i++) {
        std::vector<float> middle_points(NX * NY, 0.0f);
        for (int i = 1; i <= 30; i++) {
            std::vector<float> row;
            for (int j = 1; j <= 30; j++) {  
                middle_points.push_back(static_cast<float>(u_next[i*NY+j]));  
            }
            
        }
        std::vector<int> shape2 = {30, 30};
        Tensor<float> middle_tensor(middle_points, shape2);

        stencilShell(u_curr_tensor, middle_tensor) <-> stencil;

        for (int i = 1; i <= 30; i++) {
            for(int j = 1; j <=30; j++){
                float* data = new float[1];
                middle_tensor[i-1][j-1].tensor2Array(data);
                u_curr_tensor[i][j].array2Tensor(data);
            }
        }

        // 处理边界条件（绝热边界：导数为零）
        for (int j = 0; j < NY; ++j) {
            float* data = new float[1];
            u_curr_tensor[1][j].tensor2Array(data);
            u_curr_tensor[0][j].array2Tensor(data);              // 顶部边界
            u_curr_tensor[NX-2][j].tensor2Array(data);
            u_curr_tensor[NX - 1][j].array2Tensor(data);
             // 底部边界
        }
        for (int i = 0; i < NX; ++i) {
            float* data = new float[1];
            u_curr_tensor[i][1].tensor2Array(data);
            u_curr_tensor[i][0].array2Tensor(data);              // 顶部边界
            u_curr_tensor[i][NY-2].tensor2Array(data);
            u_curr_tensor[i][NY-1].array2Tensor(data);
        }
        
    }
    

    // 输出最终结果的某些值作为示例
    //cout << "Final temperature at center: " << vec2D[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}
