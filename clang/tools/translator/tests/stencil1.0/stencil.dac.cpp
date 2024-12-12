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

shell dacpp::list stencilShell(const dacpp::Tensor<float>& matIn, Tensor<float>& matOut) {
    dacpp::RegularSplit sp1("S1",3, 1), sp2("S2",3, 1);
    dacpp::Index idx1("idx1"), idx2("idx2");
    binding(sp1, idx1+1);
    binding(sp2, idx2+1);
    dacpp::list dataList{matIn[{sp1}][{sp2}], matOut[{idx1}][{idx2}]};
    return dataList;
}

calc void stencil(float* mat, float* out) {
    out[0] = mat[1 * 32 + 1] + (mat[0 * 32 + 1] + mat[2 * 32 + 1] + mat[1 * 32 + 0] + mat[1 * 32 + 2] - mat[1 * 32 + 1]);
}

int main() {
    // 空间步长
    float dx = Lx / (NX - 1);
    float dy = Ly / (NY - 1);

    // 稳定性条件
    float dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
    float delta_t = 0.4f * dt_stability; // 选择一个更严格的时间步长以确保稳定性

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
                u_curr_tensor[i][j] = middle_tensor[i-1][j-1];
            }
        }

        // 处理边界条件（绝热边界：导数为零）
        for (int j = 0; j < NY; ++j) {
            u_curr_tensor[0 * NY + j] = u_curr_tensor[1 * NY + j];              // 顶部边界
            u_curr_tensor[(NX - 1) * NY + j] = u_curr_tensor[(NX - 2) * NY + j]; // 底部边界
        }
        for (int i = 0; i < NX; ++i) {
            u_curr_tensor[i * NY + 0] = u_curr_tensor[i * NY + 1];              // 左边界
            u_curr_tensor[i * NY + (NY - 1)] = u_curr_tensor[i * NY + (NY - 2)]; // 右边界
        }
        
    }
    float* data = new float[32 * 32];
    u_curr_tensor.tensor2Array(data);

    std::vector<std::vector<int>> vec2D;
    vec2D.resize(32, std::vector<int>(32));

    // 将一维数组的数据填充到二维数组中
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            vec2D[i][j] = data[i * 32 + j];
        }
    }
    for (const auto& row : vec2D) {  // 遍历二维vector的每一行
        for (int val : row) {  // 遍历每一行中的元素
            std::cout << val << " ";  // 打印元素
        }
        std::cout << std::endl;  // 每行结束后换行
    }

    // 输出最终结果的某些值作为示例
    //cout << "Final temperature at center: " << vec2D[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}
