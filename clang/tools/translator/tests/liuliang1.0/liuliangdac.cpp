#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Slice.h"
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"

using dacpp::Tensor;
namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 100;       // 路段长度
const double TIME_STEPS = 200;  // 时间步数
const double DELTA_T = 0.01; // 时间步长
const double DELTA_X = 1.0;  // 空间步长

// 流量函数，考虑密度对流量的影响
double q(double rho) {
    double V_max = 30; // 最大速度
    double rho_max = 50; // 最大密度
    return rho * V_max * (1 - rho / rho_max);
}

// 初始化密度，使用随机的分布
void initializeDensity(std::vector<double>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40; // 高密度区
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20; // 中密度区
        } else {
            rho[i] = 10; // 低密度区
        }
    }
}

// 计算交通流量
calc void lwr(double rho[], double new_rho[]) {
    new_rho[0] = rho[0] - (DELTA_T / DELTA_X) * (q(rho[1]) - q(rho[0]));
    new_rho[0] = std::max(0.0, new_rho[0]);
    //new_rho += 0.1 * (rho[0] + new_rho - 2 * rho[1]);
}

shell dacpp::list LWR_shell(const dacpp::Tensor<double> & rho, dacpp::Tensor<double> & new_rho) {
    dacpp::Index idx1("idx1");
    dacpp::RegularSplit S1("S1",2, 1);
    binding(idx1, S1);
    dacpp::list dataList{rho[{S1}], new_rho[{idx1}]};
    return dataList;
}

int main() {
    // 创建 Tensor 类型对象
    std::vector<double> rho(WIDTH, 0.0);
    std::vector<double> new_rho(WIDTH, 0.0);
    //std::vector<int> shape3 = {1, 100};
    //Tensor<int> new_rho(middle_points, shape3);

    initializeDensity(rho);

    for (int t = 0; t < TIME_STEPS; ++t) {

        // 使用 LWR 算法
        std::vector<double> middle_points_out;
        for (int i = 1; i <= 98; i++) {
          middle_points_out.push_back(static_cast<double>(new_rho[i]));
        }
        std::vector<int> shape = {98,1};
        Tensor<double> middle_out_tensor(middle_points_out, shape);

        std::vector<double> middle_points_in;
        for (int i = 1; i <= 99; i++) {
          middle_points_in.push_back(static_cast<double>(rho[i]));
        }
        std::vector<int> shape2 = {99,1};
        Tensor<double> middle_in_tensor(middle_points_in, shape2);
        LWR_shell(middle_in_tensor, middle_out_tensor) <-> lwr;

        double* r_1 = new double[98];
        middle_out_tensor.tensor2Array(r_1);
        std::vector<double> r(r_1, r_1 + 98);

        for (int i = 1; i <= 98; i++) {
            rho[i] = r[i-1];
        }
        
        // 处理边界条件
        rho[0] = rho[1]; // 左边界无车流
        rho[WIDTH - 1] = rho[WIDTH - 2]; // 右边界无车流

    }
    for (int x = 0; x < WIDTH; ++x) {
        std::cout <<  static_cast<int>(rho[x]) <<",";
    }
    //rho_tensor.print();

    // 释放动态分配的内存

    return 0;
}
