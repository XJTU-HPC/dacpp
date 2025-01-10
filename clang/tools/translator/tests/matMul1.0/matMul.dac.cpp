#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;

shell dacpp::list matrixMultiply_shell(const dacpp::Tensor<int, 2>& matA,
                                        const dacpp::Tensor<int, 2>& matB,
                                        dacpp::Tensor<int, 2>& matC) {
    dacpp::index idx1, idx2;
    //dacpp::split s(3,1);
    //binding(i, s);
    dacpp::list dataList{matA[idx1][{}], matB[{}][idx2], matC[idx1][idx2]};
    return dataList;
}

calc void matrixMultiply_calc(dacpp::Tensor<int, 1>& vecA,
                dacpp::Tensor<int, 1>& vecB,
                dacpp::Tensor<int, 1>& dotProduct) {
    for (int i = 0; i < 5; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}

int main() {
    // 初始化两个矩阵 A 和 B
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    dacpp::Tensor<int, 2> matA({4, 5}, dataA);

    std::vector<int> dataB{1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20};
    dacpp::Tensor<int, 2> matB({5, 4}, dataB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    dacpp::Tensor<int, 2> matC({4, 4}, dataC);

    matrixMultiply_shell(matA, matB, matC) <-> matrixMultiply_calc;
    matC.print();

    return 0;
}
