#include "../dacppLib/include/Slice.h"
#include "../dacppLib/include/Tensor.hpp"

namespace dacpp {
typedef std::vector<std::any> list;
}

using dacpp::Tensor;

shell dacpp::list split(const Tensor<int>& matA, const Tensor<int>& matB, Tensor<int>& matC) {
    dacpp::RegularSplit s1("i", 2, 2);
    dacpp::RegularSplit s2("j", 2, 2);
    dacpp::list dataList{matA[s1][{}], matB[{}][s2], matC[s1][s2]};
    return dataList;
}

calc void multiplication(const Tensor<int>& mat1, const Tensor<int>& mat2, Tensor<int>& mat3) {
    // 分块矩阵乘法
}

int main() {
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeA{4, 4};
    Tensor<int> matA(dataA, shapeA);

    std::vector<int> dataB{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeB{4, 4};
    Tensor<int> matB(dataB, shapeB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeC{4, 4};
    Tensor<int> matC(dataC, shapeC);

    split(matA, matB, matC) <-> multiplication;

    return 0;
}