namespace dacpp {

void reduce(int idx, int type) {}

}

#define $REDUCE_SUM(X) dacpp::reduce(X, 0);
#define $REDUCE_MAX(X) dacpp::reduce(X, 1);
#define $REDUCE_MIN(X) dacpp::reduce(X, 2);
#define $REDUCE_MAXLOC(X) dacpp::reduce(X, 3);
#define $REDUCE_MINLOC(X) dacpp::reduce(X, 4);

#include <vector>

#include "../dacppLib/Slice.h"
#include "../dacppLib/Tensor.hpp"

using dacpp::Tensor;

namespace dacpp {

typedef std::vector<std::any> list;

}

shell dacpp::list matMulSplit(const Tensor<int>& matA, const Tensor<int>& matB, Tensor<int>& matC) {
    int i;
    int j;
    $REDUCE_MAX(2)
    dacpp::list dataList{matA[{i}][{}], matB[{}][{j}], matC[{i}][{j}]};
    return dataList;
}

calc void matMul(Tensor<int>& vecA, Tensor<int>& vecB, Tensor<int>& dotProduct) {
    // vecMulSplit(vecA, vecB, dotProduct) <-> vecMul;
    for(int i = 0; i < vecA.getShape(0); i++) {
        dotProduct += vecA[i] * vecB[i];
    }
}

int main() {
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> shapeA{4, 5};
    Tensor<int> matA(dataA, shapeA);

    std::vector<int> dataB{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> shapeB{5, 4};
    Tensor<int> matB(dataB, shapeB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeC{4, 4};
    Tensor<int> matC(dataC, shapeC);

    matMulSplit(matA, matB, matC) <-> matMul;
    matC.print();
    return 0;
}