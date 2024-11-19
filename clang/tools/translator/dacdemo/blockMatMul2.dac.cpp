#include "../dacppLib/include/Slice.h"
#include "../dacppLib/include/Tensor.hpp"

namespace dacpp {
typedef std::vector<std::any> list;
}

using dacpp::Tensor;
shell dacpp::list split1(Tensor<int>& matA, Tensor<int>& matB, Tensor<int>& matC) {
    dacpp::RegularSplit si("idx1", 2, 2);
    dacpp::RegularSplit sj("idx2", 2, 2);
    dacpp::RegularSplit sk("idx3", 2, 2);
    dacpp::list dataList{matA[si][sk], matB[sk][sj], matC[si][sj]};
    return dataList;
}

shell dacpp::list split2(const Tensor<int>& matA, const Tensor<int>& matB, Tensor<int>& matC) {
    dacpp::index i("idx4");
    dacpp::index j("idx5");
    dacpp::list dataList{matA[{i}][{}], matB[{}][{j}], matC[{i}][{j}]};
    return dataList;
}

shell dacpp::list split3(Tensor<int>& vecA, Tensor<int>& vecB, Tensor<int>& dotProduct) {
    dacpp::index k("idx6");
    dacpp::list dataList{vecA[{k}], vecB[{k}], dotProduct};
    return dataList;
}

calc calc1(Tensor<int>& matA, Tensor<int>& matB, Tensor<int>& matC) {
    split2(matA, matB, matC) <-> calc2;
}

calc calc2(Tensor<int>& vecA, Tensor<int>& vecB, Tensor<int>& dotProduct) {
    split3(vecA, vecB, dotProduct) <-> calc3;
}

calc calc3(int& a, int& b, int& c) {
    c = a*b;
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

    split1(matA, matB, matC) <-> calc1;

    return 0;
}