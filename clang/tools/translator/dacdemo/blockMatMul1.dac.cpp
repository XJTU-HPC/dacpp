shell dacpp::list split1(Tensor<int>& matA, Tensor<int>& matB, Tensor<int>& matC) {
    dacpp::RegularSplit si("si", 2, 2);
    dacpp::RegularSplit sj("sj", 2, 2);
    dacpp::list dataList{matA[si][{}], matB[{}][sj], matC[si][sj]};
    return dataList;
}

calc calc1(Tensor<int>& matA, Tensor<int>& matB, Tensor<int>& matC) {
    split2(matA, matB, matC) <-> calc2;
}

shell dacpp::list Split2(const Tensor<int>& matA, const Tensor<int>& matB, Tensor<int>& matC) {
    dacpp::index i("i");
    dacpp::index j("j");
    dacpp::list dataList{matA[{i}][{}], matB[{}][{j}], matC[{i}][{j}]};
    return dataList;
}

calc calc2(Tensor<int>& vecA, Tensor<int>& vecB, Tensor<int>& dotProduct) {
    for(int i = 0; i < vecA.getShape(0); i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}

split1 (matA, matB, matC) <-> calc1;