shell dacpp::list split1(Tensor<int>& matA, Tensor<int>& matB, Tensor<int>& matC) {
    dacpp::slice s1("s1", 2, 2);
    dacpp::slice s2("s2", 2, 2);
    dacpp::list dataList{matA[s1][{}], matB[{}][s2], matC[s1][s2]};
    return dataList;
}

calc calc1(Tensor<int>& matA, Tensor<int>& setB, Tensor<int>& setC) {
    split2(matA, matB, matC) <-> calc2;
}

shell dacpp::list split2(Tensor<int>& matA, Tensor<int>& matB, Tensor<int>& matC) {
    dacpp::slice s1("s3", 2, 2);
    dacpp::list dataList{matA[{}][s1], matB[s1][{}], matC[{}][{}]};
    return dataList;
}

calc calc2(Tensor<int>& matA, Tensor<int>& matB, Tensor<int>& matC) {
    split3(matA, matB, matC) <-> calc3;
}

shell dacpp::list Split3(const Tensor<int>& matA, const Tensor<int>& matB, Tensor<int>& matC) {
    dacpp::index i("i");
    dacpp::index j("j");
    dacpp::list dataList{matA[{i}][{}], matB[{}][{j}], matC[{i}][{j}]};
    return dataList;
}

calc calc3(Tensor<int>& vecA, Tensor<int>& vecB, Tensor<int>& dotProduct) {
    for(int i = 0; i < vecA.getShape(0); i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}