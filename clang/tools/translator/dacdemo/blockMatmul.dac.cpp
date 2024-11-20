shell dacpp::list split1(Tensor<int>& matA, Tensor<int>& matB, Tensor<int>& matC) {
    dacpp::slice si("si", 2, 2);
    dacpp::slice sj("sj", 2, 2);
    dacpp::slice sk("sk", 2, 2);
    dacpp::list dataList{matA[si][sk], matB[sk][sj], matC[si][sj]};
    return dataList;
}

calc calc1(Tensor<int>& matA, Tensor<int>& matB, Tensor<int>& matC) {
    M = matA.getShape(0);
    K = matA.getShape(1);
    N = matB.getShape(1);
    for (int i=0;i<M;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<K;k++){
                matC[i][j]+= matA[i][k]*matB[k][j];
            }
        }
    } 
}

split1(matA, matB, matC) <-> calc1;

