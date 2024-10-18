shell dacpp::list MatMulSplit(Tensor<int>& a, Tensor<int>& b, Tensor<int>& c) {
    dacpp::index i;
    dacpp::index j;
    dacpp::list dataList{a[{i}][{}], b[{}][{j}], c[{i}][{j}]};
    return dataList;
}

shell dacpp::list VecMulSplit(Tensor<int>& a, Tensor<int>& b, int& c) {
    dacpp::index i;
    dacpp::list dataList{a[{i}],b[{i}],c}
}
calc void Mul(int& a,int& b,int& c) {
    c+=a*b;
}
calc void ExtraVecMul(Tensor<int>& a, Tensor<int>& b, int& c) {
    for(int id=0;id<aLen;id++) a[id]++;
    VecMulSplit(a,b,c) <-> Mul;
    c++;
}
int main() {
    MatMulSplit(a,b,c) <-> ExtraVecMul;
}