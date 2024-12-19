#include "/data/tqc/dacpp/clang/tools/translator/dacppLib/include/TensorProxy.hpp"
#include "/data/tqc/dacpp/clang/tools/translator/dacppLib/include/RefacTensor.hpp"
#include <iostream>
#include <vector>
//using dacpp::TensorProxy;
using dacpp::Tensor;

using std::vector;
using std::cout;
using std::endl;

int main() {
    vector<int> data{1,2,3,4,5,6,7,8,9};
    Tensor<int,3> u_tensor({3,3,1}, data);
    //TensorProxy<int, 2> pu_tensor(u_tensor, 0, 1);
    // for(int i=0;i<3;i++){
    //     for(int j=0;j<1;j++)
    //         cout<<pu_tensor[i][j]<< " ";
    //     cout<<std::endl;
    // }
    // cout<<std::string(50,'=')<<endl;
    //TensorProxy<int, 2> c_tensor = u_tensor[{}][{}];
    // c_tensor = u_tensor[{}][{}];
    //cout<<c_tensor.getCurrentDim();
    // for(int i=0;i<c_tensor.getShape(0);i++){
    //     for(int j=0;j<c_tensor.getShape(1);j++)
    //         cout<<c_tensor[i][j]<< " ";
    //     cout<<std::endl;
    // }
    // cout<<std::string(50,'=')<<endl;
    // c_tensor = u_tensor[{0}][{0,2}];
    // for(int i=0;i<c_tensor.getShape(0);i++){
    //     for(int j=0;j<c_tensor.getShape(1);j++)
    //         cout<<c_tensor[i][j]<< " ";
    //     cout<<std::endl;
    // }
    // cout<<std::string(50,'=')<<endl;
    return 0;
}
