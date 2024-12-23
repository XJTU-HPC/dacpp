#include "/data/tqc/dacpp/clang/tools/translator/dacppLib/include/ReconTensor.h"
#include <iostream>
#include <vector>
//using dacpp::TensorProxy;
using dacpp::Tensor;

using std::vector;
using std::cout;
using std::endl;

int main() {
    vector<int> data{1,2,3,4,5,6,7,8,9};
    Tensor<int,2> u_tensor({3,3}, data);
    cout<<std::string(50,'=')<<endl;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++)
            cout<<u_tensor[i][j]<< " ";
        cout<<std::endl;
    }
    cout<<std::string(50,'=')<<endl;
    Tensor<int, 2> c_tensor = u_tensor[{1}][{}];
    //cout<<c_tensor.getOffset()<<" "<<c_tensor.getDim()<<endl;
    for(int i=0;i<c_tensor.getShape(0);i++){
        for(int j=0;j<c_tensor.getShape(1);j++)
            cout<<c_tensor[i][j]<< " ";
        cout<<std::endl;
    }

    //cout<<c_tensor[0][1]<<" "<<c_tensor[0][2]<<endl;
    cout<<std::string(50,'=')<<endl;
    // c_tensor = u_tensor[{0}][{0,2}];
    // for(int i=0;i<c_tensor.getShape(0);i++){
    //     for(int j=0;j<c_tensor.getShape(1);j++)
    //         cout<<c_tensor[i][j]<< " ";
    //     cout<<std::endl;
    // }
    // cout<<std::string(50,'=')<<endl;
    return 0;
}
