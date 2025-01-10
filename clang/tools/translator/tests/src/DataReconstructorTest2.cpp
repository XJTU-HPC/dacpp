#include "DataReconstructor.h"
#include "sub_template.h"

void debug(int num){
    std::cout<<"ok"<<num<<"\n";
}
int main(){
    int* res=(int*)malloc(sizeof(int)*36);
    
    std::vector<int> data{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    // std::vector<int> shape{4,5};
    dacpp::Tensor<int,2> myTensor({4,5},data);
    myTensor.print();

    DataInfo info_myTensor;
    info_myTensor.dim = myTensor.getDim();
    for(int i = 0; i < info_myTensor.dim; i++) info_myTensor.dimLength.push_back(myTensor.getShape(i));

    RegularSlice si = RegularSlice("si", 2, 2);
	si.SetSplitSize(2);
	RegularSlice sj = RegularSlice("sj", 3, 1);
	sj.SetSplitSize(3);
    Dac_Ops data_ops;
    si.setDimId(0);
	si.setSplitLength(10);
	data_ops.push_back(si);
    sj.setDimId(1);
    sj.setSplitLength(6);
    data_ops.push_back(sj);
    DataReconstructor<int> tool;
    tool.init(info_myTensor,data_ops);
    tool.Reconstruct(res,myTensor);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
  
    for(int i=0;i<36;i++){
        if(i%6==0) std::cout<<"\n";
        std::cout<<res[i]<<" ";
    }
    std::cout<<"\n";
    return 0;
}