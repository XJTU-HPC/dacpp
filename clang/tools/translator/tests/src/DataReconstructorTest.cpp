#include "DataReconstructor.h"
#include "sub_template.h"

void debug(int num){
    std::cout<<"ok"<<num<<"\n";
}
int main(){
    int* res=(int*)malloc(sizeof(int)*16);
    
    std::vector<int> data{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    std::vector<int> shape{4,4};
    dacpp::Tensor<int> myTensor(data,shape);
    myTensor.print();
    RegularSlice si = RegularSlice("si", 2, 2);
	si.SetSplitSize(2);
	RegularSlice sj = RegularSlice("sj", 2, 2);
	sj.SetSplitSize(2);
    Dac_Ops data_ops;
    si.setDimId(0);
	si.setSplitLength(8);
	data_ops.push_back(si);
    sj.setDimId(1);
    sj.setSplitLength(4);
    data_ops.push_back(sj);
    DataReconstructor<int> tool;
    tool.init(myTensor,data_ops);
    tool.Reconstruct(res);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
  
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<res[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    return 0;
}