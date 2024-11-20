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
    Index i = Index("i");
	i.SetSplitSize(4);
	Index j = Index("j");
	j.SetSplitSize(4);
    Dac_Ops data_ops;
    i.setDimId(0);
	i.setSplitLength(4);
	data_ops.push_back(i);
    j.setDimId(1);
    j.setSplitLength(1);
    data_ops.push_back(j);
    DataReconstructor<int> tool;
    tool.init(myTensor,data_ops);
    tool.Reconstruct(res);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
  
    for(int i=0;i<16;i++){
        if(i%4==0) std::cout<<"\n";
        std::cout<<res[i]<<" ";
    }
    std::cout<<"\n";

    int* res2=(int*)malloc(sizeof(int)*16);
    data_ops.clear();
    j.setDimId(1);
    j.setSplitLength(1);
    data_ops.push_back(j);
    DataReconstructor<int> tool2;
    tool2.init(myTensor,data_ops);
    tool2.Reconstruct(res2);        

    for(int i=0;i<16;i++){
        if(i%4==0) std::cout<<"\n";
        std::cout<<res2[i]<<" ";
    }
    std::cout<<"\n";
    return 0;
}