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

    Index i = Index("i");
    i.SetSplitSize(4);
    Index j = Index("j");
    j.SetSplitSize(4);

    // matA[si][] -> matA[i][]
    Dac_Ops matA_ops;
    si.setDimId(0);
	si.setSplitLength(8);
	matA_ops.push_back(si);
    i.setDimId(0);
    i.setSplitLength(4);
    matA_ops.push_back(i);
    DataReconstructor<int> tool;
    tool.init(myTensor,matA_ops);
    tool.Reconstruct(res);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
  
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<res[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    // matB[][sj] -> matA[][sj]
    Dac_Ops matB_ops;
    sj.setDimId(1);
	sj.setSplitLength(8);
	matB_ops.push_back(sj);
    j.setDimId(1);
    j.setSplitLength(1);
    matB_ops.push_back(j);

    DataReconstructor<int> tool2;
    tool2.init(myTensor,matB_ops);
    int* res2=(int*)malloc(sizeof(int)*16);
    tool2.Reconstruct(res2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
  
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<res2[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    return 0;
}