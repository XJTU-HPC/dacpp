#include "DataReconstructor.h"
#include "sub_template.h"

void debug(int num){
    std::cout<<"ok"<<num<<"\n";
}
int main(){
    std::vector<int> data{90,100,110,120,202,228,254,280,314,356,398,440,426,484,542,600};
    std::vector<int> shape{4,4};
    dacpp::Tensor<int> matC(data,shape);
    matC.print();

    RegularSlice si = RegularSlice("si", 2, 2);
	si.SetSplitSize(2);
	RegularSlice sj = RegularSlice("sj", 2, 2);
	sj.SetSplitSize(2);
    
    DataReconstructor<int> matC_tool;
    int* r_matC=(int*)malloc(sizeof(int)*16);
    Dac_Ops matC_ops;
    si.setDimId(0);
    si.setSplitLength(8);
    matC_ops.push_back(si);
    sj.setDimId(1);
    sj.setSplitLength(4);
    matC_ops.push_back(sj);
    matC_tool.init(matC,matC_ops);
    matC_tool.Reconstruct(r_matC);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
  
    for(int i=0;i<16;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<"\n";

    r_matC[3]=0;

    for(int i=0;i<16;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<"\n";

    dacpp::Tensor<int> res=matC_tool.UpdateData(r_matC);
    res.print();

    return 0;
}