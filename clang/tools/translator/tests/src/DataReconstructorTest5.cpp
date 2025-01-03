#include "DataReconstructor.h"
#include "sub_template.h"

void debug(int num){
    std::cout<<"ok"<<num<<"\n";
}
int main(){
    std::vector<int> data{90,100,110,120,202,228,254,280,314,356,398,440,426,484,542,600};
    // std::vector<int> shape{4,4};
    dacpp::Tensor<int,2> matC({4,4},data);
    matC.print();

    DataInfo info_matC;
    info_matC.dim = matC.getDim();
    for(int i = 0; i < info_matC.dim; i++) info_matC.dimLength.push_back(matC.getShape(i));

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
    matC_tool.init(info_matC,matC_ops);
    matC_tool.Reconstruct(r_matC,matC);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
  
    for(int i=0;i<16;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<"\n";

    r_matC[3]=0;

    for(int i=0;i<16;i++) std::cout<<r_matC[i]<<" ";
    std::cout<<"\n";

    matC_tool.UpdateData(r_matC,matC);
    matC.print();

    return 0;
}