#include "ParameterGeneration.h"
#include "sub_template.h"

int main(){
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeA{4, 4};
    dacpp::Tensor<int> matA(dataA, shapeA);

    RegularSlice s1("s1", 2, 2);
    RegularSlice s2("s2", 2, 2);
    RegularSlice s3("s3", 2, 2);

    s1.SetSplitSize(2);
    s2.SetSplitSize(2);
    s3.SetSplitSize(2);

    Dac_Ops in_ops;
    Dac_Ops out_ops;

    std::vector<Dac_Ops> Param;
    Dac_Ops p1;
    p1.push_back(s1);
    p1.push_back(s3);
    Param.push_back(p1);
    Dac_Ops p2;
    p2.push_back(s3);
    p2.push_back(s2);
    Param.push_back(p2);
    Dac_Ops p3;
    p3.push_back(s1);
    p3.push_back(s2);
    Param.push_back(p3);

    in_ops.push_back(s1);
    in_ops.push_back(s2);
    in_ops.push_back(s3);

    out_ops.push_back(s1);
    out_ops.push_back(s3);

    ParameterGeneration<int> para_gene_tool;

    std::cout << "Countin :" << para_gene_tool.init_Countin(in_ops) << " its should be 8" << std::endl;
    std::cout << "Countout :" << para_gene_tool.init_Countout(out_ops) << " its should be 4" << std::endl;
    
    para_gene_tool.init_Exop(in_ops,&out_ops);
    for(int i = 0; i < out_ops.size; i++){
        std::cout << out_ops[i].name << std::endl;
    }

    int *mem = para_gene_tool.init_mem(matA,Param);
    for(int i = 0; i < Param.size(); i++){
        std::cout << mem[i] << std::endl;
    }
}