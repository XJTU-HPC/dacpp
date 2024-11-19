#include "DataReconstructor.h"
#include "sub_template.h"

void debug(int num){
    std::cout<<"ok"<<num<<"\n";
}
int main(){
    // 数据包装
    std::vector<int> dataB{1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16};
    std::vector<int> shapeB{4,4};
    dacpp::Tensor<int> matB(dataB,shapeB);
    matB.print();
    
    // 算子初始化
    RegularSlice si = RegularSlice("idx1", 2, 2);
	si.SetSplitSize(2);
	RegularSlice sj = RegularSlice("idx2", 2, 2);
	sj.SetSplitSize(2);
    RegularSlice sk = RegularSlice("idx3", 2, 2);
	sk.SetSplitSize(2);

    // 数据划分重组
    DataReconstructor<int> matB_tool;
    int* r_matB=(int*)malloc(sizeof(int)*16);
    Dac_Ops matB_ops;
    sk.setDimId(0);
    sk.setSplitLength(8);
    matB_ops.push_back(sk);
    sj.setDimId(1);
    sj.setSplitLength(4);
    matB_ops.push_back(sj);
    matB_tool.init(matB,matB_ops);
    matB_tool.Reconstruct(r_matB);
    
    std::cout <<  "加入sk、sj算子,，matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    
    // ------------------------------------------------------分割线，进入内层calc kernel-----------------------------------------------------------------------------
    
    // 算子初始化
    Index i = Index("idx4");
	i.SetSplitSize(2);
	Index j = Index("idx5");
	j.SetSplitSize(2);
    
    // 数据划分重组
    j.setDimId(1);
    j.setSplitLength(2);
    matB_tool.push_back(j);
    matB_tool.Reconstruct(r_matB);
    
    std::cout <<  "加入j算子，matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    // ------------------------------------------------------分割线，进入内层calc kernel-----------------------------------------------------------------------------

    // 算子初始化
    Index k = Index("idx6");
	k.SetSplitSize(2);

    // 数据划分重组
    k.setDimId(1);
    k.setSplitLength(1);
    matB_tool.push_back(k);
    matB_tool.Reconstruct(r_matB);

    std::cout <<  "加入k算子，matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }

    // ------------------------------------------------------分割线，退回层calc kernel-----------------------------------------------------------------------------
    
    matB_tool.pop_back();
    matB_tool.Reconstruct(r_matB);

    std::cout <<  "去掉k算子，matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    
    // ------------------------------------------------------分割线，退回层calc kernel-----------------------------------------------------------------------------
    matB_tool.pop_back();
    matB_tool.Reconstruct(r_matB);

    std::cout <<  "去掉i算子，matB重组结果:\n";
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            std::cout<<r_matB[i*4+j*1]<<" ";
        }
        std::cout<<std::endl;
    }
    
    return 0;
}