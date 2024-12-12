#include "ParameterGeneration.h"
#include "sub_template.h"

int main()
{
    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeA{4, 4};
    Tensor<int> matA(dataA, shapeA);

    std::vector<int> dataB{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeB{4, 4};
    Tensor<int> matB(dataB, shapeB);

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> shapeC{4, 4};
    Tensor<int> matC(dataC, shapeC);

    RegularSplit si("si", 2, 2);
    si.setDimId(0);//这个作用的维度是0吗？
    RegularSplit sj("sj", 2, 2);
    sj.setDimId(0);
    RegularSplit sk("sk", 2, 2);
    sk.setDimId(0);

    //下面测试设备内存分配大小的size
    int matA_size = para_gene_tool.init_device_memory_size(matA);
    std::cout << "matA_size:" << matA_size << std::endl; //应该输出16

    int matB_size = para_gene_tool.init_device_memory_size(matB);
    std::cout << "matB_size:" << matB_size << std::endl; //应该输出16

    int in_retults = para_gene_tool.init_device_memory_size()
}

    // dacpp::RegularSplit si("idx1", 2, 2);
    // dacpp::RegularSplit sj("idx2", 2, 2);
    // dacpp::RegularSplit sk("idx3", 2, 2);
    // dacpp::list dataList{matA[si][sk], matB[sk][sj], matC[si][sj]};