#include <iostream>
#include <string>
#include "/data/zjx/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"
#include "sub_template.h"

template <typename ImplType>
class ParameterGeneration
{
    public:
        ParameterGeneration(){

        }

        //生成算子的划分数 分区算子
        int init_regularslice_operetor_splitnumber(RegularSlice si,dacpp::Tensor<ImplType> tensor)
        {  
            int split_num = (tensor.getShape(si.dimId) - si.size) / si.stride + 1;
            return split_num;
        }

        //生成算子的划分数 降维算子
        int init_operetor_splitnumber(Index si,dacpp::Tensor<ImplType> tensor)
        {  
            int split_num = tensor.getShape(si.dimId); //算子作用维度的划分数
            return split_num;
        }     
};