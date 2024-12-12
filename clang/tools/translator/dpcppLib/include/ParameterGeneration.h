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
        int init_operetor_splitnumber(RegularSlice si,dacpp::Tensor<ImplType> tensor)
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

        //生成设备内存的分配大小
        int init_device_memory_size(dacpp::Tensor<ImplType> tensor)
        {
            return tensor.getSize();//数据的大小即为需要分配的设备内存大小
        }

        //生成设备内存的分配大小
        int init_device_memory_size(dacpp::Tensor<ImplType> tensor,Dac_Ops in_ops,Dac_Ops out_ops)
        {
            int in_op_product = 1;//输入算子划分数的乘积
            for(int i = 0;i < in_ops.size;i ++)
            {
                in_op_product *= init_operetor_splitnumber(in_ops[i]);
            }
            int out_op_product = 1;//输出算子划分数的乘积
            for(int i = 0;i < out_ops.size;i ++)
            {
                out_op_product *= init_operetor_splitnumber(out_ops[i]);
            }
            int multiple = in_op_product / out_op_product;//得到输入是输出的倍数
            return tensor.getSize() * multiple;
        }
};