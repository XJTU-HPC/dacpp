#include <iostream>
#include <string>
#include <unordered_set>
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

        //生成设备内存的分配大小 支持情况：mat[分区][分区] mat[分区][降维] mat[分区][] mat[降维][]
        int init_device_memory_size(dacpp::Tensor<ImplType> tensor,Dac_Ops ops)
        {
            int result = 1;//初始化结果为1 
            std::unordered_set<int> mySet;//用来存储算子作用的维度
            for(int i = 0;i < ops.size;i ++)
            {
                int dimId = ops[i].dimId;//拿到算子的维度
                mySet.insert(dimId);//将算子作用的维度放入集合中
                int split_num = (tensor.getShape(dimId) - ops[i].size) / ops[i].stride + 1;//计算算子的划分数
                int length = split_num * ops[i].size;//划分数乘以划分的大小
                result *= length;
            }
            for(int i = 0;i < tensor.getDim();i ++)
            {
                if (mySet.find(i) == mySet.end()) 
                {
                    //i不存在也就是说算子没有作用这个维度 这个维度是保型算子
                    result *= tensor.getShape(i);
                }
            }
            return result;
        }

        //生成设备内存分配大小 支持情况：mat[][]
        int init_device_memory_size(dacpp::Tensor<ImplType> tensor)
        {
            return tensor.getSize();
        }

        //生成设备内存的分配大小 下面这个是不对的
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