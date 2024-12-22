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

        // //生成算子的划分数 分区算子
        // int init_operetor_splitnumber(RegularSlice si,dacpp::Tensor<ImplType> tensor)
        // {  
        //     int split_num = (tensor.getShape(si.dimId) - si.size) / si.stride + 1;
        //     return split_num;
        // }

        // //生成算子的划分数 降维算子
        // int init_operetor_splitnumber(Index si,dacpp::Tensor<ImplType> tensor)
        // {  
        //     int split_num = tensor.getShape(si.dimId); //算子作用维度的划分数
        //     return split_num;
        // }

        //生成算子的划分数 不用区分分区算子和降维算子 直接用Dac_Op去计算就可以了
        int init_operetor_splitnumber(Dac_Op si,dacpp::Tensor<ImplType> tensor)
        {
            int split_num = (tensor.getShape(si.dimId) - si.size) / si.stride + 1;
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

        /*下面函数已废弃  接口太复杂了*/
        //生成设备内存的分配大小 支持情况：数据重组时中间需要的内存 内核函数中localsize的大小 为啥了不知道 反正就这样算的
        //传入四个参数 分别是输入的tensor组 输出的tensor 输入tensor的算子组的数组 输出tensor的算子组
        int init_device_memory_size(std::vector<dacpp::Tensor<ImplType>> tensor_in,dacpp::Tensor<ImplType> tensor_out,std::vector<Dac_Ops> ops_in,Dac_Ops ops_out)
        {
            int in_op_product = 1;//输入算子划分数的乘积
            std::unordered_set<std::string> mySet;//输入算子是否相同 现在只能拿名字去判断
            //下面遍历所有输入算子组里面的算子
            for(int i = 0;i < ops_in.size();i ++)
            {
                for(int j = 0;j < ops_in[i].size;j ++)
                {
                    if(mySet.find(ops_in[i][j].name) == mySet.end()) //如果这个算子不存在于集合中
                    {
                        mySet.insert(ops_in[i][j].name);//插入集合中去
                        in_op_product *= init_operetor_splitnumber(ops_in[i].DacOps[j],tensor_in[i]);
                    }
                }
            }

            int out_op_product = 1;//输出就一个计算比较简单
            for(int i = 0;i < ops_out.size;i ++)
            {
                out_op_product *= init_operetor_splitnumber(ops_out.DacOps[i],tensor_out);
            }
            return init_device_memory_size(tensor_out,ops_out) * in_op_product / out_op_product;
        }
        /*上面函数已废弃*/

        //生成设备内存的分配大小 支持情况：数据重组时中间需要的内存 内核函数中localsize的大小
        //Dac_Ops ops_in中所有非算子是不同的，由抽象语法树后端进行去重
        int init_device_memory_size(Dac_Ops ops_in,Dac_Ops ops_out,dacpp::Tensor<ImplType> tensor_out)
        {
            int in_op_product = 1;//输入算子划分数的乘积
            for(int i = 0;i < ops_in.size;i ++)
            {
                in_op_product *= ops_in.DacOps[i].split_size; //spilit在前面初始化算子的时候已经完成
            }
            int out_op_pruduct = 1;//输出算子划分数的乘积
            for(int i = 0;i < ops_out.size;i ++)
            {
                out_op_pruduct *= ops_out.DacOps[i].split_size; //spilit在前面初始化算子的时候已经完成
            }
            return init_device_memory_size(tensor_out,ops_out) * in_op_product / out_op_product;
        }

};