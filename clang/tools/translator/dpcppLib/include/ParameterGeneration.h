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



        //计算输入数量
        int init_Countin(Dac_Ops in_ops){
            int countIn = 1;
            std::string set;
            std::set<std::string> setIn;
            for(int i = 0; i < in_ops.size; i++){
                if(setIn.count(in_ops[i].name))
                    continue;
                countIn *= in_ops[i].split_size;
                setIn.insert(in_ops[i].name);
            }
            return countIn;
        }

        //计算输出数量
        int init_Countout(Dac_Ops out_ops){
            int countOut = 1;
            std::string set;
            std::set<std::string> setOut;
            for(int i = 0; i < out_ops.size; i++){
                if(setOut.count(out_ops[i].name))
                    continue;
                countOut *= out_ops[i].split_size;
                setOut.insert(out_ops[i].name);
            }
            return countOut;
        }

        //添加算子
        void init_Exop(Dac_Ops in_ops, Dac_Ops* out_ops){
            std::set<std::string> set;
            for(int i = 0; i < out_ops->size; i++){
                set.insert(out_ops->DacOps[i].name);
            }
            for(int i = 0; i < in_ops.size; i++){
                if(set.count(in_ops[i].name))
                    continue;
                out_ops->push_back(in_ops[i]);
            }
                for(int i = 0; i < out_ops->size; i++){
                    std::cout << out_ops->DacOps[i].name << std::endl;
                }
        }
        //计算内存分配
        int* init_mem(dacpp::Tensor<ImplType> tensor,std::vector<Dac_Ops> Params){
            int* mem = new int[Params.size()];
            for(int j = 0; j < Params.size(); j++) {
                mem[j] = 1;
                for(int k = 0; k < Params[j].size; k++) {
                    // 降维划分
                    if(Params[j].DacOps[k].type == "Index") {
                        mem[j] = tensor.getShape(k);
                    }
                    // 规则分区划分
                    else if(Params[j].DacOps[k].type == "RegularSlice") {
                        mem[j] *= (tensor.getShape(k) - Params[j].DacOps[k].split_size + Params[j].DacOps[k].stride) / Params[j].DacOps[k].stride * Params[j].DacOps[k].split_size;
                    }
                    // 保形划分
                    else {
                        mem[j] *= tensor.getShape(k);
                    }
                }
            }
            return mem;
        }

        
};