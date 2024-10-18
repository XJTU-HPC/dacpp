#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<map>
#include "Tensor.hpp"
#include "dacInfo.h"
#include "sub_template.h"



struct Range
{
    int start;
    int end;
};

struct PosNumber
{
    std::vector<int> number;
    std::vector<int> pos;
};


/*
    数据重组器类，用于数据重组。
*/
template<typename ImplType>
class DataReconstructor{
    private:
        dacpp::Tensor<ImplType> myTensor;      // 数据
        Dac_Ops ops;                           // 作用于数据的算子
        int cnt;                               // 结果长向量的计数器
        std::vector<int> number;               // 存编号的中间变量
        std::vector<PosNumber> posNumberList;  // 编号与其对应的位置
        std::vector<int> pos;                  // 存位置的中间变量
    public:
        DataReconstructor(){

        }

        /*
            通过 数据内容，作用于数据的算子 初始化数据重组器
        */
        void init(dacpp::Tensor<ImplType> myTensor, Dac_Ops ops){
            this->myTensor=myTensor;
            this->ops=ops;
            GetPos(ops, 0);     
        }

        void GetPos(Dac_Ops ops, int now) {
            if (now == this->myTensor.getDim()) {
                // std::cout<<pos[0]<<" "<<pos[1]<<std::endl;
                GetPosNumber(this->pos, ops);
                return;
            }
            for (int i = 0; i < this->myTensor.getShape(now); i++) {
                this->pos.push_back(i);
                GetPos(ops, now + 1);
                this->pos.pop_back();
            }
        }

        void GetPosNumber(std::vector<int> pos, Dac_Ops ops) {
            int dimNum = this->myTensor.getDim();
            std::vector<Range> region;
            for (int i = 0; i < dimNum; i++) {
                Range r;
                r.start = 0;
                r.end = this->myTensor.getShape(i);
                region.push_back(r);
            }
            // std::cout<<pos[0]<<" "<<pos[1]<<std::endl;
            RecursiveTraversal(pos, region, 0);
        }

        void RecursiveTraversal(std::vector<int> pos, std::vector<Range> region, int now) {
            if (now == ops.size) {
                // std::cout<<pos[0]<<" "<<pos[1]<<std::endl;
                PosNumber posNum;
                posNum.number = this->number;
                posNum.pos = pos;
                this->posNumberList.push_back(posNum);
                return;
            }
            Dac_Op op = ops[now];
            int id = pos[op.dimId]-region[op.dimId].start;
            int l;
            if (id-op.size<0) l=0;
            else l = ((id-op.size+1)%op.stride==0) ? (id-op.size+1)/op.stride : ((id-op.size+1+op.stride)&(-op.stride))/op.stride;
            // while (region[op.dimId].start + l * op.stride + op.size <= id) l++;
            int r = (region[op.dimId].start+(id&(-op.stride))+op.size<region[op.dimId].end) ? (id&(-op.stride))/op.stride : (region[op.dimId].end-op.size-region[op.dimId].start)/op.stride;
            // while (region[op.dimId].start + r * op.stride > id) r--;
            // std::cout<<pos[0]<<" "<<pos[1]<<" "<<op.dimId<<" "<<id<<" "<<l<<" "<<r<<std::endl;
            for(int i = l; i <= r; i++) {
                this->number.push_back(i);
                region[op.dimId].start = region[op.dimId].start + i * op.stride;
                region[op.dimId].end = region[op.dimId].start + (i * op.stride + op.size);
                RecursiveTraversal(pos, region, now + 1);
                region[op.dimId].end = region[op.dimId].start - (i * op.stride + op.size);
                region[op.dimId].start = region[op.dimId].start - i * op.stride;
                this->number.pop_back();
            }
        }

        /*
            将特定位置元素写入res长向量。
        */
        void WriteData(ImplType* res, std::vector<int> pos) {
            res[this->cnt++]=this->myTensor.getElement(pos);
            // std::cout<<this->myTensor.getDim()<<" "<<pos.size()<<std::endl;
            // std::cout<<pos[0]<<" "<<pos[1]<<std::endl;
        }

        /*
            将重组结果写入res长向量。
        */
        void Reconstruct(ImplType* res){
            this->cnt=0;
            std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
            std::cout<<this->posNumberList.size()<<std::endl;
            for (int i=0; i<this->posNumberList.size(); i++) {
                this->WriteData(res,this->posNumberList[i].pos);
            }
        }
};