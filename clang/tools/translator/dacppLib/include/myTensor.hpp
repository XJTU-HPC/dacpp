#ifndef MYTENSOR_HPP
#define MYTENSOR_HPP

#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "Slice.h"

namespace dacpp {

template<class ImplType, int N>
class Tensor {
private:
    std::shared_ptr<ImplType> data_;
    int offset_;
    int dim_;
    std::shared_ptr<int> shape_;
    std::shared_ptr<int> stride_;

    void recursiveTake(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == dim_) {
            int index = offset_;
            for(int i = 0; i < dim_; i++) {
                index += indices[i] * stride_.get()[i];
            }
            data[idx++] = data_.get()[index];
            return;
        }
        for(int i = 0; i < shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveTake(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }

    void recursiveBring(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) {
        if(dimIdx == dim_) {
            int index = offset_;
            for(int i = 0; i < dim_; i++) {
                index += indices[i] * stride_.get()[i];
            }
            data_.get()[index] = data[idx++];
            return;
        }
        for(int i = 0; i < shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveBring(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }

    void recursivePrint(std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == dim_) {
            int index = offset_;
            for(int i = 0; i < dim_; i++) {
                index += indices[i] * stride_.get()[i];
            }
            std::cout << data_.get()[index];
            return;
        }
        std::cout << "{";
        for(int i = 0; i < shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursivePrint(indices, dimIdx + 1);
            if(i != shape_.get()[dimIdx] - 1) std::cout << ", ";
            indices.pop_back();
        }
        std::cout << "}";
    }

public:
    Tensor() {}
    Tensor(const std::initializer_list<int> values, const ImplType* data, int len){
        if(N != values.size())
            throw std::invalid_argument("N must be equal to the size of list.");
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        if(len != ElementSize)
            throw std::invalid_argument("The size of array must be equal to the product of list");
        data_ = std::shared_ptr<ImplType>
            (new ImplType[ElementSize], std::default_delete<ImplType[]>());
        for(int i = 0; i < ElementSize; i++)
            data_.get()[i] = data[i];
        offset_ = 0;
        dim_ = values.size();
        shape_ = std::shared_ptr<int>(new int[dim_], std::default_delete<int[]>());
        stride_ = std::shared_ptr<int>(new int[dim_], std::default_delete<int[]>());
        auto it = values.end();
        it--;
        for(int idx = dim_ - 1; idx >= 0; idx--) {
            shape_.get()[idx] = *it;
            if(idx == dim_ - 1) 
                stride_.get()[idx] = 1;
            else 
                stride_.get()[idx] = stride_.get()[idx + 1] * shape_.get()[idx + 1];
            it--;
        }
    }
    Tensor(const std::initializer_list<int> values, const std::vector<ImplType> data){
        if(N != values.size())
            throw std::invalid_argument("N must be equal to the size of list.");
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        if(data.size() != ElementSize)
            throw std::invalid_argument("The size of vector must be equal to the product of list");
        data_ = std::shared_ptr<ImplType>
            (new ImplType[data.size()], std::default_delete<ImplType[]>());
        for(int i = 0; i < data.size(); i++)
            data_.get()[i] = data[i];
        offset_ = 0;
        dim_ = values.size();
        shape_ = std::shared_ptr<int>(new int[dim_], std::default_delete<int[]>());
        stride_ = std::shared_ptr<int>(new int[dim_], std::default_delete<int[]>());
        auto it = values.end();
        it--;
        for(int idx = dim_ - 1; idx >= 0; idx--) {
            shape_.get()[idx] = *it;
            if(idx == dim_ - 1) 
                stride_.get()[idx] = 1;
            else 
                stride_.get()[idx] = stride_.get()[idx + 1] * shape_.get()[idx + 1];
            it--;
        }
        // for(int i = 0; i < data.size(); i++)
        //     std::cout<< data[i];
    }
    Tensor(const std::initializer_list<int> values, ImplType data = 0){
        if(N != values.size())
            throw std::invalid_argument("N must be equal to the size of list.");
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        data_ = std::shared_ptr<ImplType>
            (new ImplType[ElementSize], std::default_delete<ImplType[]>());
        for(int i = 0; i < ElementSize; i++)
            data_.get()[i] = data;
        offset_ = 0;
        dim_ = values.size();
        shape_ = std::shared_ptr<int>(new int[dim_], std::default_delete<int[]>());
        stride_ = std::shared_ptr<int>(new int[dim_], std::default_delete<int[]>());
        auto it = values.end();
        it--;
        for(int idx = dim_ - 1; idx >= 0; idx--) {
            shape_.get()[idx] = *it;
            if(idx == dim_ - 1) 
                stride_.get()[idx] = 1;
            else 
                stride_.get()[idx] = stride_.get()[idx + 1] * shape_.get()[idx + 1];
            it--;
        }
    }


    Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride) 
    : data_(data), offset_(offset), dim_(dim), shape_(shape), stride_(stride) {}

    ~Tensor() {}

    std::shared_ptr<ImplType> getDataPtr() const {
        return data_;
    }

    int getOffset() const {
        return offset_;
    }


    int getDim() const {
        return dim_;
    }

    std::shared_ptr<int> getShapePtr() const {
        return shape_;
    }

    std::shared_ptr<int> getStridePtr() const {
        return stride_;
    }

    int getShape(int dimIdx) const {
        return shape_.get()[dimIdx]; 
    }

    int getStride(int dimIdx) const {
        return stride_.get()[dimIdx];
    }

    int getSize() const {
        int size = 1;
        for(int dimIdx = 0; dimIdx < getDim(); dimIdx++) {
            size *= shape_.get()[dimIdx];
        }
        return size;
    }

    Tensor<ImplType, N> operator+(const Tensor<ImplType, N>& operand) const {

    }

    Tensor<ImplType, N> operator-(const Tensor<ImplType, N>& operand) const {

    }

    Tensor<ImplType, N> operator*(const Tensor<ImplType, N>& operand) const {

    }

    Tensor<ImplType, N> operator/(const Tensor<ImplType, N>& operand) const {

    }

    Tensor<ImplType, N> operator%(const Tensor<ImplType, N>& operand) const {

    }

    void operator+=(const Tensor<ImplType, N>& operand) {

    }

    void operator-=(const Tensor<ImplType, N>& operand) {

    }

    void operator*=(const Tensor<ImplType, N>& operand) {

    }

    void operator/=(const Tensor<ImplType, N>& operand) {

    }

    void operator%=(const Tensor<ImplType, N>& operand) {

    }

    void operator=(const Tensor<ImplType, N>& operand) {
        data_ = operand.getDataPtr();
        offset_ = operand.getOffset();
        dim_ = operand.getDim();
        shape_ = operand.getShapePtr();
        stride_ = operand.getStridePtr();
    }

    Tensor<ImplType, N-1> operator[](int idx) const {
        return slice(0, idx);
    }

    // Tensor 切片操作
    // dimIdx：进行切片的维度
    // idx：索引
    Tensor<ImplType, N-1> slice(int dimIdx, int idx) const {
        // 参数检查
        // if(dimIdx >= dim_ || idx >= shape_.get()[dimIdx]) {}

        int offset = offset_ + idx * stride_.get()[dimIdx];
        int dim = dim_ - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = shape_.get()[idx];
                stride.get()[idx] = stride_.get()[idx];
            }
            else {
                shape.get()[idx] = shape_.get()[idx + 1];
                stride.get()[idx] = stride_.get()[idx + 1];
            }
        }
        return Tensor<ImplType, N - 1>(data_, offset, dim, shape, stride);
    }

    // 获取 Tensor 中的数据，用基本类型数组保存
    // data：基本类型数组
    void tensor2Array(ImplType* data) const {
        std::vector<int> indices;
        int idx = 0;
        recursiveTake(data, idx, indices, 0);
    }
    // 改变 Tensor 中的数据
    // data：基本类型数组
    void array2Tensor(ImplType* data) {
        std::vector<int> indices;
        int idx = 0;
        recursiveBring(data, idx, indices, 0);
    }

    void print() const {
        std::vector<int> indices;
        recursivePrint(indices, 0);
        std::cout << "\n";
    }
};

template<class ImplType>
class Tensor <ImplType, 1>{
public:
    Tensor(){}
    ~Tensor(){}
    Tensor(std::vector<ImplType> vec) : data_(vec) {
        this->shape_ = vec.size();
    }
    Tensor(std::initializer_list<ImplType> init) : data_(init) {
        this->shape_ = init.size();
    }
    Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride) {
        this->data_ = std::vector<ImplType> (shape.get()[0]);
        int index = offset;
        for(int i = 0; i < shape.get()[0]; i++){
            this->data_[i] = data.get()[index];
            index += stride.get()[0];
        }
        this->dim_ = 1;
        this->shape_ = shape.get()[0];
    }
    ImplType operator[](int idx) const {
        return this->data_[idx];
    }
    std::vector<ImplType>& getData(){
        return this->data_;
    }
    void print(){
        std::cout<<"{";
        for(int i=0;i<data_.size() - 1;i++) std::cout<<data_[i]<<", ";
        std::cout<<data_[data_.size() - 1]<<"}\n";
    }

    Tensor<ImplType, 1> operator+(const Tensor<ImplType, 1>& operand) const {

    }

    Tensor<ImplType, 1> operator-(const Tensor<ImplType, 1>& operand) const {

    }

    Tensor<ImplType, 1> operator*(const Tensor<ImplType, 1>& operand) const {

    }

    Tensor<ImplType, 1> operator/(const Tensor<ImplType, 1>& operand) const {

    }

    Tensor<ImplType, 1> operator%(const Tensor<ImplType, 1>& operand) const {

    }

    void operator+=(const Tensor<ImplType, 1>& operand) {

    }

    void operator-=(const Tensor<ImplType, 1>& operand) {

    }

    void operator*=(const Tensor<ImplType, 1>& operand) {

    }

    void operator/=(const Tensor<ImplType, 1>& operand) {

    }

    void operator%=(const Tensor<ImplType, 1>& operand) {

    }

private:
    std::vector<ImplType> data_;
    int offset_ = 0;
    int dim_ = 1;
    int stride_ = 1;
    int shape_;
};


} // namespace dacpp

#endif