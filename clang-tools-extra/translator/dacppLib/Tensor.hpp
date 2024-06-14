#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <vector>
#include <iostream>

#include "Slice.h"

namespace dacpp {

template<typename ImplType>
class Tensor {
private:
    std::shared_ptr<ImplType> data_;
    int offset_;
    int dim_;
    int* shape_;
    int* stride_;

    void recursiveTake(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == dim_) {
            int index = offset_;
            for(int i = 0; i < dim_; i++) {
                index += indices[i] * stride_[i];
            }
            data[idx++] = data_.get()[index];
            return;
        }
        for(int i = 0; i < shape_[dimIdx]; i++) {
            indices.push_back(i);
            recursiveTake(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }

    void recursiveBring(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) {
        if(dimIdx == dim_) {
            int index = offset_;
            for(int i = 0; i < dim_; i++) {
                index += indices[i] * stride_[i];
            }
            data_.get()[index] = data[idx++];
            return;
        }
        for(int i = 0; i < shape_[dimIdx]; i++) {
            indices.push_back(i);
            recursiveBring(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }

    void recursivePrint(std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == dim_) {
            int index = offset_;
            for(int i = 0; i < dim_; i++) {
                index += indices[i] * stride_[i];
            }
            std::cout << data_.get()[index];
            return;
        }
        std::cout << "{";
        for(int i = 0; i < shape_[dimIdx]; i++) {
            indices.push_back(i);
            recursivePrint(indices, dimIdx + 1);
            if(i != shape_[dimIdx] - 1) std::cout << ", ";
            indices.pop_back();
        }
        std::cout << "}";
    }

public:
    Tensor() {
        
    }
    Tensor(ImplType* data, int size, int* shape, int dim) {
        // 参数检查
        // int count = 1;
        // for(int i = 0; i < dim; i++) count *= shape[i];
        // if(count != size) {}

        data_ = std::shared_ptr<ImplType>(new ImplType[size], std::default_delete<ImplType[]>());
        for(int i = 0; i < size; i++) {
            data_.get()[i] = data[i];
        }
        offset_ = 0;
        dim_ = dim;
        shape_ = new int[dim_];
        stride_ = new int[dim_];
        for(int i = dim_ - 1; i >= 0; i--) {
            shape_[i] = shape[i];
            if(i == dim_ - 1) stride_[i] = 1;
            else stride_[i] = stride_[i + 1] * shape[i + 1];
        }
    }

    Tensor(const std::vector<ImplType>& data, const std::vector<int>& shape) {
        // 参数检查
        // int count = 1;
        // for(int i = 0; i < shape.size(); i++) count *= shape[i];
        // if(count != data.size()) {}

        data_ = std::shared_ptr<ImplType>(new ImplType[data.size()], std::default_delete<ImplType[]>());
        for(int i = 0; i < data.size(); i++) {
            data_.get()[i] = data[i];
        }
        offset_ = 0;
        dim_ = shape.size();
        shape_ = new int[dim_];
        stride_ = new int[dim_];
        for(int i = dim_ - 1; i >= 0; i--) {
            shape_[i] = shape[i];
            if(i == dim_ - 1) stride_[i] = 1;
            else stride_[i] = stride_[i + 1] * shape[i + 1];
        }
    }

    Tensor(std::shared_ptr<ImplType> data, int offset, int dim, int* shape, int* stride) : data_(data), offset_(offset), dim_(dim), shape_(shape), stride_(stride) {}

    ~Tensor() {
        // delete[] shape_;
        // delete[] stride_;
    }

    std::shared_ptr<ImplType> getPoint() const {
        return data_;
    };

    int getOffset() const {
        return offset_;
    }
    // 获得 Tensor 的维度
    int getDim() const {
        return dim_;
    }

    // 获得 Tensor 某一维的大小
    // dimIdx：维度索引
    int getShape(int dimIdx) const {
        // 参数检查
        // if(dimIdx >= dim_) {}

        return shape_[dimIdx]; 
    }

    int getStride(int dimIdx) const {
        return stride_[dimIdx];
    }

    int getSize() const {
        int size = 1;
        for(int dimIdx = 0; dimIdx < getDim(); dimIdx++) {
            size *= shape_[dimIdx];
        }
        return size;
    }

    // 获得 Tensor 指定下标数据
    // indices：所有维度的下标组成的数组
    ImplType getData(int* indices) const {
        int index = offset_;
        for(int i = 0; i < dim_; i++) {
            index += indices[i] * stride_[i];
        }
        return data_.get()[index];
    }

    Tensor<ImplType> operator+(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] + operand.getData(indices);
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    // Tensor<ImplType> operator+(ImplType> operand) const {

    // }

    Tensor<ImplType> operator-(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] - operand.getData(indices);
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    Tensor<ImplType> operator*(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] * operand.getData(indices);
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    Tensor<ImplType> operator/(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] / operand.getData(indices);
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    Tensor<ImplType> operator%(const Tensor<ImplType>& operand) const {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            ImplType tmp = data_.get()[offset_] % operand.getData(indices);
            int* a = new ImplType[1];
            a[0] = tmp;
            return Tensor(a, 1, nullptr, 0);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator+=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] += operand.getData(indices);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator-=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] -= operand.getData(indices);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator*=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] *= operand.getData(indices);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator/=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] /= operand.getData(indices);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator%=(const Tensor<ImplType>& operand) {
        // // 参数检查
        // if(dim_ != operand.getDim()) {}
        // for(int i = 0; i < dim_; i++) {
        //     if(shape_[i] != operand.getShape(i)) {}
        // }
        if(dim_ == 0) {
            int* indices = new int[1];
            indices[0] = 0;
            data_.get()[offset_] %= operand.getData(indices);
        }
        else if(dim_ == 1) {

        }
        else if(dim_ == 2) {

        }
        else {

        }
    }

    void operator=(const Tensor<ImplType>& operand) {
        data_ = operand.getPoint();
        offset_ = operand.getOffset();
        dim_ = operand.getDim();
        shape_ = new int[dim_];
        stride_ = new int[dim_];
        for(int i = 0; i < dim_; i++) {
            shape_[i] = operand.getShape(i);
            stride_[i] = operand.getStride(i);
        }
    }

    Tensor<ImplType> operator[](int idx) const {
        return slice(0, idx);
    }

    Tensor<ImplType> operator[](Slice slc) const {
        if(slc.isAll_) {
            return slice(0, 0, shape_[0]);
        }
        else {
            return slice(0, slc.start_, slc.end_, slc.stride_);
        }
    }

    // Tensor 切片操作
    // dimIdx：进行切片的维度
    // idx：索引
    Tensor<ImplType> slice(int dimIdx, int idx) const {
        // 参数检查
        // if(dimIdx >= dim_ || idx >= shape_[dimIdx]) {}

        int offset = offset_ + idx * stride_[dimIdx];
        int dim = dim_ - 1;
        int* shape = new int[dim];
        int* stride = new int[dim];
        for(int i = 0; i < dim; i++) {
            if(i < dimIdx) {
                shape[i] = shape_[i];
                stride[i] = stride_[i];
            }
            else {
                shape[i] = shape_[i + 1];
                stride[i] = stride_[i + 1];
            }
        }
        return Tensor<ImplType>(data_, offset, dim, shape, stride);
    }

    // Tensor 切片操作
    // dimIdx：进行切片的维度
    // start：开始索引
    // end：结束索引
    // sliceStride：步长
    Tensor<ImplType> slice(int dimIdx, int start, int end, int sliceStride = 1) const {
        // 参数检查
        // if(dimIdx >= dim_ || start >= shape_[dimIdx] || end > shape_[dimIdx]) {}

        int offset = start * stride_[dimIdx];
        int dim = dim_;
        int* shape = new int[dim];
        int* stride = new int[dim];
        for(int i = 0; i < dim; i++) {
            shape[i] = shape_[i];
            stride[i] = stride_[i];
        }
        shape[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride[dimIdx] = stride_[dimIdx] * sliceStride;
        return Tensor<ImplType>(data_, offset, dim, shape, stride);
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

    // // 测试代码
    // void test() {
    //     std::cout << "数据引用次数" << data_.use_count() << "\n";
    //     std::cout << "偏移" << offset_ << "\n";
    //     std::cout << "维度" << dim_ << "\n";
    //     for(int i = 0; i < dim_; i++) {
    //         std::cout << shape_[i] << " ";
    //     }
    //     std::cout << "\n";
    //     for(int i = 0; i < dim_; i++) {
    //         std::cout << stride_[i] << " ";
    //     }
    //     std::cout << "\n";
    // }

};

} // namespace dacpp

#endif