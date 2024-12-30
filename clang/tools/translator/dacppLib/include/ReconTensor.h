#ifndef RECONTENSOR_H
#define RECONTENSOR_H

#include <algorithm>
#include <iostream>
#include <vector>
#include <exception>
#include <memory>
#include "Slice.h"
#include "FuncTensor.hpp"



namespace dacpp {
    
    template <class T, int N>
    class TensorProxy;
    template <class T>
    class TensorProxy <T, 1>;

    template<class ImplType>
    class TensorBase{
    public:
        operator FuncTensor<ImplType>() const {
            return FuncTensor<ImplType>(this->data_, this->offset_, this->dim_, this->shape_, this->stride_);
        };
        std::shared_ptr<ImplType> getDataPtr() const;
        int getOffset() const;
        int getDim() const;
        std::shared_ptr<int> getShapePtr() const;
        std::shared_ptr<int> getStridePtr() const;
        int getShape(int dimIdx) const;
        int getStride(int dimIdx) const;
        int getSize() const;
        void tensor2Array(ImplType* data) const;
        void array2Tensor(ImplType* data);
        void print() const;
        int getCurrentDim() const;
    protected:
        void NextDim();
        void recursiveTake(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) const;
        void recursiveBring(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx);
        void recursivePrint(std::vector<int>& indices, int dimIdx) const;
        std::shared_ptr<ImplType> data_;
        int offset_;
        int dim_;
        std::shared_ptr<int> shape_;
        std::shared_ptr<int> stride_;
        int current_dim = 0;
    };

    template<class ImplType, int N>
    class Tensor: public TensorBase <ImplType>{
    public:
        Tensor(const TensorProxy<ImplType, N> &x);
        Tensor(const Tensor<ImplType, N> &x);
        Tensor(const std::vector<int> values, const ImplType* data);
        Tensor(const std::vector<int> values, const std::vector<ImplType> data);
        Tensor(const std::vector<int> values, ImplType data = 0);
        Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride);
        Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim);
        Tensor<ImplType, N>& operator=(const Tensor<ImplType, N>& operand);
        TensorProxy<ImplType, N> operator[](std::initializer_list<int> idx);
        TensorProxy<ImplType, N-1> operator[](int idx) const;
        TensorProxy<ImplType, N> operator[](RegularSplit sp) const;
        TensorProxy<ImplType, N - 1> operator[](Index sp) const;
        Tensor<ImplType, N-1> slice(int dimIdx, int idx) const;
        Tensor<ImplType, N> slice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const;
        Tensor<ImplType, N> operator+(const Tensor<ImplType, N>& operand) const{};
        Tensor<ImplType, N> operator-(const Tensor<ImplType, N>& operand) const{};
        Tensor<ImplType, N> operator*(const Tensor<ImplType, N>& operand) const{};
        Tensor<ImplType, N> operator/(const Tensor<ImplType, N>& operand) const{};
        Tensor<ImplType, N> operator%(const Tensor<ImplType, N>& operand) const{};
        void operator+=(const Tensor<ImplType, N>& operand){};
        void operator-=(const Tensor<ImplType, N>& operand){};
        void operator*=(const Tensor<ImplType, N>& operand){};
        void operator/=(const Tensor<ImplType, N>& operand){};
        void operator%=(const Tensor<ImplType, N>& operand){};
    };

    template<class ImplType>
    class Tensor <ImplType, 1> : public TensorBase<ImplType>{
    private:
        template <class InputIt>
        void initialize(InputIt first, InputIt last);
    public:
        Tensor(const TensorProxy<ImplType, 1> &x);
        Tensor(std::vector<ImplType> init);
        Tensor(int len, const ImplType *init);
        Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride);
        Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim);
        Tensor& operator=(const Tensor<ImplType, 1>& operand);
        TensorProxy<ImplType, 1> operator[](std::initializer_list<int> idx);
        ImplType& operator[](int idx);
        TensorProxy<ImplType, 1> operator[](RegularSplit sp) const;
        ImplType& operator[](Index sp) const;
        ImplType& slice(int dimIdx, int idx) const;
        Tensor<ImplType, 1> slice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const;
        Tensor<ImplType, 1> operator+(const Tensor<ImplType, 1>& operand) const{};
        Tensor<ImplType, 1> operator-(const Tensor<ImplType, 1>& operand) const{};
        Tensor<ImplType, 1> operator*(const Tensor<ImplType, 1>& operand) const{};
        Tensor<ImplType, 1> operator/(const Tensor<ImplType, 1>& operand) const{};
        Tensor<ImplType, 1> operator%(const Tensor<ImplType, 1>& operand) const{};
        void operator+=(const Tensor<ImplType, 1>& operand){};
        void operator-=(const Tensor<ImplType, 1>& operand){};
        void operator*=(const Tensor<ImplType, 1>& operand){};
        void operator/=(const Tensor<ImplType, 1>& operand){};
        void operator%=(const Tensor<ImplType, 1>& operand){};
    };

    template<class ImplType>
    class ProxyBase{
    protected:
        std::shared_ptr<ImplType> getDataPtr() const;
        int getOffset() const;
        int getDim() const;
        std::shared_ptr<int> getShapePtr() const;
        std::shared_ptr<int> getStridePtr() const;
        int getShape(int dimIdx) const;
        int getStride(int dimIdx) const;
        int getSize() const;
        int getCurrentDim() const;
        void NextDim();
        std::shared_ptr<ImplType> data_;
        int offset_;
        int dim_;
        std::shared_ptr<int> shape_;
        std::shared_ptr<int> stride_;
        int current_dim = 0;
    };

    template<class ImplType, int N>
    class TensorProxy: public ProxyBase <ImplType>{
        friend class dacpp::Tensor<ImplType, N>;
        friend class dacpp::Tensor<ImplType, 1>;
    public:
        TensorProxy(dacpp::Tensor<ImplType, N + 1> tensor, int dimIdx, int idx);
        TensorProxy(dacpp::Tensor<ImplType, N> tensor, int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0);
        TensorProxy(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim);
        TensorProxy<ImplType, N> operator[](std::initializer_list<int> idx);
        TensorProxy<ImplType, N-1> operator[](int idx) const;
        TensorProxy<ImplType, N> operator[](RegularSplit sp) const;
        TensorProxy<ImplType, N - 1> operator[](Index sp) const;
        TensorProxy<ImplType, N-1> Pslice(int dimIdx, int idx) const;
        TensorProxy<ImplType, N> Pslice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const;
        TensorProxy<ImplType, N> operator+(const TensorProxy<ImplType, N>& operand) const{};
        TensorProxy<ImplType, N> operator-(const TensorProxy<ImplType, N>& operand) const{};
        TensorProxy<ImplType, N> operator*(const TensorProxy<ImplType, N>& operand) const{};
        TensorProxy<ImplType, N> operator/(const TensorProxy<ImplType, N>& operand) const{};
        TensorProxy<ImplType, N> operator%(const TensorProxy<ImplType, N>& operand) const{};
        void operator+=(const TensorProxy<ImplType, N>& operand){};
        void operator-=(const TensorProxy<ImplType, N>& operand){};
        void operator*=(const TensorProxy<ImplType, N>& operand){};
        void operator/=(const TensorProxy<ImplType, N>& operand){};
        void operator%=(const TensorProxy<ImplType, N>& operand){};
    };

    template<class ImplType>
    class TensorProxy <ImplType, 1> : public ProxyBase<ImplType>{
        friend class dacpp::Tensor<ImplType, 2>;
        friend class dacpp::Tensor<ImplType, 1>;
    public:
        TensorProxy(dacpp::Tensor<ImplType, 2> tensor, int dimIdx, int idx);
        TensorProxy(dacpp::Tensor<ImplType, 1> tensor, int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0);
        TensorProxy(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim);
        TensorProxy<ImplType, 1> operator[](std::initializer_list<int> idx);
        ImplType& operator[](int idx);
        TensorProxy<ImplType, 1> operator[](RegularSplit sp)const;
        ImplType& operator[](Index sp) const;
        ImplType& Pslice(int dimIdx, int idx) const;
        TensorProxy<ImplType, 1> Pslice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const;
        TensorProxy<ImplType, 1> operator+(const TensorProxy<ImplType, 1>& operand) const{};
        TensorProxy<ImplType, 1> operator-(const TensorProxy<ImplType, 1>& operand) const{};
        TensorProxy<ImplType, 1> operator*(const TensorProxy<ImplType, 1>& operand) const{};
        TensorProxy<ImplType, 1> operator/(const TensorProxy<ImplType, 1>& operand) const{};
        TensorProxy<ImplType, 1> operator%(const TensorProxy<ImplType, 1>& operand) const{};
        void operator+=(const TensorProxy<ImplType, 1>& operand){};
        void operator-=(const TensorProxy<ImplType, 1>& operand){};
        void operator*=(const TensorProxy<ImplType, 1>& operand){};
        void operator/=(const TensorProxy<ImplType, 1>& operand){};
        void operator%=(const TensorProxy<ImplType, 1>& operand){};
    };

    template <class T>
    using Vector = Tensor<T, 1>;
    template <class T>
    using Matrix = Tensor<T, 2>;

} // namespace dacpp

namespace dacpp {
    #define Base TensorBase<ImplType>
    
    template <class ImplType>
    int Base :: getStride(int dimIdx) const {return this->stride_.get()[dimIdx];}
    template <class ImplType>
    int Base :: getShape(int dimIdx) const {return this->shape_.get()[dimIdx]; }
    template <class ImplType>
    std::shared_ptr<int> Base :: getStridePtr() const {return this->stride_;}
    template <class ImplType>
    std::shared_ptr<ImplType> Base :: getDataPtr() const{return this->data_;}
    template <class ImplType>
    std::shared_ptr<int> Base :: getShapePtr() const {return this->shape_;}
    template <class ImplType>
    int Base :: getOffset() const {return this->offset_;}
    template <class ImplType>
    int Base :: getDim() const {return this->dim_;}

    template<class ImplType>
    int Base :: getSize() const {
        int size = 1;
        for(int dimIdx = 0; dimIdx < getDim(); dimIdx++) {
            size *= this->shape_.get()[dimIdx];
        }
        return size;
    }
    template<class ImplType>
    void Base :: tensor2Array(ImplType* data) const {
        std::vector<int> indices;
        int idx = 0;
        recursiveTake(data, idx, indices, 0);
    }
    template<class ImplType>
    void Base :: array2Tensor(ImplType* data) {
        std::vector<int> indices;
        int idx = 0;
        recursiveBring(data, idx, indices, 0);
    }
    template<class ImplType>
    void Base :: print() const {
        std::vector<int> indices;
        recursivePrint(indices, 0);
        std::cout << "\n";
    }
    template<class ImplType>
    int Base :: getCurrentDim() const {
        return this->current_dim;
    }
    template<class ImplType>
    void Base :: NextDim(){
        this->current_dim ++;
    }
    template<class ImplType>
    void Base :: recursiveTake(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) {
                index += indices[i] * this->stride_.get()[i];
            }
            data[idx++] = this->data_.get()[index];
            return;
        }
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveTake(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }
    template<class ImplType>
    void Base :: recursiveBring(ImplType* data, int& idx, std::vector<int>& indices, int dimIdx) {
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) {
                index += indices[i] * this->stride_.get()[i];
            }
            this->data_.get()[index] = data[idx++];
            return;
        }
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveBring(data, idx, indices, dimIdx + 1);
            indices.pop_back();
        }
    }
    template<class ImplType>
    void Base :: recursivePrint(std::vector<int>& indices, int dimIdx) const {
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) {
                index += indices[i] * this->stride_.get()[i];
            }
            std::cout << this->data_.get()[index];
            return;
        }
        std::cout << "{";
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursivePrint(indices, dimIdx + 1);
            if(i != this->shape_.get()[dimIdx] - 1) std::cout << ", ";
            indices.pop_back();
        }
        std::cout << "}";
    }

    #undef Base

    #define Base Tensor<ImplType, N>

    template<class ImplType, int N>
    Base :: Tensor(const TensorProxy<ImplType, N> &x){
        this->data_ = x.getDataPtr();
        this->offset_ = x.getOffset();
        this->dim_ = x.getDim();
        this->shape_ = x.getShapePtr();
        this->stride_ = x.getStridePtr();
        this->current_dim = 0;
    }
    template<class ImplType, int N>
    Base :: Tensor(const Tensor<ImplType, N> &x){
        this->data_ = x.data_;
        this->offset_ = x.offset_;
        this->dim_ = x.dim_;
        this->shape_ = x.shape_;
        this->stride_ = x.stride_;
        this->current_dim = 0;
    }
    template<class ImplType, int N>
    Base :: Tensor(const std::vector<int> values, const ImplType* data){
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[ElementSize], std::default_delete<ImplType[]>());
        for(int i = 0; i < ElementSize; i++)
            this->data_.get()[i] = data[i];
        this->offset_ = 0;
        this->dim_ = values.size();
        this->shape_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        auto it = values.end();
        it--;
        for(int idx = this->dim_ - 1; idx >= 0; idx--) {
            this->shape_.get()[idx] = *it;
            if(idx == this->dim_ - 1) 
                this->stride_.get()[idx] = 1;
            else 
                this->stride_.get()[idx] = this->stride_.get()[idx + 1] * this->shape_.get()[idx + 1];
            it--;
        }
    }
    template<class ImplType, int N>
    Base :: Tensor(const std::vector<int> values, const std::vector<ImplType> data){
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        if(ElementSize != data.size())  
            throw std::runtime_error("The number of elements in the vector does not correspond to the Shape.");
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[data.size()], std::default_delete<ImplType[]>());
        for(size_t i = 0; i < data.size(); i++)
            this->data_.get()[i] = data[i];
        this->offset_ = 0;
        this->dim_ = values.size();
        this->shape_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        auto it = values.end();
        it--;
        for(int idx = this->dim_ - 1; idx >= 0; idx--) {
            this->shape_.get()[idx] = *it;
            if(idx == this->dim_ - 1) 
                this->stride_.get()[idx] = 1;
            else 
                this->stride_.get()[idx] = this->stride_.get()[idx + 1] * this->shape_.get()[idx + 1];
            it--;
        }
    }
    template<class ImplType, int N>
    Base :: Tensor(const std::vector<int> values, ImplType data){
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[ElementSize], std::default_delete<ImplType[]>());
        for(int i = 0; i < ElementSize; i++)
            this->data_.get()[i] = data;
        this->offset_ = 0;
        this->dim_ = values.size();
        this->shape_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        auto it = values.end();
        it--;
        for(int idx = this->dim_ - 1; idx >= 0; idx--) {
            this->shape_.get()[idx] = *it;
            if(idx == this->dim_ - 1) 
                this->stride_.get()[idx] = 1;
            else 
                this->stride_.get()[idx] = this->stride_.get()[idx + 1] * this->shape_.get()[idx + 1];
            it--;
        }
    }
    template<class ImplType, int N>
    Base :: Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
    }
    template<class ImplType, int N>
    Base :: Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    template<class ImplType, int N>
    Tensor<ImplType, N>& Base :: operator=(const Tensor<ImplType, N>& operand) {
        this->data_ = operand.getDataPtr();
        this->offset_ = operand.getOffset();
        this->dim_ = operand.getDim();
        this->shape_ = operand.getShapePtr();
        this->stride_ = operand.getStridePtr();
        this->current_dim = 0;
        return *this;
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Base :: operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0){
            return TensorProxy<ImplType, N>(*this, this->current_dim, 0, this->shape_.get()[this->current_dim], 1, 1);
        }
        else if(idx.size() == 1){
            const int i = *(idx.begin());
            return TensorProxy<ImplType, N>(*this, this->current_dim, i, i + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            return TensorProxy<ImplType, N>(*this, this->current_dim, start, end, stride, 1);
        }
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N-1> Base :: operator[](int idx) const {
        return TensorProxy<ImplType, N - 1>(*this, this->current_dim, idx);
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Base :: operator[](RegularSplit sp) const {
        return TensorProxy<ImplType, N>(*this, 0, 0, this->shape_.get()[this->current_dim], 1, 0);
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N - 1> Base :: operator[](Index sp) const {
        return TensorProxy<ImplType, N-1>();
    }
    template<class ImplType, int N>
    Tensor<ImplType, N-1> Base :: slice(int dimIdx, int idx) const {
        // 参数检查
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw std::runtime_error("[int] operates on dimensions that exceed those of Tensor.");
        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        int dim = this->dim_ - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = this->shape_.get()[idx];
                stride.get()[idx] = this->stride_.get()[idx];
            }
            else {
                shape.get()[idx] = this->shape_.get()[idx + 1];
                stride.get()[idx] = this->stride_.get()[idx + 1];
            }
        }
        return Tensor<ImplType, N - 1>(this->data_, offset, dim, shape, stride, this->current_dim);
    }
    template<class ImplType, int N>
    Tensor<ImplType, N> Base :: slice(int dimIdx, int start, int end, int sliceStride, int ModifyDim) const {
        // 参数检查
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of Tensor.");
        int offset = this->offset_ + start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return Tensor<ImplType, N>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    #undef Base

    #define Base Tensor<ImplType, 1>

    template<class ImplType>
    template <class InputIt>
    void Base :: initialize(InputIt first, InputIt last) {
        size_t size = std::distance(first, last);
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[size], std::default_delete<ImplType[]>());
        std::copy(first, last, this->data_.get());
        this->offset_ = 0;
        this->dim_ = 1;
        this->shape_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->shape_.get()[0] = size;
        this->stride_.get()[0] = 1;
    }
    template<class ImplType>
    Base :: Tensor(const TensorProxy<ImplType, 1> &x){
        this->data_ = x.getDataPtr();
        this->offset_ = x.getOffset();
        this->dim_ = x.getDim();
        this->shape_ = x.getShapePtr();
        this->stride_ = x.getStridePtr();
        this->current_dim = 0;
    }
    template<class ImplType>
    Base :: Tensor(std::vector<ImplType> init){
        initialize(init.begin(), init.end());
    }
    template<class ImplType>
    Base :: Tensor(int len, const ImplType *init){
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[len], std::default_delete<ImplType[]>());
        std::copy_n(init, len, this->data_.get());
        this->offset_ = 0;
        this->dim_ = 1;
        this->shape_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->shape_.get()[0] = len;
        this->stride_.get()[0] = 1;
    }
    template<class ImplType>
    Base :: Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
    }
    template<class ImplType>
    Base :: Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    template<class ImplType>
    Tensor<ImplType, 1>& Base :: operator=(const Tensor<ImplType, 1>& operand) {
        this->data_ = operand.getDataPtr();
        this->offset_ = operand.getOffset();
        this->dim_ = operand.getDim();
        this->shape_ = operand.getShapePtr();
        this->stride_ = operand.getStridePtr();
        this->current_dim = 0;
        return *this;
    }
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0)
            return TensorProxy<ImplType, 1>(*this, this->current_dim, 0, this->shape_.get()[this->current_dim], 1, 1);
        else if(idx.size() == 1){
            const int i = *(idx.begin());
            return TensorProxy<ImplType, 1>(*this, this->current_dim, i, i + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            return TensorProxy<ImplType, 1>(*this, this->current_dim, start, end, stride, 1);
        }
    }
    template<class ImplType>
    ImplType& Base :: operator[](int idx) {return slice(this->current_dim, idx);}
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: operator[](RegularSplit sp) const {return TensorProxy<ImplType, 1>(*this, 0, 0, this->shape_.get()[this->current_dim], 1, 0);}
    template<class ImplType>
    ImplType& Base :: operator[](Index sp) const{}
    template<class ImplType>
    ImplType& Base :: slice(int dimIdx, int idx) const {
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw std::runtime_error("[int] operates on dimensions that exceed those of Tensor.");
        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        return this->data_.get()[offset];
    }
    template<class ImplType>
    Tensor<ImplType, 1> Base :: slice(int dimIdx, int start, int end, int sliceStride, int ModifyDim) const {
        // 参数检查
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of Tensor.");
        int offset = start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return Tensor<ImplType, 1>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    #undef Base
    #define Base ProxyBase<ImplType>

    template <class ImplType>
    int Base :: getStride(int dimIdx) const {return this->stride_.get()[dimIdx];}
    template <class ImplType>
    int Base :: getShape(int dimIdx) const {return this->shape_.get()[dimIdx]; }
    template <class ImplType>
    std::shared_ptr<int> Base :: getStridePtr() const {return this->stride_;}
    template <class ImplType>
    std::shared_ptr<ImplType> Base :: getDataPtr() const{return this->data_;}
    template <class ImplType>
    std::shared_ptr<int> Base :: getShapePtr() const {return this->shape_;}
    template <class ImplType>
    int Base :: getOffset() const {return this->offset_;}
    template <class ImplType>
    int Base :: getDim() const {return this->dim_;}

    template <class ImplType>
    int Base :: getSize() const {
        int size = 1;
        for(int dimIdx = 0; dimIdx < getDim(); dimIdx++) {
            size *= this->shape_.get()[dimIdx];
        }
        return size;
    }
    template <class ImplType>
    int Base :: getCurrentDim() const {
        return this->current_dim;
    }
    template <class ImplType>
    void Base :: NextDim(){
        this->current_dim ++;
    }
    #undef Base

    #define Base TensorProxy<ImplType, N>

    template<class ImplType, int N>
    Base :: TensorProxy(dacpp::Tensor<ImplType, N + 1> tensor, int dimIdx, int idx){
        if(dimIdx >= tensor.getDim() || idx >= tensor.getShape(dimIdx))
            throw std::runtime_error("[int] operates on dimensions that exceed those of Tensor.");
        int offset = tensor.getOffset() + idx * tensor.getStride(dimIdx);
        int dim = tensor.getDim() - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = tensor.getShape(idx);
                stride.get()[idx] = tensor.getStride(idx);
            }
            else {
                shape.get()[idx] = tensor.getShape(idx + 1);
                stride.get()[idx] = tensor.getStride(idx + 1);
            }
        }
        *this = TensorProxy(tensor.getDataPtr(), offset, dim, shape, stride, tensor.getCurrentDim());
    }
    template<class ImplType, int N>
    Base :: TensorProxy(dacpp::Tensor<ImplType, N> tensor, int dimIdx, int start, int end, int sliceStride, int ModifyDim){

        if(dimIdx >= tensor.getDim() || start >= tensor.getShape(dimIdx) || end > tensor.getShape(dimIdx)
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of TensorProxy.");
        int offset = start * tensor.getStride(dimIdx);
        int dim = tensor.getDim();
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = tensor.getShape(i);
            stride.get()[i] = tensor.getStride(i);
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = tensor.getStride(dimIdx) * sliceStride;
        *this = TensorProxy(tensor.getDataPtr(), offset, dim, shape, stride, tensor.getCurrentDim() + ModifyDim);
    }
    template<class ImplType, int N>
    Base :: TensorProxy(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Base :: operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0)
            return Pslice(this->current_dim, 0, this->shape_.get()[this->current_dim], 1, 1);
        else if(idx.size() == 1){
            start = *(idx.begin());
            return Pslice(this->current_dim, start, start + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            return Pslice(this->current_dim, start, end, stride, 1);
        }
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N-1> Base :: operator[](int idx) const {
        return slice(this->current_dim, idx);
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Base :: operator[](RegularSplit sp) const {
        return *this;
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N - 1> Base :: operator[](Index sp) const {
        return TensorProxy<ImplType, N-1>();
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N-1> Base :: Pslice(int dimIdx, int idx) const {
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw std::runtime_error("[int] operates on dimensions that exceed those of TensorProxy.");
        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        int dim = this->dim_ - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = this->shape_.get()[idx];
                stride.get()[idx] = this->stride_.get()[idx];
            }
            else {
                shape.get()[idx] = this->shape_.get()[idx + 1];
                stride.get()[idx] = this->stride_.get()[idx + 1];
            }
        }
        return TensorProxy<ImplType, N - 1>(this->data_, offset, dim, shape, stride, this->current_dim);
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Base :: Pslice(int dimIdx, int start, int end, int sliceStride, int ModifyDim) const {
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of TensorProxy.");
        int offset = this->offset_ + start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return TensorProxy<ImplType, N>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    #undef Base

    #define Base TensorProxy<ImplType, 1>

    template<class ImplType>
    Base :: TensorProxy(dacpp::Tensor<ImplType, 2> tensor, int dimIdx, int idx){
        if(dimIdx >= tensor.getDim() || idx >= tensor.getShape(dimIdx))
            throw std::runtime_error("[int] operates on dimensions that exceed those of Tensor.");
        int offset = tensor.getOffset() + idx * tensor.getStride(dimIdx);
        int dim = tensor.getDim() - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = tensor.getShape(idx);
                stride.get()[idx] = tensor.getStride(idx);
            }
            else {
                shape.get()[idx] = tensor.getShape(idx + 1);
                stride.get()[idx] = tensor.getStride(idx + 1);
            }
        }
        *this = TensorProxy(tensor.getDataPtr(), offset, dim, shape, stride, tensor.getCurrentDim());
    }
    template<class ImplType>
    Base :: TensorProxy(dacpp::Tensor<ImplType, 1> tensor, int dimIdx, int start, int end, int sliceStride, int ModifyDim){
        if(dimIdx >= tensor.getDim || start >= tensor.getShape(dimIdx) || end > tensor.getShape(dimIdx)
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of TensorProxy.");
        int offset = start * tensor.getStride(dimIdx);
        int dim = tensor.getDim();
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = tensor.getShape(i);
            stride.get()[i] = tensor.getStride(i);
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = tensor.getStride(dimIdx) * sliceStride;
        *this = TensorProxy(tensor.getDataPtr(), offset, dim, shape, stride, tensor.getCurrentDim() + ModifyDim);
    }
    template<class ImplType>
    Base :: TensorProxy(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0)
            return Pslice(this->current_dim, 0, this->shape_.get()[this->current_dim], 1, 1);
        else if(idx.size() == 1){
            const int i = *(idx.begin());
            return Pslice(this->current_dim, i, i + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            return slice(this->current_dim, start, end, stride, 1);
        }
    }
    template<class ImplType>
    ImplType& Base :: operator[](int idx) {
        return Pslice(this->current_dim, idx);
    }
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: operator[](RegularSplit sp) const {
        return *this;
    }
    template<class ImplType>
    ImplType& Base :: operator[](Index sp) const{}
    template<class ImplType>
    ImplType& Base ::Pslice(int dimIdx, int idx) const {
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw("Slice operates on dimensions that exceed those of TensorProxy.");
        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        return this->data_.get()[offset];
    }
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: Pslice(int dimIdx, int start, int end, int sliceStride, int ModifyDim) const {
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw("Slice operates on dimensions that exceed those of TensorProxy.");
        int offset = this->offset_ + start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return TensorProxy<ImplType, 1>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    #undef Base
}

#endif