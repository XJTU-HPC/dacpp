#ifndef LIB_H
#define LIB_H

#include <string>
#include <vector>

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace dacpp {

// 头文件类
class HeaderFile {
private:
    std::string name_;

public:
    HeaderFile() {}
    HeaderFile(std::string name) : name_(name) {}

    void setName(std::string name) { name_ = name; }

    std::string getName() const { return name_; }
    std::string getStr() const { return "#include " + name_; }
};

// 命名空间类
class NameSpace {
private:
    std::string name_;

public:
    NameSpace() {}
    NameSpace(std::string name) : name_(name) {}

    void setName(std::string name) { name_ = name; }

    std::string getName() const { return name_; }
    std::string getStr() const { return "using namespace " + name_ + ";"; }
};

// 参数类
class Param {
private:
    bool rw_; // 读写属性
    std::string type_; // 类型
    std::string name_; // 变量名
    std::string basicType_; // 基本类型
    int dim_; // 维度
    std::vector<int> shapes_; // 形状

public:
    Param() {}

    void setType(std::string type) { type_ = type; }
    void setName(std::string name) { name_ = name; }
    void setBasicType(std::string basicType) { basicType_ = basicType; }
    void setDim(int dim) { dim_ = dim; }
    void setShapes(int shape) { shapes_.push_back(shape); }

    std::string getType() const { return type_; }
    std::string getName() const { return name_; }
    std::string getBasicType() const { return basicType_; }
    int getDim() { return dim_; }
    int getNumShapes() { return shapes_.size(); }
    int getShape(int idx) { return shapes_[idx]; }
};

// 计算函数类
class Calc {
private:
    std::string name_; // 函数名
    std::vector<Param> params_; // 参数
    std::string body_; // 函数体

public:
    Calc() {}

    void setName(std::string name) { name_ = name; }
    void setParams(Param param) { params_.push_back(param); }
    void setBody(std::string body) { body_ = body; }

    std::string getName() const { return name_; }
    int getNumParams() const { return params_.size(); }
    Param getParam(int idx) const { return params_[idx]; }
    std::string getBody() const { return body_; }
};

// DAC 算子类
class DacOp {
private:
    std::string type_; // 类型
    std::string name_; // 变量名
    int splitSize_; // 划分份数

public:
    DacOp() {}
    
    void setType(std::string type) { type_ = type; }
    void setName(std::string name) { name_ = name; }
    void setSplitSize(int splitSize) { splitSize_ = splitSize; }

    std::string getType() const { return type_; }
    std::string getName() const { return name_; }
    int getSplitSize() const { return splitSize_; }
};

// 关联结构参数类
class ShellParam {
private:
    bool rw_; // 读写属性
    std::string type_; // 类型
    std::string name_; // 变量名
    std::string basicType_; // 基本类型
    int dim_; // 维度
    std::vector<int> shapes_; // 形状
    std::vector<DacOp> ops_; // 算子

public:
    ShellParam() {}

    void setRw(bool rw) { rw_ = rw; }
    void setType(std::string type) { type_ = type; }
    void setName(std::string name) { name_ = name; }
    void setBasicType(std::string basicType) { basicType_ = basicType; }
    void setDim(int dim) { dim_ = dim; }
    void setShapes(int shape) { shapes_.push_back(shape); }
    void setParams(DacOp op) { ops_.push_back(op); }

    bool getRw() const { return rw_; }
    std::string getType() const { return type_; }
    std::string getName() const { return name_; }
    std::string getBasicType() const { return basicType_; }
    int getDim() { return dim_; }
    int getNumShapes() { return shapes_.size(); }
    int getShape(int idx) { return shapes_[idx]; }
    int getNumOps() const { return ops_.size(); }
    DacOp getOp(int idx) const{ return ops_[idx]; }
};

// 关联结构函数类
class Shell {
private:
    std::string name_; // 函数名
    std::vector<Param> params_; // 参数
    std::vector<DacOp> ops_; // 算子
    std::string calcName_; // 计算函数名
    std::vector<ShellParam> shellParams_; // 关联结构参数

public:
    Shell() {}

    void setName(std::string name) { name_ = name; }
    void setParams(Param param) { params_.push_back(param); }
    void setOps(DacOp op) { ops_.push_back(op); }
    void setCalcName(std::string name) { calcName_ = name; }
    void setShellParams(ShellParam param) { shellParams_.push_back(param); }
    
    std::string getName() const { return name_; }
    int getNumParams() const { return params_.size(); }
    Param getParam(int idx) const { return params_[idx]; }
    int getNumOps() const { return ops_.size(); }
    DacOp getOp(int idx) const { return ops_[idx]; }
    std::string getCalcName() const { return calcName_; }
    int getNumShellParams() const { return shellParams_.size(); }
    ShellParam getShellParam(int idx) const { return shellParams_[idx]; }

    void setOpSplitSize(int idx, int splitsize) { ops_[idx].setSplitSize(splitsize); }
};

// 源到源翻译类
class Source2Source {
private:
    clang::Rewriter* rewriter_;
    const clang::FunctionDecl* mainFunc_;

    std::vector<HeaderFile> headerFiles_; // 头文件
    std::vector<NameSpace> nameSpaces_; // 命名空间
    std::vector<Calc> calcs_; // 计算函数
    std::vector<Shell> shells_; // 关联结构函数

    void recursiveRewriteMain(clang::Stmt* curStmt);

public:
    Source2Source() {}

    void setRewriter(clang::Rewriter* rewriter) { rewriter_ = rewriter; }
    void setMainFunc(const clang::FunctionDecl* mainFunc) { mainFunc_ = mainFunc; }
    
    void setHeaderFiles(std::string name);
    void setNameSpaces(std::string name);
    void setCalcs(const clang::BinaryOperator* dacExpr);
    void setShells(const clang::BinaryOperator* dacExpr);

    int getNumHeaderFiles() const { return headerFiles_.size(); }
    HeaderFile getHeaderFile(int idx) const { return headerFiles_[idx]; }
    int getNumNameSpaces() const { return nameSpaces_.size(); }
    NameSpace getNameSpace(int idx) const { return nameSpaces_[idx]; }
    int getNumCalcs() const { return calcs_.size(); }
    Calc getCalc(int idx) const { return calcs_[idx]; }
    int getNumShells() const { return shells_.size(); }
    Shell getShell(int idx) const { return shells_[idx]; }

    // 将头文件与命名空间加入到源文件中
    // 将数据与计算关联起来生成新的函数加入到源文件中
    void rewriteDac();
    // 重写 DAC main 函数源代码
    void rewriteMain();
};

}

#endif