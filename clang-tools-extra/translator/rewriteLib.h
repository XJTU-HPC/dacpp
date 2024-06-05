#ifndef LIB_H
#define LIB_H

#include <string>
#include <vector>

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace dacpp {

class HeaderFile {
private:
    std::string file_;
public:
    HeaderFile() {}
    HeaderFile(std::string file) : file_(file) {}
    void setFile(std::string file) { file_ = file; }
    std::string getFile() const { return file_; }
};

class Param {
private:
    std::string type_;
    std::string name_;
public:
    Param() {}
    Param(std::string type, std::string name) : type_(type), name_(name) {}
    void setType(std::string type) { type_ = type; }
    void setName(std::string name) { name_ = name; }
    std::string getType() const { return type_; }
    std::string getName() const { return name_; }
};

class Calc {
private:
    std::string name_;
    std::vector<Param> params_;
    std::string body_;
public:
    Calc() {}
    Calc(std::string name, std::vector<Param> params, std::string body) : name_(name), params_(params), body_(body) {}
    void setName(std::string name) { name_ = name; }
    void setParams(Param param) { params_.push_back(param); }
    void setBody(std::string body) { body_ = body; }
    std::string getName() const { return name_; }
    int getNumParams() const { return params_.size(); }
    Param getParam(int idx) const { return params_[idx]; }
    std::string getBody() const { return body_; }
};

class ShellParam {
private:
    std::string name_;
    std::vector<Param> ops_;
    bool rw_;
public:
    ShellParam() {}
    ShellParam(std::string name, bool rw, std::vector<Param>& ops) : name_(name), rw_(rw), ops_(ops) {}
    void setName(std::string name) { name_ = name; }
    void setParams(Param op) { ops_.push_back(op); }
    void setRw(bool rw) { rw_ = rw; }
    std::string getName() const { return name_; }
    int getNumOps() const { return ops_.size(); }
    Param getOp(int idx) const{ return ops_[idx]; }
    bool getRw() const { return rw_; }
};

class Shell {
private:
    std::string name_;
    std::vector<Param> params_;
    std::string calcName_;
    std::vector<ShellParam> shellParams_;
public:
    Shell() {}
    void setName(std::string name) { name_ = name; }
    void setParams(Param param) { params_.push_back(param); }
    void setCalcName(std::string name) { calcName_ = name; }
    void setShellParams(ShellParam param) { shellParams_.push_back(param); }
    std::string getName() const { return name_; }
    int getNumParams() const { return params_.size(); }
    Param getParam(int idx) const { return params_[idx]; }
    std::string getCalcName() const { return calcName_; }
    int getNumShellParams() const { return shellParams_.size(); }
    ShellParam getShellParam(int idx) const { return shellParams_[idx]; }
};

class Source2Source {
private:
    clang::Rewriter* rewriter_;
    const clang::FunctionDecl* mainFunc_;
    std::vector<HeaderFile> headerFiles_;
    std::vector<Calc> calcs_;
    std::vector<Shell> shells_;
    void recursiveRewriteMain(clang::Stmt* curStmt);
public:
    Source2Source() {}
    void setRewriter(clang::Rewriter* rewriter) { rewriter_ = rewriter; }
    void setMainFunc(const clang::FunctionDecl* mainFunc) { mainFunc_ = mainFunc; }
    void setHeaderFile(std::string file);
    void setCalc(clang::FunctionDecl* calcFunc);
    void setShell(const clang::BinaryOperator* dacExpr);
    int getNumHeaderFiles() const { return headerFiles_.size(); }
    HeaderFile getHeaderFile(int idx) const { return headerFiles_[idx]; }
    int getNumCalcs() const { return calcs_.size(); }
    Calc getCalc(int idx) const { return calcs_[idx]; }
    int getNumShells() const { return shells_.size(); }
    Shell getShell(int idx) const { return shells_[idx]; }
    void rewriteDac();
    void rewriteMain();
};

}

#endif