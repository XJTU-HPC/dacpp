#include <vector>

#include "clang/AST/AST.h"

using namespace clang;

namespace dacpp {

// 将 clang 节点对应的代码转换成 std::string
std::string stmt2String(Stmt *stmt) {
    clang::LangOptions lo;
    std::string out_str;
    llvm::raw_string_ostream outstream(out_str);
    stmt->printPretty(outstream, NULL, PrintingPolicy(lo));
    return out_str;
}

// 从结点中找到特定结点，深度优先搜索
template <typename NodeType>
NodeType* getNode(Stmt* curStmt) {
  if(!curStmt) return nullptr;
  for(Stmt::child_iterator it = curStmt->child_begin(); it != curStmt->child_end(); it++) {
    if(isa<NodeType>(*it)) return dyn_cast<NodeType>(*it);
    if(getNode<NodeType>(*it)) {
      return getNode<NodeType>(*it);
    }
  }
  return nullptr;
}

// 得到 Slice
void getSlice(ShellParam& shellParam, Expr* curExpr, std::vector<InitListExpr*>& op) {
    CXXOperatorCallExpr* opExpr = dacpp::getNode<CXXOperatorCallExpr>(curExpr);
    if(!opExpr) {
      int count = 0;
      for(Stmt::child_iterator it = curExpr->child_begin(); it != curExpr->child_end(); it++, count++) {
        if(count != 1) { continue; }
        if(isa<DeclRefExpr>(*it)) { shellParam.setName(dyn_cast<DeclRefExpr>(*it)->getNameInfo().getAsString()); }
        else { shellParam.setName(getNode<DeclRefExpr>(*it)->getNameInfo().getAsString()); }
      }
      return;
    }
    getSlice(shellParam, opExpr, op);
    for(Stmt::child_iterator it = opExpr->child_begin(); it != opExpr->child_end(); it++) {
        if(isa<InitListExpr>(*it)) {
            op.push_back(dyn_cast<InitListExpr>(*it));
        }
    }
}

// 判断数据的输入输出属性
bool inputOrOutput(std::string dataType) {
  std::string pre = "const";
  std::string::size_type idx;
  idx = dataType.find(pre);
  if(idx == std::string::npos) {
    return true;
  }
  else {
    return false;
  }
}

// 从抽象语法树中获得函数参数的信息
void getParamInfo(Expr* curExpr, Param& param) {
  DeclRefExpr* declRefExpr;
  if(isa<DeclRefExpr>(curExpr)) {
    declRefExpr = dyn_cast<DeclRefExpr>(curExpr);
  }
  else {
    declRefExpr = getNode<DeclRefExpr>(curExpr);
  }
  int count = 0;
  for(Stmt::child_iterator it = dyn_cast<VarDecl>(declRefExpr->getDecl())->getInit()->child_begin(); it != dyn_cast<VarDecl>(declRefExpr->getDecl())->getInit()->child_end(); it++) {
    if(count != 1) {
      count++;
      continue;
    }
    VarDecl* shapeDecl = dyn_cast<VarDecl>(getNode<DeclRefExpr>(*it)->getDecl());
    InitListExpr* initListExpr = getNode<InitListExpr>(shapeDecl->getInit());
    for(Stmt::child_iterator it = initListExpr->child_begin(); it != initListExpr->child_end(); it++) {
      IntegerLiteral* interger = dyn_cast<IntegerLiteral>(*it);
      param.setShapes(std::stoi(interger->getValue().toString(10, true)));
    }
    param.setDim(param.getNumShapes());
    count++;
  }
}

// 根据数据类型获得其基本类型
std::string getBasicType(std::string type) {
  if(type.find("Tensor<int>") != -1) {
      return "int";
  }
  else if(type.find("Tensor<unsigned int>") != -1) {
      return "unsigned int";
  }
  else if(type.find("Tensor<signed int>") != -1) {
      return "signed int";
  }
  else if(type.find("Tensor<short int>") != -1) {
      return "short int";
  }
  else if(type.find("Tensor<unsigned short int>") != -1) {
      return "unsigned short int";
  }
  else if(type.find("Tensor<signed short int>") != -1) {
      return "signed short int";
  }
  if(type.find("Tensor<long int>") != -1) {
      return "long int";
  }
  else if(type.find("Tensor<unsigned long int>") != -1) {
      return "unsigned long int";
  }
  else if(type.find("Tensor<signed long int>") != -1) {
      return "signed long int";
  }
  else if(type.find("Tensor<float>") != -1) {
      return "float";
  }
  else if(type.find("Tensor<double>") != -1) {
      return "double";
  }
  else if(type.find("Tensor<long long>") != -1) {
      return "long long";
  }
  else {
      return "long double";
  }
}

}