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

}