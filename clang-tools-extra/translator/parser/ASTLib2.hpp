#ifndef ASTLIB2_HPP
#define ASTLIB2_HPP

#include <vector>

#include "clang/AST/AST.h"

using namespace clang;

namespace dacpp {

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

}

#endif