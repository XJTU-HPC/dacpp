#ifndef ASTLIB_H
#define ASTLIB_H

#include <vector>

#include "clang/AST/AST.h"
#include "rewriteLib.h"

using namespace clang;

namespace dacpp {

// 将 clang 节点对应的代码转换成 std::string
std::string stmt2String(Stmt *stmt);

// 得到 Slice
void getSlice(ShellParam& shellParam, Expr* curExpr, std::vector<InitListExpr*>& op);

// 判断数据的输入输出属性
bool inputOrOutput(std::string dataType);

// 从抽象语法树中获得函数参数的信息
void getParamInfo(const BinaryOperator* dacExpr, std::vector<std::vector<int>>& shapes);

// 根据数据类型获得其基本类型
std::string getBasicType(std::string type);

}

#endif