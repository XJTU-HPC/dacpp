#include <set>
#include <iostream>
#include <sstream>

#include "clang/Rewrite/Core/Rewriter.h"
#include "Rewriter.h"
#include "Split.h"
#include "Param.h"
#include "dacInfo.h"
#include "sub_template.h"
#include "test.h"

// void DFS(int node, std::vector<bool>& visited, Dac_Ops ops, Dac_Ops component, int componentID, std::vector<std::string>& sets) {
//             visited[node] = true;
//             sets[node] = std::to_string(componentID);  // 给当前节点标记一个连通分量的ID

//             // 将当前节点加入到当前连通分量
//             component.push_back(ops[node]);

//             // 遍历所有与当前节点相连的节点
//             for (int i = 0; i < ops.size; ++i) {
//                 int k;
//                 if (!visited[i] && GetBindInfo(node, i,k)) {
//                     DFS(i, visited, ops, component, componentID, sets);
//                 }
//             }   
// }



void dacppTranslator::Rewriter::rewriteDac_Soft() {



    std::string code = "";
    
    // 添加头文件
    for(int i = 0; i < dacppFile->getNumHeaderFile(); i++) {
        code += "#include " + dacppFile->getHeaderFile(i)->getName() + "\n";
    }
    code += "\n";
    
    // 添加命名空间
    for(int i = 0; i < dacppFile->getNumNameSpace(); i++) {
        code += "using namespace " + dacppFile->getNameSpace(i)->getName() + ";\n";
    }
    code += "\n";
    
    // 添加数据关联表达式对应的划分结构和计算结构
    for(int i = 0; i < dacppFile->getNumExpression(); i++) {
        Expression* expr = dacppFile->getExpression(i);
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        
        // 计算结构
        code += "void " + calc->getName() + "(";
        for(int count = 0; count < calc->getNumParams(); count++) {
            code += calc->getParam(count)->getBasicType() + "* " + calc->getParam(count)->getName();
            if(count != calc->getNumParams() - 1) {
                code += ", ";
            }
        }
        code += ") \n";
        for(int count = 0; count < calc->getNumBody(); count++) {
            
            code += calc->getBody(count) + "\n";
        }
        



        






}

