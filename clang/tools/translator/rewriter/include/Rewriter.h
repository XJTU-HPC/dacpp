#ifndef TRANSLATOR_REWRITER_REWRITER_H
#define TRANSLATOR_REWRITER_REWRITER_H

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "DacppStructure.h"
#include "Param.h"
#include "dacInfo.h"

#include <string>

namespace dacppTranslator {

class Rewriter {
private:
    clang::Rewriter* rewriter;
    DacppFile* dacppFile;

    std::string generateCalc(std::string code, Expression* expr) {
        Calc* calc = expr->getCalc();
        // 计算结构
        code += "void " + calc->getName() + "(";
        for(int paramCount = 0; paramCount < calc->getNumParams(); paramCount++) {
            Param* param = calc->getParam(paramCount);
            code += param->getBasicType() + "* " + param->getName();
            if(paramCount != calc->getNumParams() - 1) {
                code += ", ";
            }
        }
        code += ") {\n";
        int exprCount = 0;
        for(int bodyCount = 0; bodyCount < calc->getNumBody(); bodyCount++) {
            if (calc->getBody(bodyCount).compare("Expression") != 0) {
                code += "   " + calc->getBody(bodyCount) + "\n";
                continue;
            }
            code += "}\n\n";
            generateCalc(code, calc->getExpr(exprCount++));
            if (bodyCount + 1 == calc->getNumBody()) {
                continue;
            }
            code += "void " + calc->getName() + "(";
            for(int paramCount = 0; paramCount < calc->getNumParams(); paramCount++) {
                Param* param = calc->getParam(paramCount);
                code += param->getBasicType() + "* " + param->getName();
                if(paramCount != calc->getNumParams() - 1) {
                    code += ", ";
                }
            }
            code += ") {\n";
        }
        code += "}\n\n";
        return code;
    }

public:
    void setRewriter(clang::Rewriter* rewriter);

    void setDacppFile(DacppFile* dacppFile);

    std::string generateCalc(Calc* calc);

    void addSplit(std::vector<std::vector<int>>& shapes, std::vector<std::vector<Split*>>& splits,
                  Expression* expr);
    
    std::string generateChildExpr(Expression* ancestor, Expression* expr, Dac_Ops& index, std::vector<Dac_Ops>& tensorOps,
                                  int* mem);
    
    std::string generateChildExpr2(Expression* ancestor, Expression* expr, Dac_Ops& index, std::vector<Dac_Ops>& tensorOps,
                                  int* mem);
    
    std::string generateSyclFunc(Expression* expr);

    void rewriteDac();

    void rewriteDac_buffer();

    void rewriteDac_Soft_Usm();

    void rewriteDac_Soft_Buffer();

    void rewriteDac_usm();
    
    void rewriteDac_Soft();
    

    void rewriteMain();
};

// namespace dacppTranslator
}

# endif