#ifndef TRANSLATOR_REWRITER_REWRITER_H
#define TRANSLATOR_REWRITER_REWRITER_H

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "DacppStructure.h"

#include <string>

namespace dacppTranslator {

class Rewriter {
private:
    clang::Rewriter* rewriter;
    DacppFile* dacppFile;

    std::string getDacExpr(Expression* expr);

public:   

    void setRewriter(clang::Rewriter* rewriter);

    void setDacppFile(DacppFile* dacppFile);

    void rewriteDac();

    void rewriteMain();
};

// namespace dacppTranslator
}

# endif