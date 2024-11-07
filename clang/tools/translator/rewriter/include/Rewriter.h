#ifndef TRANSLATOR_REWRITER_REWRITER_H
#define TRANSLATOR_REWRITER_REWRITER_H

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "DacppStructure.h"

namespace dacppTranslator {

class Rewriter {

private:
    clang::Rewriter* rewriter;
    DacppFile* dacppFile;

public:

    /**
     * 生成数据重排的代码
     */
    std::string generateDataRearrangement(ShellParam* shellParam);

    void setRewriter(clang::Rewriter* rewriter);

    void setDacppFile(DacppFile* dacppFile);

    void rewriteDac();

    void rewriteMain();
};

}

# endif