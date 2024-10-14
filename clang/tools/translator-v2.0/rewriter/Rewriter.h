#ifndef TRANSLATOR_REWRITER_REWRITER_H
#define TRANSLATOR_REWRITER_REWRITER_H

#include "../parser/DacppStructure.h"

namespace dacppTranslator {

class Rewriter {

private:
    DacppFile dacppFile;

public:
    void setDacppFile(DacppFile dacppFile);

};

}

# endif