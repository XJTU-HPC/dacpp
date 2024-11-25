#include <iostream>
#include <string>
#include <typeinfo>

#include "Param.h"
#include "DacppStructure.h"
#include "test.h"

void dacppTranslator::printParamInfo(dacppTranslator::Param* param) {
    std::cout << "\t\t读写属性：" << param->getRw() << std::endl;
    std::cout << "\t\t包装类型：" << param->getType() << std::endl;
    std::cout << "\t\t基本类型：" << param->getBasicType() << std::endl;
    std::cout << "\t\t参数名：" << param->getName() << std::endl;
    std::cout << "\t\t参数形状：";
    for (int i = 0; i < param->getDim(); i++) {
        std::cout << param->getShape(i) << " ";
    }
    std::cout << "\n\n";
}

void dacppTranslator::printShellParamInfo(dacppTranslator::ShellParam* shellParam) {
    std::cout << "\t\t读写属性：" << shellParam->getRw() << std::endl;
    std::cout << "\t\t包装类型：" << shellParam->getType() << std::endl;
    std::cout << "\t\t基本类型：" << shellParam->getBasicType() << std::endl;
    std::cout << "\t\t参数名：" << shellParam->getName() << std::endl;
    std::cout << "\t\t参数形状：";
    for(int i = 0; i < shellParam->getDim(); i++) {
        std::cout << shellParam->getShape(i) << " ";
    }
    std::cout << "\n\t\t划分列表：";
    for(int i = 0; i < shellParam->getNumSplit(); i++) {
        std::cout << shellParam->getSplit(i)->getId() << " ";
        std::cout << shellParam->getSplit(i)->getDimIdx() << "  ";
    }
    std::cout << "\n\n";
}

void dacppTranslator::printShellInfo(dacppTranslator::Shell* shell) {
    std::cout << "\tShell名：" << shell->getName() << std::endl;
    std::cout << "\tShell参数信息：\n";
    for(int i = 0; i < shell->getNumParams(); i++) {
        printParamInfo(shell->getParam(i));
    }
    std::cout << "\t关联结构参数信息：\n";
    for(int i = 0; i < shell->getNumShellParams(); i++) {
        printShellParamInfo(shell->getShellParam(i));
    }
}

void dacppTranslator::printCalcInfo(dacppTranslator::Calc* calc) {
    std::cout << "\tCalc名：" << calc->getName() << std::endl;
    std::cout << "\tCalc参数信息：\n";
    for(int i = 0; i < calc->getNumParams(); i++) {
        printParamInfo(calc->getParam(i));
    }
    std::cout << "\tCalc函数体：\n";
    for(int i = 0; i < calc->getNumBody(); i++) {
        if(calc->getBody(i).compare("Expression") != 0) {
            std::cout << "\t\t" << calc->getBody(i) << "\n";
            continue;
        }
        std::cout << "子数据关联计算表达式：" << std::endl;
        for(int j = 0; j < calc->getNumExprs(); j++) {
            printShellInfo(calc->getExpr(j)->getShell());
            printCalcInfo(calc->getExpr(j)->getCalc());
        }
    }
}

void dacppTranslator::printDacppFileInfo(dacppTranslator::DacppFile* dacppFile) {
    std::cout << "头文件：" << std::endl;
    for(int i = 0; i < dacppFile->getNumHeaderFile(); i++) {
        std::cout << dacppFile->getHeaderFile(i)->getName() << std::endl;
    }
    std::cout << std::endl;

    std::cout << "命名空间：" << std::endl;
    for(int i = 0; i < dacppFile->getNumNameSpace(); i++) {
        std::cout << dacppFile->getNameSpace(i)->getName() << std::endl;
    }
    std::cout << std::endl;

    std::cout << "数据关联计算表达式：" << std::endl;
    for(int i = 0; i < dacppFile->getNumExpression(); i++) {
        std::cout << "Shell：" << std::endl;        
        printShellInfo(dacppFile->getExpression(i)->getShell());
        std::cout << std::endl;
        std::cout << "Calc：" << std::endl;
        printCalcInfo(dacppFile->getExpression(i)->getCalc());
        std::cout << std::endl;
    }
    std::cout << std::endl;
}