#include <iostream>
#include <string>
#include <typeinfo>

#include "../parser/Param.h"
#include "../parser/DacppStructure.h"
#include "test.h"

void dacppTranslator::printParamInfo(dacppTranslator::Param* param) {
    std::cout << "读写属性" << param->getRw() << std::endl;
    std::cout << "包装类型" << param->getType() << std::endl;
    std::cout << "基本类型" << param->getBasicType() << std::endl;
    std::cout << "参数名" << param->getName() << std::endl;
    std::cout << "参数形状：";
    for(int i = 0; i < param->getDim(); i++) {
        std::cout << param->getShape(i) << " ";
    }
    std::cout << std::endl;
}

void dacppTranslator::printShellParamInfo(dacppTranslator::ShellParam* shellParam) {
    std::cout << "读写属性" << shellParam->getRw() << std::endl;
    std::cout << "包装类型" << shellParam->getType() << std::endl;
    std::cout << "基本类型" << shellParam->getBasicType() << std::endl;
    std::cout << "参数名" << shellParam->getName() << std::endl;
    std::cout << "参数形状：";
    for(int i = 0; i < shellParam->getDim(); i++) {
        std::cout << shellParam->getShape(i) << " ";
    }
    std::cout << "划分列表：\n";
    for(int i = 0; i < shellParam->getNumSplit(); i++) {
        std::cout << shellParam->getSplit(i)->getId() << std::endl;
        std::cout << shellParam->getSplit(i)->getDimIdx() << std::endl;
    }
    std::cout << std::endl;
}

void dacppTranslator::printShellInfo(dacppTranslator::Shell* shell) {
    std::cout << shell->getName() << std::endl;
    for(int i = 0; i < shell->getNumParams(); i++) {
        printParamInfo(shell->getParam(i));
    }
    for(int i = 0; i < shell->getNumShellParams(); i++) {
        printShellParamInfo(shell->getShellParam(i));
    }
}

void dacppTranslator::printCalcInfo(dacppTranslator::Calc* calc) {
    std::cout << calc->getName() << std::endl;
    for(int i = 0; i < calc->getNumParams(); i++) {
        printParamInfo(calc->getParam(i));
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

    std::cout << "数据管理计算表达式：" << std::endl;
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