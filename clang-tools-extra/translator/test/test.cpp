#include <iostream>

#include "../rewriteLib.h"

// 测试 Shell
void shellTest(dacpp::Shell* shell) {
    std::cout << "*****This is shellTest.*****" << "\n";
    for(int i = 0; i < shell->getNumOps(); i++) {
        std::cout << shell->getOp(i).getType() << "\n";
        std::cout << shell->getOp(i).getName() << "\n";
        std::cout << shell->getOp(i).getSplitSize() << "\n";
    }
    for(int i = 0; i < shell->getNumParams(); i++) {
        std::cout << shell->getParam(i).getType() << "\n";
        std::cout << shell->getParam(i).getName() << "\n";
        std::cout << shell->getParam(i).getBasicType() << "\n";
        std::cout << shell->getParam(i).getDim() << "\n";
        for(int j = 0; j < shell->getParam(i).getNumShapes(); j++) {
            std::cout << shell->getParam(i).getShape(j) << " ";
        }
        std::cout << "\n";
    }
    for(int i = 0; i < shell->getNumShellParams(); i++) {
        std::cout << shell->getShellParam(i).getRw() << "\n";
        std::cout << shell->getShellParam(i).getType() << "\n";
        std::cout << shell->getShellParam(i).getName() << "\n";
        std::cout << shell->getShellParam(i).getBasicType() << "\n";
        std::cout << shell->getShellParam(i).getDim() << "\n";
        for(int j = 0; j < shell->getShellParam(i).getNumShapes(); j++) {
            std::cout << shell->getShellParam(i).getShape(j) << " ";
        }
        std::cout << "\n";
        for(int j = 0; j < shell->getShellParam(i).getNumOps(); j++) {
            std::cout << shell->getShellParam(i).getOp(j).getType() << "\n";
            std::cout << shell->getShellParam(i).getOp(j).getName() << "\n";
            std::cout << shell->getShellParam(i).getOp(j).getSplitSize() << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "*****This is shellTest.*****" << "\n";
}

// 测试 calc
void calcTest(dacpp::Calc* calc) {
    std::cout << "*****This is calcTest.*****" << "\n";
    for(int i = 0; i < calc->getNumParams(); i++) {
        std::cout << "Param:\n";
        std::cout << "类型" << calc->getParam(i).getType() << "\n";
        std::cout << "变量名" << calc->getParam(i).getName() << "\n";
        std::cout << "基本数据类型" << calc->getParam(i).getBasicType() << "\n";
        std::cout << "维度" << calc->getParam(i).getDim() << "\n";
        for(int j = 0; j < calc->getParam(i).getNumShapes(); j++) {
            std::cout << calc->getParam(i).getShape(j) << " ";
        }
        std::cout << "\n";
    }
    while(1) {
        std::string temp = calc->getBody();
        if(temp == "None") { break; }
        std::cout << temp << "\n";
    }
    std::cout << "*****This is calcTest.*****" << "\n";
}