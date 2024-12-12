#ifndef TRANSLATOR_PARSER_PARAM_H
#define TRANSLATOR_PARSER_PARAM_H

#include <string>
#include <vector>

#include "Split.h"

namespace dacppTranslator {

/*
    参数
*/
class Param {

private:
    bool rw; // 读写属性
    std::string type; // 参数包装类型
    std::string basicType; // 参数基本类型
    std::string name; // 参数名
    std::vector<int> shape; // 参数形状

public:
    Param();

    void setRw(bool rw);
    bool getRw();

    void setType(std::string type);
    std::string getType();
    std::string getBasicType();

    void setName(std::string name);
    std::string getName();

    void __attribute__(( unused, deprecated )) setShape(int size) ;
    void __attribute__(( unused, deprecated )) setShape(int idx, int size);
    int __attribute__(( unused, deprecated )) getShape(int idx);
    int __attribute__(( unused, deprecated )) getDim();

    /* 维度：通过tensor的第二个模版参数获取。  */
    int dim;
};


/*
    划分结构参数
*/
class ShellParam : public Param {

private:
    std::vector<Split*> splits; // 划分组

public:
    ShellParam();
    
    void setSplit(Split* split);
    Split* getSplit(int idx);
    int getNumSplit();

};

}

#endif