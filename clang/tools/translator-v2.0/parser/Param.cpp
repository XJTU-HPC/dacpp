#include <string>

#include "Param.h"

/*
    参数
*/
dacppTranslator::Param::Param() {
}

void dacppTranslator::Param::setRw(bool rw) {
    this->rw = rw;
}

bool dacppTranslator::Param::getRw() {
    return rw;
}

void dacppTranslator::Param::setType(std::string type) {
    this->type = type;
    if(type.find("Tensor<int>") != -1) {
        this->basicType = "int";
    }
    else if(type.find("Tensor<float>") != -1) {
        this->basicType =  "float";
    }
    else {
        this->basicType =  "double";
    }
}

std::string dacppTranslator::Param::getType() {
    return type;
}

std::string dacppTranslator::Param::getBasicType() {
    return basicType;
}

void dacppTranslator::Param::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::Param::getName() {
    return name;
}

void dacppTranslator::Param::setShape(int size) {
    shape.push_back(size);
}

void dacppTranslator::Param::setShape(int idx, int size) {
    shape[idx] = size;
}

int dacppTranslator::Param::getShape(int idx) {
    return shape[idx];
}

int dacppTranslator::Param::getDim() {
    return shape.size();
}

/*
    划分结构参数
*/
dacppTranslator::ShellParam::ShellParam() {
}

void dacppTranslator::ShellParam::setSplit(Split* split) {
    splits.push_back(split);
}

dacppTranslator::Split* dacppTranslator::ShellParam::getSplit(int idx) {
    return splits[idx];
}

int dacppTranslator::ShellParam::getNumSplit() {
    return splits.size();
}