#include <string>

#include "Split.h"

/*
    划分父类
*/
dacppTranslator::Split::Split(clang::ValueDecl *D) {
    this->D = D;
}

dacppTranslator::Split::Split(clang::ValueDecl *D, std::string id, int dimIdx) {
    this->D = D;
    this->id = id;
    this->dimIdx = dimIdx;
}

void dacppTranslator::Split::setId(std::string id) {
    this->id = id;
}

std::string dacppTranslator::Split::getId() {
    return id;
}

void dacppTranslator::Split::setDimIdx(int dimIdx) {
    this->dimIdx = dimIdx;
}

int dacppTranslator::Split::getDimIdx() {
    return dimIdx;
}

/*
    降维划分
*/
dacppTranslator::IndexSplit::IndexSplit(clang::ValueDecl *D)  : Split(D){
}

dacppTranslator::IndexSplit::IndexSplit(clang::ValueDecl *D, std::string id, int dimIdx, int splitNumber) : Split(D, id, dimIdx) {
    this->splitNumber = splitNumber;
}

void dacppTranslator::IndexSplit::setSplitNumber(int splitNumber) {
    this->splitNumber = splitNumber;
}

int dacppTranslator::IndexSplit::getSplitNumber() {
    return splitNumber;
}

/*
    规则分区划分
*/
dacppTranslator::RegularSplit::RegularSplit(clang::ValueDecl *D): Split(D) {
}

dacppTranslator::RegularSplit::RegularSplit(clang::ValueDecl *D, std::string id, int dimIdx, int splitSize, int splitStride, int splitNumber) : Split(D, id, dimIdx) {
    this->splitSize = splitSize;
    this->splitStride = splitStride;
    this->splitNumber = splitNumber;
}

void dacppTranslator::RegularSplit::setSplitSize(int splitSize) {
    this->splitSize = splitSize;
}

int dacppTranslator::RegularSplit::getSplitSize() {
    return splitSize;
}

void dacppTranslator::RegularSplit::setSplitStride(int splitStride) {
    this->splitStride = splitStride;
}

int dacppTranslator::RegularSplit::getSplitStride() {
    return splitStride;
}

void dacppTranslator::RegularSplit::setSplitNumber(int splitNumber) {
    this->splitNumber = splitNumber;
}

int dacppTranslator::RegularSplit::getSplitNumber() {
    return splitNumber;
}