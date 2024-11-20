#ifndef SLICE_H
#define SLICE_H

#include <string>
#include <vector>
#include <set>
#include <any>

namespace dacpp {


// 降维算子
class Index {
private:
    std::string variable_;
public:
    Index(std::string variable) : variable_(variable) {}
};


// 规则分区算子
class RegularSplit {
private:
    std::string variable_;
    int size_;
    int stride_;

public:
    RegularSplit(std::string variable, int size, int stride) : variable_(variable), size_(size), stride_(stride) {}
}; 


// 不规则分区算子
class IrregularSlice {
private:
    std::string pattern_;
    std::set<std::string> variables_;
    std::vector<std::vector<std::string>> indices_;

public:
    IrregularSlice(std::string pattern) : pattern_(pattern) {
        int left = pattern_.find("("), right = pattern_.find(")");
        while(left != std::string::npos && right != std::string::npos) {
            std::vector<std::string> index;
            int cur = left + 1;
            while(1) {
                if(pattern_.find(",", cur) != std::string::npos && pattern_.find(",", cur) < right) {
                    index.push_back(pattern_.substr(cur, pattern_.find(",", cur) - cur));
                    cur = pattern_.find(",", cur) + 1;
                }
                else {
                    index.push_back(pattern_.substr(cur, right - cur));
                    break;
                }
            }
            indices_.push_back(index);
            left = pattern_.find("(", left + 1), right = pattern_.find(")", right + 1);
        }
    }
};


class ConformalSlice {};


struct Slice {
    bool isIndex_;
    bool isAll_;
    int start_;
    int end_;
    int stride_;

    Slice() : isIndex_(false), isAll_(true) {}
    Slice(int idx) : isIndex_(false), isAll_(false), start_(idx), end_(idx + 1), stride_(1) {}
    Slice(int start, int end, int stride = 1) : isIndex_(false), isAll_(false), start_(start), end_(end), stride_(stride) {}
    Slice(Index i) : isIndex_(true) {}
};


} // namespace dacpp


#endif