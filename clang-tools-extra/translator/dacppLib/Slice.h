#ifndef SLICE_H
#define SLICE_H

#include <vector>
#include <any>

namespace dacpp {

typedef double index;

struct Slice {
    bool isIndex_;
    bool isAll_;
    int start_;
    int end_;
    int stride_;

    Slice() : isIndex_(false), isAll_(true) {}
    Slice(int idx) : isIndex_(false), isAll_(false), start_(idx), end_(idx + 1), stride_(1) {}
    Slice(int start, int end, int stride = 1) : isIndex_(false), isAll_(false), start_(start), end_(end), stride_(stride) {}
    Slice(index i) : isIndex_(true) {}
};

}

#endif