#include "../dacppLib/include/Slice.h"
#include "../dacppLib/include/Tensor.hpp"

namespace dacpp {
typedef std::vector<std::any> list;
}

using dacpp::Tensor;
shell void waveEqShell(Tensor<float>& matCur, Tensor<float>& matPrev, Tensor<float>& matNext) {
    split sp1(3, 1), sp2(3, 1);
    index idx1, idx2;
    binding(sp1, idx1+1);
    binding(sp2, idx2+1);
    float courant = 0.4;
    list dataList{matCur[{sp1}][{sp2}], matPrev[{idx1}][{idx2}], matNext[{idx1}][{idx2}], courant};
}

calc void waveEq(Tensor<float>& cur, Tensor<float>& prev, Tensor<float>& next, float c) {
    next = 2*cur[1][1] - prev + c*c*(cur[2][1]+cur[0][1]+cur[1][2]+cur[1][0]-4*cur[1][1])
}

int main() {
    Tensor<float> matCur;
    Tensor<float> matPrev;
    Tensor<float> matNext;
    
    ......
    
    for(int i=0;i<iteration;i++) {
        waveEqShell(matCur, matPrev, matNext) <-> waveEq;
        
        std::swap(matPrev, matCur);
        std::swap(matCur, matNext);
    }
    
    return 0;
}