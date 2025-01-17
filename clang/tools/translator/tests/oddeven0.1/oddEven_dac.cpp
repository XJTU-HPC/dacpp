#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}
const int N = 1024;  // 假设数组的大小为1024


// 交换函数
void swap(vector<int>& array, int i, int j) {
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

shell dacpp::list ODDEVEN(const dacpp::Tensor<int,1> & array, dacpp::Tensor<int,1> & array_out) {
    //dacpp::Index idx1("idx1");
    dacpp::split S1(2,2);
    
    //binding(idx1,S1);
    dacpp::list dataList{array[{S1}], array_out[{S1}]};
    return dataList;
}

calc void oddeven(dacpp::Tensor<int,1> & array, dacpp::Tensor<int,1>  & array_out) {
    if (array[0] > array[1]) {
        array_out[0] = array[1];
        array_out[1] = array[0];
    }
}

// 奇偶归并排序的核心操作
void oddEvenMergeSort(vector<int>& array, int n) {
    dacpp::Tensor<int, 1> array_tensor(array);
    vector<int> array_out(N);
    dacpp::Tensor<int, 1> array_out_tensor(array_out);

    // 每一轮排序进行多次比较
    for (int phase = 0; phase < n; phase++) {
        // 奇数阶段：比较相邻的奇数索引
        ODDEVEN(array_tensor,array_out_tensor) <-> oddeven;
        dacpp::Tensor<int, 1> array2_tensor = array_out_tensor[{1,1024}];
        dacpp::Tensor<int, 1> array_out2_tensor = array2_tensor;

        ODDEVEN(array2_tensor,array_out2_tensor) <-> oddeven;

        for(int i = 1;i < N-1; i++){
            array_tensor[i] = array_out2_tensor[i-1];
        }
        array_tensor[0] = array_out_tensor[0];
        array_tensor[N-1] = array_out_tensor[N-1];
    }
    array_tensor.print();
}

// 主函数：初始化数据并调用奇偶归并排序
int main() {
    vector<int> array(N);

    // 初始化数据
    srand(time(0));
    for (int i = 0; i < N; i++) {
        array[i] = rand() % 1000;  // 随机生成0到1000之间的整数
    }

    // 打印排序前的数组（前10个）
    std::cout << "Array before sorting (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    // 执行奇偶归并排序
    oddEvenMergeSort(array, N);


    return 0;
}
