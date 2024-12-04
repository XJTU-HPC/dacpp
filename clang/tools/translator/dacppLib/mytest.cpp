#include <iostream>
#include "include/myTensor.hpp" // 包含 Tensor 类的头文件
// #include "TensorException.hpp" // 包含 TensorException 类的头文件
#include <stdexcept>


void testTensorConstructors() {

    int data1[] = {1, 2, 3, 4, 5, 6};

    dacpp::Tensor<int, 2> ({2, 3}, data1);
    std::cout << "Tensor constructed successfully with pointer and size." << std::endl;

    // 正常情况：使用向量构造 Tensor
    std::vector<int> dataVec = {1, 2, 3, 4, 5, 6};

    dacpp::Tensor<int, 2> tensor2({2,3}, dataVec);
    std::cout << "Tensor constructed successfully with vector." << std::endl;

    dacpp::Tensor<int, 1> tensor3(6, data1);
    std::cout << "Tensor constructed successfully with pointer." << std::endl;

    // 测试无效参数，预期抛出异常
    // std::vector<int> invalidShape1 = {2, 2}; // 期望大小 4
    // dacpp::Tensor<int, 6> tensorInvalid({2,2}, data1); // 应抛出异常

    // std::vector<int> invalidShape2 = {2, 2}; // 期望大小 4
    // dacpp::Tensor<int, 6> tensorInvalidVec({2,2}, dataVec); // 应抛出异常
}

int main() {
    testTensorConstructors();
    // test_matrix_multiplication();
    return 0;
}
