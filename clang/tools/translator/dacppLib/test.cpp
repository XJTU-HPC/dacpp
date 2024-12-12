#include <iostream>
#include "Tensor.hpp" // 包含 Tensor 类的头文件
// #include "TensorException.hpp" // 包含 TensorException 类的头文件
#include <stdexcept>


void testTensorConstructors() {
    // 正常情况：使用指针和大小构造 Tensor
    int shape1[] = {2, 3}; // 期望大小 6
    int data1[] = {1, 2, 3, 4, 5, 6};

    dacpp::Tensor<int> tensor1(data1, 6, shape1, 2);
    std::cout << "Tensor constructed successfully with pointer and size." << std::endl;

    // 正常情况：使用向量构造 Tensor
    std::vector<int> dataVec = {1, 2, 3, 4, 5, 6};
    std::vector<int> shapeVec = {2, 3}; // 期望大小 6

    dacpp::Tensor<int> tensor2(dataVec, shapeVec);
    std::cout << "Tensor constructed successfully with vector." << std::endl;

    // 测试无效参数，预期抛出异常
    std::vector<int> invalidShape1 = {2, 2}; // 期望大小 4
    dacpp::Tensor<int> tensorInvalid(data1, 6, invalidShape1.data(), 2); // 应抛出异常

    std::vector<int> invalidShape2 = {2, 2}; // 期望大小 4
    dacpp::Tensor<int> tensorInvalidVec(dataVec, invalidShape2); // 应抛出异常
}
void testSliceFunctions() {
    
    int shape[] = {2, 3};
    int data1[] = {1, 2, 3, 4, 5, 6};
    dacpp::Tensor<int> tensor(data1, 6, shape, 2);

    // 测试第一个 slice 函数
    auto result1 = tensor.slice(1, 5); // 正常情况
    std::cout << "slice(1, 5) succeeded." << std::endl;

    // 测试无效参数，预期抛出异常
    tensor.slice(1, -1); // 应抛出异常

    // 测试第二个 slice 函数
    auto result2 = tensor.slice(1, 2, 6, 1); // 正常情况
    std::cout << "slice(1, 2, 6, 1) succeeded." << std::endl;

    // 测试无效参数，预期抛出异常
    tensor.slice(1, 3, 2); // 应抛出异常
}

void testAddition() {
    int shape[] = {2, 3};
    int shape2[] = {2, 3};
    int data1[] = {1, 2, 3, 4, 5, 6};
    int data2[] = {6, 5, 4, 3, 2, 1};
 
    dacpp::Tensor<int> tensor1(data1, 6, shape, 2);
    dacpp::Tensor<int> tensor2(data2, 6, shape2, 2);

    std::cout << tensor1.getDim();

    dacpp::Tensor<int> result = tensor1 + tensor2;
    tensor1 += tensor2;
    result = tensor1;

    int indices[] = {1, 2}; 
    std::cout << "result[1][2] = " << result.getData(indices) << std::endl;  // 应该输出 7.0
    std::cout << "42";
}

void test_matrix_multiplication() {
 // 测试矩阵 A (2x3) 和 B (3x2) 相乘，结果 C 应该是一个 2x2 矩阵
    int M = 2, K = 3, N = 2;

    // 初始化矩阵 A 和 B 的数据
    int dataA[6] = {1, 2, 3, 4, 5, 6};  // 2x3 矩阵
    int dataB[6] = {7, 8, 9, 10, 11, 12}; // 3x2 矩阵

    // 定义矩阵的形状
    int shapeA[2] = {M, K};
    int shapeB[2] = {K, N};
    
    // 创建 Tensor 对象
    dacpp::Tensor<int> A(dataA, 6, shapeA, 2);
    dacpp::Tensor<int> B(dataB, 6, shapeB, 2);
    
    // 进行矩阵乘法
    dacpp::Tensor<int> C = A * B;

    // 打印期望结果和实际结果
    std::cout << "Matrix A (2x3):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << dataA[i * K + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix B (3x2):\n";
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << dataB[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Expected Result C (2x2):\n";
    int expected_result[4] = {58, 64, 139, 154}; // 计算得出的期望结果
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << expected_result[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Actual Result C (2x2):\n";
    // 修改此处，使用 Tensor 类的公共方法访问数据
    const int* result_data = C.get_data();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << result_data[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // 验证结果
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            assert(result_data[i * N + j] == expected_result[i * N + j]);
        }
    }

    std::cout << "Matrix multiplication passed!" << std::endl;

   
}

void test_large_matrix_multiplication() {
    // 定义矩阵大小
    int M = 32, K = 32, N = 32;

    // 初始化矩阵 A 和 B 的数据
    std::vector<int> dataA(M * K);
    std::vector<int> dataB(K * N);
    std::vector<int> expected_result(M * N, 0);

    // 为矩阵 A 和 B 赋值，简单起见，使用从1递增的数值
    for (int i = 0; i < M * K; ++i) {
        dataA[i] = i + 1;
    }
    for (int i = 0; i < K * N; ++i) {
        dataB[i] = (i + 1) % 10 + 1;  // 值在1-10之间循环
    }

    // 计算期望结果
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += dataA[i * K + k] * dataB[k * N + j];
            }
            expected_result[i * N + j] = sum;
        }
    }

    // 创建 Tensor 对象
    int shapeA[2] = {M, K};
    int shapeB[2] = {K, N};
    dacpp::Tensor<int> A(dataA.data(), M * K, shapeA, 2);
    dacpp::Tensor<int> B(dataB.data(), K * N, shapeB, 2);

    // 进行矩阵乘法
    dacpp::Tensor<int> C = A * B;

    // 打印期望结果和实际结果
    std::cout << "Expected Result C (32x32):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << expected_result[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Actual Result C (32x32):\n";
    // 使用 Tensor 类的公共方法访问数据
    const int* result_data = C.get_data();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << result_data[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // 验证结果
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            assert(result_data[i * N + j] == expected_result[i * N + j]);
        }
    }

    std::cout << "Large matrix multiplication test passed!" << std::endl;
}

int main() {
    // testTensorConstructors();
    // testSliceFunctions();
    // testAddition();
    test_matrix_multiplication();
    test_large_matrix_multiplication();
    // test_high_dim_matrix_multiplication();
    // std::cout << "42" << std::endl;
    return 0;
}
