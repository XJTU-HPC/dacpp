#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string inputFile = argv[1];
    std::string outputFile = "";
    if (argc >= 3) {
        outputFile = argv[2];
    }
    
    std::string command = 
        "translator " + inputFile + " -extra-arg=-std=c++17 "
        "-- -I/data/qinian/clang/lib/clang/12.0.1/include/ "
        "-I/data/zjx/dacpp/clang/tools/translator/dacppLib/include/";
    
    std::cout << "Executing command: " << command << std::endl;

    int result = std::system(command.c_str());
    
    if (result == 0) {
        std::cout << "Command executed successfully!" << std::endl;
        std::cout << outputFile << std::endl;
    } else {
        std::cerr << "Command failed with exit code: " << result << std::endl;
    }

    // 定义 SYCL 源文件路径和目标文件路径
    std::string syclFile = "example.cpp";
    std::string outputFile = "example.out";

    // 定义编译命令（以 DPC++ 为例）
    std::string compileCommand = 
        "dpcpp -std=c++17 -fsycl " + syclFile + " -o " + outputFile;

    // 打印编译命令
    std::cout << "Compiling SYCL file with command: " << compileCommand << std::endl;

    // 执行编译命令
    int result = std::system(compileCommand.c_str());

    // 检查编译结果
    if (result == 0) {
        std::cout << "Compilation successful! Output file: " << outputFile << std::endl;
    } else {
        std::cerr << "Compilation failed with error code: " << result << std::endl;
    }

    return result;
}
