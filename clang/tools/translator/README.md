# DACPP to SYCL Source-to-Source Translator

## Deployment Steps

To run the program under the `test` branch:

```shell
git clone --branch test git@github.com:XJTU-HPC/dacpp.git
cd dacpp
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_INSTALL_PREFIX=/data/user_name/clang -DCMAKE_BUILD_TYPE=debug
mkdir build
cd build
ninja -j8
ninja install
```

The generated output, including the translator and test programs, will be located in `dacpp/build/bin/translator`.

## Directory Structure Description
```
├── README.md                   // Help
├── dacdemo                     // Application
├── dacppLib                    // Syntax support and data organization
│   ├── exception.h             // Error handling
│   ├── Slice.h                 // Slice syntax
│   └── Tensor.h                // Tensor data class
├── dpcppLib                    // SYCL library
│   └── DataReconstructor.h     // Data reconstructor
├── parser                      // Abstract Syntax Tree (AST) parsing
│   ├── ASTParse.h              // AST analysis
│   ├── DacppStructure.h        // Dacpp structure
│   ├── Param.h                 // Parameters
│   └── Split.h                 // Partitioning
├── rewriter                    // Code rewriting
│   ├── dacInfo.h               // DAC data information storage
│   ├── sub_template.h          // SYCL template
│   └── Rewriter.h              // Rewriting logic
├── test                        // Translator test information
│   └── test.h                  // Translator test information printing
├── tests                       // Functional tests
│   ├── src                     // Functional test source files
│   └── CMakeLists.txt
└── CMakeLists.txt
```