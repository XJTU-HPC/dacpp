#!/usr/bin/env bash

exec 2>/dev/null

#Delelte all tmporary files
rm -rf ./tmp
mkdir ./tmp

# Edit examples here
examples=("matMul"
    # "block_mat_mul"
    "waveEquation1.0"
    "stencil1.0"
    "jacobi1.0"
    "FOuLa1.0"
)


WORK_DIR=../../../../clang/tools/translator
INCLUDE_DIRS=("$WORK_DIR/dpcppLib/include/"
    "$WORK_DIR/dacppLib/include/"
    "$WORK_DIR/rewriter/include/"
)
SRC_DIRS=("$WORK_DIR/dpcppLib/lib/"
    "$WORK_DIR/rewriter/lib/"
    )
SRC_FILES=()
for dir in ${SRC_DIRS[@]}; do
    SRC_FILES+=($(find "$dir" -type f -name "*.cpp"))
done

echo "------------------------------------------------------------------------------------------"
echo "DACPP to SYCL transpilation test"
echo

# DACPP to SYCL transpilation test
for dir in ${examples[@]}; do
    dacpp_file=$(find "./$dir" -type f -name "*.dac.cpp" | head -n 1)
    if [ -z "$dacpp_file" ]; then
        echo "Example $dir: DACPP source file not found"
        continue
    fi
    mkdir -p "./tmp/$dir"
    cp "$dacpp_file" "./tmp/$dir"
    ../../../../build/bin/translator/translator "./tmp/$dacpp_file" \
    -extra-arg=-std=c++17 -- -I/data/qinian/clang/lib/clang/12.0.1/include/ 
    sycl_file=$(find "./tmp/$dir" -type f -name "*.dac_sycl.cpp")
    if [ -z "$sycl_file" ]; then
        echo "Example $dir: DACPP to SYCL transpilation failed"
    else
        echo "Example $dir: DACPP to SYCL transpilation succeeded"
    fi
done

echo "------------------------------------------------------------------------------------------"
echo "Generated SYCL files compilation test"
echo

# Generated SYCL files compilation test
for dir in ${examples[@]}; do
    sycl_file=$(find "./tmp/$dir" -type f -name "*.dac_sycl.cpp")
    if [ -z "$sycl_file" ]; then
        continue
    fi
    # exe_name="${sycl_file%.dac_sycl.cpp}"
    # exe_name="${exe_name#./tmp/$dir/}"
    icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
    "$sycl_file" \
    "${SRC_FILES[@]}" \
    "${INCLUDE_DIRS[@]/#/-I}" \
    -o "$dir" 
    exe_file=$(find "./tmp/$dir" -type f -name "$dir")
    if [ -z "$exe_file" ]; then
        echo "Example $dir: SYCL compilation failed"
    else
        echo "Example $dir: SYCL compilation succeeded"
    fi
done

echo "------------------------------------------------------------------------------------------"
echo "SYCL files ouput result test"
echo

# SYCL files ouput result test
for dir in ${examples[@]}; do
    exe_file=$(find "./tmp/$dir" -type f -name "$dir")
    if [ -z "$exe_file" ]; then
        continue
    fi
    exe_file="${exe_file#./tmp/$dir/}"
    "./tmp/$dir/$exe_file" > "./tmp/$dir/$exe_file.out" 
    std_res=$(find "./$dir/" -type f -name "*.out" | head -n 1)
    if diff -y --suppress-common-lines "./tmp/$dir/$exe_file.out" "$std_res"; then
        echo "Example $dir: execution test succeeded"
    else
        echo "Example $dir: execution test failed with some different lines listed above"
    fi
done

echo "------------------------------------------------------------------------------------------"

# rm -rf ./tmp