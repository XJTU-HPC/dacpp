# NOTE: Assertions have been autogenerated by utils/update_mca_test_checks.py
# RUN: llvm-mca -mtriple=riscv64-unknown-unknown -mcpu=syntacore-scr3-rv64 --iterations=1 < %s | FileCheck %s --check-prefixes=CHECK,RV64
# RUN: llvm-mca -mtriple=riscv32-unknown-unknown -mcpu=syntacore-scr3-rv32 --iterations=1 < %s | FileCheck %s --check-prefixes=CHECK,RV32

div a0, a0, a0
mul t0, a0, t0
add t1, a0, t0
add t2, t2, t2
div a1, a1, a1
mul s0, a1, s0
add s1, s0, s1
add s2, s2, s2

# CHECK:      Iterations:        1
# CHECK-NEXT: Instructions:      8

# RV32-NEXT:  Total Cycles:      25
# RV64-NEXT:  Total Cycles:      31

# CHECK-NEXT: Total uOps:        8

# CHECK:      Dispatch Width:    1

# RV32-NEXT:  uOps Per Cycle:    0.32
# RV32-NEXT:  IPC:               0.32
# RV32-NEXT:  Block RThroughput: 16.0

# RV64-NEXT:  uOps Per Cycle:    0.26
# RV64-NEXT:  IPC:               0.26
# RV64-NEXT:  Block RThroughput: 22.0

# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects (U)

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]    Instructions:

# RV32-NEXT:   1      8     8.00                        div	a0, a0, a0
# RV64-NEXT:   1      11    11.00                       div	a0, a0, a0

# CHECK-NEXT:  1      2     1.00                        mul	t0, a0, t0
# CHECK-NEXT:  1      1     1.00                        add	t1, a0, t0
# CHECK-NEXT:  1      1     1.00                        add	t2, t2, t2

# RV32-NEXT:   1      8     8.00                        div	a1, a1, a1
# RV64-NEXT:   1      11    11.00                       div	a1, a1, a1

# CHECK-NEXT:  1      2     1.00                        mul	s0, a1, s0
# CHECK-NEXT:  1      1     1.00                        add	s1, s1, s0
# CHECK-NEXT:  1      1     1.00                        add	s2, s2, s2

# CHECK:      Resources:

# RV32-NEXT:  [0]   - SCR3RV32_ALU
# RV32-NEXT:  [1]   - SCR3RV32_CFU
# RV32-NEXT:  [2]   - SCR3RV32_DIV
# RV32-NEXT:  [3]   - SCR3RV32_LSU
# RV32-NEXT:  [4]   - SCR3RV32_MUL

# RV64-NEXT:  [0]   - SCR3RV64_ALU
# RV64-NEXT:  [1]   - SCR3RV64_CFU
# RV64-NEXT:  [2]   - SCR3RV64_DIV
# RV64-NEXT:  [3]   - SCR3RV64_LSU
# RV64-NEXT:  [4]   - SCR3RV64_MUL

# CHECK:      Resource pressure per iteration:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]

# RV32-NEXT:  4.00    -     16.00   -     2.00
# RV64-NEXT:  4.00    -     22.00   -     2.00

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    Instructions:

# RV32-NEXT:   -      -     8.00    -      -     div	a0, a0, a0
# RV64-NEXT:   -      -     11.00   -      -     div	a0, a0, a0

# CHECK-NEXT:  -      -      -      -     1.00   mul	t0, a0, t0
# CHECK-NEXT: 1.00    -      -      -      -     add	t1, a0, t0
# CHECK-NEXT: 1.00    -      -      -      -     add	t2, t2, t2

# RV32-NEXT:   -      -     8.00    -      -     div	a1, a1, a1
# RV64-NEXT:   -      -     11.00   -      -     div	a1, a1, a1

# CHECK-NEXT:  -      -      -      -     1.00   mul	s0, a1, s0
# CHECK-NEXT: 1.00    -      -      -      -     add	s1, s1, s0
# CHECK-NEXT: 1.00    -      -      -      -     add	s2, s2, s2