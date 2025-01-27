// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-windows-msvc -gcodeview -debug-info-kind=limited -o - | FileCheck %s

struct b {
  b(char *);
  ~b();
};
struct a {
  ~a();
};
struct {
  b c;
  const a &d;
} e[]{nullptr, {}};

// CHECK: define internal void @__cxx_global_array_dtor(ptr noundef %0)
// CHECK-SAME: !dbg ![[SUBPROGRAM:[0-9]+]] {
// CHECK: arraydestroy.body
// CHECK: %arraydestroy.elementPast =
// CHECK-SAME: !dbg ![[LOCATION:[0-9]+]]
// CHECK: call void @"??1<unnamed-type-e>@@QEAA@XZ"(ptr {{[^,]*}} %arraydestroy.element)
// CHECK-SAME: !dbg ![[LOCATION]]
// CHECK: ![[SUBPROGRAM]] = distinct !DISubprogram(name: "__cxx_global_array_dtor"
// CHECK-SAME: flags: DIFlagArtificial
// CHECK: ![[LOCATION]] = !DILocation(line: 0,
