; RUN: llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s | FileCheck -check-prefix=GFX12 %s

global_atomic_csub v1, v0, v2, s[0:1] offset:64 th:TH_ATOMIC_RETURN
// GFX12: global_atomic_sub_clamp_u32 v1, v0, v2, s[0:1] offset:64 th:TH_ATOMIC_RETURN ; encoding: [0x00,0xc0,0x0d,0xee,0x01,0x00,0x10,0x01,0x00,0x40,0x00,0x00]

global_atomic_csub_u32 v1, v0, v2, s[0:1] offset:64 th:TH_ATOMIC_RETURN
// GFX12: global_atomic_sub_clamp_u32 v1, v0, v2, s[0:1] offset:64 th:TH_ATOMIC_RETURN ; encoding: [0x00,0xc0,0x0d,0xee,0x01,0x00,0x10,0x01,0x00,0x40,0x00,0x00]

flat_atomic_csub_u32 v1, v[0:1], v2 offset:64 th:TH_ATOMIC_RETURN
// GFX12: flat_atomic_sub_clamp_u32 v1, v[0:1], v2 offset:64 th:TH_ATOMIC_RETURN ; encoding: [0x7c,0xc0,0x0d,0xec,0x01,0x00,0x10,0x01,0x00,0x40,0x00,0x00]

flat_atomic_fmax v[0:1], v2 offset:64
// GFX12: flat_atomic_max_num_f32 v[0:1], v2 offset:64 ; encoding: [0x7c,0x80,0x14,0xec,0x00,0x00,0x00,0x01,0x00,0x40,0x00,0x00]

flat_atomic_max_f32 v[0:1], v2 offset:64
// GFX12: flat_atomic_max_num_f32 v[0:1], v2 offset:64 ; encoding: [0x7c,0x80,0x14,0xec,0x00,0x00,0x00,0x01,0x00,0x40,0x00,0x00]

flat_atomic_fmin v[0:1], v2 offset:64
// GFX12: flat_atomic_min_num_f32 v[0:1], v2 offset:64 ; encoding: [0x7c,0x40,0x14,0xec,0x00,0x00,0x00,0x01,0x00,0x40,0x00,0x00]

flat_atomic_min_f32 v[0:1], v2 offset:64
// GFX12: flat_atomic_min_num_f32 v[0:1], v2 offset:64 ; encoding: [0x7c,0x40,0x14,0xec,0x00,0x00,0x00,0x01,0x00,0x40,0x00,0x00]

global_atomic_fmax v0, v2, s[0:1] offset:64
// GFX12: global_atomic_max_num_f32 v0, v2, s[0:1] offset:64 ; encoding: [0x00,0x80,0x14,0xee,0x00,0x00,0x00,0x01,0x00,0x40,0x00,0x00]

global_atomic_max_f32 v0, v2, s[0:1] offset:64
// GFX12: global_atomic_max_num_f32 v0, v2, s[0:1] offset:64 ; encoding: [0x00,0x80,0x14,0xee,0x00,0x00,0x00,0x01,0x00,0x40,0x00,0x00]

global_atomic_fmin v0, v2, s[0:1] offset:64
// GFX12: global_atomic_min_num_f32 v0, v2, s[0:1] offset:64 ; encoding: [0x00,0x40,0x14,0xee,0x00,0x00,0x00,0x01,0x00,0x40,0x00,0x00]

global_atomic_min_f32 v0, v2, s[0:1] offset:64
// GFX12: global_atomic_min_num_f32 v0, v2, s[0:1] offset:64 ; encoding: [0x00,0x40,0x14,0xee,0x00,0x00,0x00,0x01,0x00,0x40,0x00,0x00]