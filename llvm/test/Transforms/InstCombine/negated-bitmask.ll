; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; PR53610 - sub(0,and(lshr(X,C1),1)) --> ashr(shl(X,(BW-1)-C1),BW-1)
; PR53610 - sub(C2,and(lshr(X,C1),1)) --> add(ashr(shl(X,(BW-1)-C1),BW-1),C2)

define i8 @neg_mask1_lshr(i8 %a0) {
; CHECK-LABEL: @neg_mask1_lshr(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i8 [[A0:%.*]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = ashr i8 [[TMP1]], 7
; CHECK-NEXT:    ret i8 [[TMP2]]
;
  %shift = lshr i8 %a0, 3
  %mask = and i8 %shift, 1
  %neg = sub i8 0, %mask
  ret i8 %neg
}

define i8 @sub_mask1_lshr(i8 %a0) {
; CHECK-LABEL: @sub_mask1_lshr(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i8 [[A0:%.*]], 6
; CHECK-NEXT:    [[TMP2:%.*]] = ashr i8 [[TMP1]], 7
; CHECK-NEXT:    [[NEG:%.*]] = add nsw i8 [[TMP2]], 10
; CHECK-NEXT:    ret i8 [[NEG]]
;
  %shift = lshr i8 %a0, 1
  %mask = and i8 %shift, 1
  %neg = sub i8 10, %mask
  ret i8 %neg
}

define <4 x i32> @neg_mask1_lshr_vector_uniform(<4 x i32> %a0) {
; CHECK-LABEL: @neg_mask1_lshr_vector_uniform(
; CHECK-NEXT:    [[TMP1:%.*]] = shl <4 x i32> [[A0:%.*]], <i32 28, i32 28, i32 28, i32 28>
; CHECK-NEXT:    [[TMP2:%.*]] = ashr <4 x i32> [[TMP1]], <i32 31, i32 31, i32 31, i32 31>
; CHECK-NEXT:    ret <4 x i32> [[TMP2]]
;
  %shift = lshr <4 x i32> %a0, <i32 3, i32 3, i32 3, i32 3>
  %mask = and <4 x i32> %shift, <i32 1, i32 1, i32 1, i32 1>
  %neg = sub <4 x i32> zeroinitializer, %mask
  ret <4 x i32> %neg
}

define <4 x i32> @neg_mask1_lshr_vector_nonuniform(<4 x i32> %a0) {
; CHECK-LABEL: @neg_mask1_lshr_vector_nonuniform(
; CHECK-NEXT:    [[TMP1:%.*]] = shl <4 x i32> [[A0:%.*]], <i32 28, i32 27, i32 26, i32 25>
; CHECK-NEXT:    [[TMP2:%.*]] = ashr <4 x i32> [[TMP1]], <i32 31, i32 31, i32 31, i32 31>
; CHECK-NEXT:    ret <4 x i32> [[TMP2]]
;
  %shift = lshr <4 x i32> %a0, <i32 3, i32 4, i32 5, i32 6>
  %mask = and <4 x i32> %shift, <i32 1, i32 1, i32 1, i32 1>
  %neg = sub <4 x i32> zeroinitializer, %mask
  ret <4 x i32> %neg
}

define <4 x i32> @sub_mask1_lshr_vector_nonuniform(<4 x i32> %a0) {
; CHECK-LABEL: @sub_mask1_lshr_vector_nonuniform(
; CHECK-NEXT:    [[TMP1:%.*]] = shl <4 x i32> [[A0:%.*]], <i32 28, i32 27, i32 26, i32 25>
; CHECK-NEXT:    [[TMP2:%.*]] = ashr <4 x i32> [[TMP1]], <i32 31, i32 31, i32 31, i32 31>
; CHECK-NEXT:    [[NEG:%.*]] = add nsw <4 x i32> [[TMP2]], <i32 5, i32 0, i32 -1, i32 65556>
; CHECK-NEXT:    ret <4 x i32> [[NEG]]
;
  %shift = lshr <4 x i32> %a0, <i32 3, i32 4, i32 5, i32 6>
  %mask = and <4 x i32> %shift, <i32 1, i32 1, i32 1, i32 1>
  %neg = sub <4 x i32> <i32 5, i32 0, i32 -1, i32 65556>, %mask
  ret <4 x i32> %neg
}

define i8 @sub_mask1_trunc_lshr(i64 %a0) {
; CHECK-LABEL: @sub_mask1_trunc_lshr(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i64 [[A0:%.*]], 48
; CHECK-NEXT:    [[TMP2:%.*]] = ashr i64 [[TMP1]], 63
; CHECK-NEXT:    [[TMP3:%.*]] = trunc nsw i64 [[TMP2]] to i8
; CHECK-NEXT:    [[NEG:%.*]] = add nsw i8 [[TMP3]], 10
; CHECK-NEXT:    ret i8 [[NEG]]
;
  %shift = lshr i64 %a0, 15
  %trunc = trunc i64 %shift to i8
  %mask = and i8 %trunc, 1
  %neg = sub i8 10, %mask
  ret i8 %neg
}

define i32 @sub_sext_mask1_trunc_lshr(i64 %a0) {
; CHECK-LABEL: @sub_sext_mask1_trunc_lshr(
; CHECK-NEXT:    [[TMP1:%.*]] = shl i64 [[A0:%.*]], 48
; CHECK-NEXT:    [[TMP2:%.*]] = ashr i64 [[TMP1]], 63
; CHECK-NEXT:    [[TMP3:%.*]] = trunc nsw i64 [[TMP2]] to i8
; CHECK-NEXT:    [[NARROW:%.*]] = add nsw i8 [[TMP3]], 10
; CHECK-NEXT:    [[NEG:%.*]] = zext i8 [[NARROW]] to i32
; CHECK-NEXT:    ret i32 [[NEG]]
;
  %shift = lshr i64 %a0, 15
  %trunc = trunc i64 %shift to i8
  %mask = and i8 %trunc, 1
  %sext = sext i8 %mask to i32
  %neg = sub i32 10, %sext
  ret i32 %neg
}

define i32 @sub_zext_trunc_lshr(i64 %a0) {
; CHECK-LABEL: @sub_zext_trunc_lshr(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i64 [[A0:%.*]] to i32
; CHECK-NEXT:    [[TMP2:%.*]] = shl i32 [[TMP1]], 16
; CHECK-NEXT:    [[TMP3:%.*]] = ashr i32 [[TMP2]], 31
; CHECK-NEXT:    [[NEG:%.*]] = add nsw i32 [[TMP3]], 10
; CHECK-NEXT:    ret i32 [[NEG]]
;
  %shift = lshr i64 %a0, 15
  %trunc = trunc i64 %shift to i1
  %sext = zext i1 %trunc to i32
  %neg = sub i32 10, %sext
  ret i32 %neg
}

; Negative Test - wrong mask
define i8 @neg_mask2_lshr(i8 %a0) {
; CHECK-LABEL: @neg_mask2_lshr(
; CHECK-NEXT:    [[SHIFT:%.*]] = lshr i8 [[A0:%.*]], 3
; CHECK-NEXT:    [[MASK:%.*]] = and i8 [[SHIFT]], 2
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw i8 0, [[MASK]]
; CHECK-NEXT:    ret i8 [[NEG]]
;
  %shift = lshr i8 %a0, 3
  %mask = and i8 %shift, 2
  %neg = sub i8 0, %mask
  ret i8 %neg
}

; Negative Test - bad shift amount
define i8 @neg_mask2_lshr_outofbounds(i8 %a0) {
; CHECK-LABEL: @neg_mask2_lshr_outofbounds(
; CHECK-NEXT:    ret i8 poison
;
  %shift = lshr i8 %a0, 8
  %mask = and i8 %shift, 2
  %neg = sub i8 0, %mask
  ret i8 %neg
}

; Negative Test - non-constant shift amount
define <2 x i32> @neg_mask1_lshr_vector_var(<2 x i32> %a0, <2 x i32> %a1) {
; CHECK-LABEL: @neg_mask1_lshr_vector_var(
; CHECK-NEXT:    [[SHIFT:%.*]] = lshr <2 x i32> [[A0:%.*]], [[A1:%.*]]
; CHECK-NEXT:    [[MASK:%.*]] = and <2 x i32> [[SHIFT]], <i32 1, i32 1>
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw <2 x i32> zeroinitializer, [[MASK]]
; CHECK-NEXT:    ret <2 x i32> [[NEG]]
;
  %shift = lshr <2 x i32> %a0, %a1
  %mask = and <2 x i32> %shift, <i32 1, i32 1>
  %neg = sub <2 x i32> zeroinitializer, %mask
  ret <2 x i32> %neg
}

; Extra Use - mask
define i8 @neg_mask1_lshr_extrause_mask(i8 %a0) {
; CHECK-LABEL: @neg_mask1_lshr_extrause_mask(
; CHECK-NEXT:    [[SHIFT:%.*]] = lshr i8 [[A0:%.*]], 3
; CHECK-NEXT:    [[MASK:%.*]] = and i8 [[SHIFT]], 1
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw i8 0, [[MASK]]
; CHECK-NEXT:    call void @usei8(i8 [[MASK]])
; CHECK-NEXT:    ret i8 [[NEG]]
;
  %shift = lshr i8 %a0, 3
  %mask = and i8 %shift, 1
  %neg = sub i8 0, %mask
  call void @usei8(i8 %mask)
  ret i8 %neg
}

; Extra Use - shift
define <2 x i32> @neg_mask1_lshr_extrause_lshr(<2 x i32> %a0) {
; CHECK-LABEL: @neg_mask1_lshr_extrause_lshr(
; CHECK-NEXT:    [[SHIFT:%.*]] = lshr <2 x i32> [[A0:%.*]], <i32 3, i32 3>
; CHECK-NEXT:    [[MASK:%.*]] = and <2 x i32> [[SHIFT]], <i32 1, i32 1>
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw <2 x i32> zeroinitializer, [[MASK]]
; CHECK-NEXT:    call void @usev2i32(<2 x i32> [[SHIFT]])
; CHECK-NEXT:    ret <2 x i32> [[NEG]]
;
  %shift = lshr <2 x i32> %a0, <i32 3, i32 3>
  %mask = and <2 x i32> %shift, <i32 1, i32 1>
  %neg = sub <2 x i32> zeroinitializer, %mask
  call void @usev2i32(<2 x i32> %shift)
  ret <2 x i32> %neg
}

define i32 @neg_signbit(i8 %x) {
; CHECK-LABEL: @neg_signbit(
; CHECK-NEXT:    [[TMP1:%.*]] = ashr i8 [[X:%.*]], 7
; CHECK-NEXT:    [[TMP2:%.*]] = sext i8 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[TMP2]]
;
  %s = lshr i8 %x, 7
  %z = zext i8 %s to i32
  %r = sub i32 0, %z
  ret i32 %r
}

define <2 x i64> @neg_signbit_use1(<2 x i32> %x) {
; CHECK-LABEL: @neg_signbit_use1(
; CHECK-NEXT:    [[S:%.*]] = lshr <2 x i32> [[X:%.*]], <i32 31, i32 poison>
; CHECK-NEXT:    call void @usev2i32(<2 x i32> [[S]])
; CHECK-NEXT:    [[TMP1:%.*]] = ashr <2 x i32> [[X]], <i32 31, i32 31>
; CHECK-NEXT:    [[TMP2:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
; CHECK-NEXT:    ret <2 x i64> [[TMP2]]
;
  %s = lshr <2 x i32> %x, <i32 31, i32 poison>
  call void @usev2i32(<2 x i32> %s)
  %z = zext <2 x i32> %s to <2 x i64>
  %r = sub <2 x i64> zeroinitializer, %z
  ret <2 x i64> %r
}

; negative test - extra use

define i8 @neg_signbit_use2(i5 %x) {
; CHECK-LABEL: @neg_signbit_use2(
; CHECK-NEXT:    [[S:%.*]] = lshr i5 [[X:%.*]], 4
; CHECK-NEXT:    [[Z:%.*]] = zext nneg i5 [[S]] to i8
; CHECK-NEXT:    call void @usei8(i8 [[Z]])
; CHECK-NEXT:    [[R:%.*]] = sub nsw i8 0, [[Z]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %s = lshr i5 %x, 4
  %z = zext i5 %s to i8
  call void @usei8(i8 %z)
  %r = sub i8 0, %z
  ret i8 %r
}

; negative test - not negation
; TODO: reduce to zext(x s> -1)

define i32 @neg_not_signbit1(i8 %x) {
; CHECK-LABEL: @neg_not_signbit1(
; CHECK-NEXT:    [[ISNOTNEG:%.*]] = icmp sgt i8 [[X:%.*]], -1
; CHECK-NEXT:    [[R:%.*]] = zext i1 [[ISNOTNEG]] to i32
; CHECK-NEXT:    ret i32 [[R]]
;
  %s = lshr i8 %x, 7
  %z = zext i8 %s to i32
  %r = sub i32 1, %z
  ret i32 %r
}

; negative test - wrong shift amount

define i32 @neg_not_signbit2(i8 %x) {
; CHECK-LABEL: @neg_not_signbit2(
; CHECK-NEXT:    [[S:%.*]] = lshr i8 [[X:%.*]], 6
; CHECK-NEXT:    [[Z:%.*]] = zext nneg i8 [[S]] to i32
; CHECK-NEXT:    [[R:%.*]] = sub nsw i32 0, [[Z]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %s = lshr i8 %x, 6
  %z = zext i8 %s to i32
  %r = sub i32 0, %z
  ret i32 %r
}

; negative test - wrong shift opcode

define i32 @neg_not_signbit3(i8 %x) {
; CHECK-LABEL: @neg_not_signbit3(
; CHECK-NEXT:    [[S:%.*]] = ashr i8 [[X:%.*]], 7
; CHECK-NEXT:    [[Z:%.*]] = zext i8 [[S]] to i32
; CHECK-NEXT:    [[R:%.*]] = sub nsw i32 0, [[Z]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %s = ashr i8 %x, 7
  %z = zext i8 %s to i32
  %r = sub i32 0, %z
  ret i32 %r
}

define i32 @neg_mask(i32 %x, i16 %y) {
; CHECK-LABEL: @neg_mask(
; CHECK-NEXT:    [[S:%.*]] = sext i16 [[Y:%.*]] to i32
; CHECK-NEXT:    [[SUB1:%.*]] = sub nsw i32 [[X:%.*]], [[S]]
; CHECK-NEXT:    [[ISNEG:%.*]] = icmp slt i16 [[Y]], 0
; CHECK-NEXT:    [[R:%.*]] = select i1 [[ISNEG]], i32 [[SUB1]], i32 0
; CHECK-NEXT:    ret i32 [[R]]
;
  %s = sext i16 %y to i32
  %sub1 = sub nsw i32 %x, %s
  %sh = lshr i16 %y, 15
  %z = zext i16 %sh to i32
  %sub2 = sub nsw i32 0, %z
  %r = and i32 %sub1, %sub2
  ret i32 %r
}

define i32 @neg_mask_const(i16 %x) {
; CHECK-LABEL: @neg_mask_const(
; CHECK-NEXT:    [[S:%.*]] = sext i16 [[X:%.*]] to i32
; CHECK-NEXT:    [[SUB1:%.*]] = sub nsw i32 1000, [[S]]
; CHECK-NEXT:    [[ISNEG:%.*]] = icmp slt i16 [[X]], 0
; CHECK-NEXT:    [[R:%.*]] = select i1 [[ISNEG]], i32 [[SUB1]], i32 0
; CHECK-NEXT:    ret i32 [[R]]
;
  %s = sext i16 %x to i32
  %sub1 = sub nsw i32 1000, %s
  %sh = lshr i16 %x, 15
  %z = zext i16 %sh to i32
  %sub2 = sub nsw i32 0, %z
  %r = and i32 %sub1, %sub2
  ret i32 %r
}

declare void @usei8(i8)
declare void @usev2i32(<2 x i32>)