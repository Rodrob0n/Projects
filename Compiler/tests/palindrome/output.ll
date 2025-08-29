; ModuleID = 'mini-c'
source_filename = "mini-c"

define i1 @palindrome(i32 %number) {
"Func entry":
  %result = alloca i1, align 1
  %rmndr = alloca i32, align 4
  %rev = alloca i32, align 4
  %t = alloca i32, align 4
  %number1 = alloca i32, align 4
  store i32 %number, ptr %number1, align 4
  store i32 0, ptr %t, align 4
  store i32 0, ptr %rev, align 4
  store i32 0, ptr %rmndr, align 4
  store i1 false, ptr %result, align 1
  store i32 0, ptr %rev, align 4
  store i1 false, ptr %result, align 1
  %number2 = load i32, ptr %number1, align 4
  store i32 %number2, ptr %t, align 4
  br label %condwhile

condwhile:                                        ; preds = %loopwhile, %"Func entry"
  %number3 = load i32, ptr %number1, align 4
  %icmpgttmp = icmp sgt i32 %number3, 0
  %whilecond = icmp ne i1 %icmpgttmp, false
  br i1 %whilecond, label %loopwhile, label %afterwhile

loopwhile:                                        ; preds = %condwhile
  %number4 = load i32, ptr %number1, align 4
  %modtmp = srem i32 %number4, 10
  store i32 %modtmp, ptr %rmndr, align 4
  %rev5 = load i32, ptr %rev, align 4
  %multmp = mul i32 %rev5, 10
  %rmndr6 = load i32, ptr %rmndr, align 4
  %addtmp = add i32 %multmp, %rmndr6
  store i32 %addtmp, ptr %rev, align 4
  %number7 = load i32, ptr %number1, align 4
  %idivtmp = sdiv i32 %number7, 10
  store i32 %idivtmp, ptr %number1, align 4
  br label %condwhile

afterwhile:                                       ; preds = %condwhile
  %t8 = load i32, ptr %t, align 4
  %rev9 = load i32, ptr %rev, align 4
  %icmpeqtmp = icmp eq i32 %t8, %rev9
  %ifcond = icmp ne i1 %icmpeqtmp, false
  br i1 %ifcond, label %if, label %else

if:                                               ; preds = %afterwhile
  store i1 true, ptr %result, align 1
  br label %ifcont

else:                                             ; preds = %afterwhile
  store i1 false, ptr %result, align 1
  br label %ifcont

ifcont:                                           ; preds = %else, %if
  %result10 = load i1, ptr %result, align 1
  ret i1 %result10
}
