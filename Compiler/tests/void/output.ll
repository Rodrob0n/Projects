; ModuleID = 'mini-c'
source_filename = "mini-c"

declare i32 @print_int(i32)

define void @Void() {
"Func entry":
  %result = alloca i32, align 4
  store i32 0, ptr %result, align 4
  store i32 0, ptr %result, align 4
  %result1 = load i32, ptr %result, align 4
  %print_int = call i32 @print_int(i32 %result1)
  br label %condwhile

condwhile:                                        ; preds = %loopwhile, %"Func entry"
  %result2 = load i32, ptr %result, align 4
  %iLTtmp = icmp slt i32 %result2, 10
  %whilecond = icmp ne i1 %iLTtmp, false
  br i1 %whilecond, label %loopwhile, label %afterwhile

loopwhile:                                        ; preds = %condwhile
  %result3 = load i32, ptr %result, align 4
  %addtmp = add i32 %result3, 1
  store i32 %addtmp, ptr %result, align 4
  %result4 = load i32, ptr %result, align 4
  %print_int5 = call i32 @print_int(i32 %result4)
  br label %condwhile

afterwhile:                                       ; preds = %condwhile
  ret void
}
