; ModuleID = 'mini-c'
source_filename = "mini-c"

declare i32 @print_int(i32)

define i32 @addNumbers(i32 %n) {
"Func entry":
  %result = alloca i32, align 4
  %n1 = alloca i32, align 4
  store i32 %n, ptr %n1, align 4
  store i32 0, ptr %result, align 4
  store i32 0, ptr %result, align 4
  %n2 = load i32, ptr %n1, align 4
  %icmpnetmp = icmp ne i32 %n2, 0
  %ifcond = icmp ne i1 %icmpnetmp, false
  br i1 %ifcond, label %if, label %else

if:                                               ; preds = %"Func entry"
  %n3 = load i32, ptr %n1, align 4
  %n4 = load i32, ptr %n1, align 4
  %isubtmp = sub i32 %n4, 1
  %addNumbers = call i32 @addNumbers(i32 %isubtmp)
  %addtmp = add i32 %n3, %addNumbers
  store i32 %addtmp, ptr %result, align 4
  br label %ifcont

else:                                             ; preds = %"Func entry"
  %n5 = load i32, ptr %n1, align 4
  store i32 %n5, ptr %result, align 4
  br label %ifcont

ifcont:                                           ; preds = %else, %if
  %result6 = load i32, ptr %result, align 4
  %print_int = call i32 @print_int(i32 %result6)
  %result7 = load i32, ptr %result, align 4
  ret i32 %result7
}

define i32 @recursion_driver(i32 %num) {
"Func entry":
  %num1 = alloca i32, align 4
  store i32 %num, ptr %num1, align 4
  %num2 = load i32, ptr %num1, align 4
  %addNumbers = call i32 @addNumbers(i32 %num2)
  ret i32 %addNumbers
}
