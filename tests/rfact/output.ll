; ModuleID = 'mini-c'
source_filename = "mini-c"

define i32 @multiplyNumbers(i32 %n) {
"Func entry":
  %result = alloca i32, align 4
  %n1 = alloca i32, align 4
  store i32 %n, ptr %n1, align 4
  store i32 0, ptr %result, align 4
  store i32 0, ptr %result, align 4
  %n2 = load i32, ptr %n1, align 4
  %icmpgetmp = icmp sge i32 %n2, 1
  %ifcond = icmp ne i1 %icmpgetmp, false
  br i1 %ifcond, label %if, label %else

if:                                               ; preds = %"Func entry"
  %n3 = load i32, ptr %n1, align 4
  %n4 = load i32, ptr %n1, align 4
  %isubtmp = sub i32 %n4, 1
  %multiplyNumbers = call i32 @multiplyNumbers(i32 %isubtmp)
  %multmp = mul i32 %n3, %multiplyNumbers
  store i32 %multmp, ptr %result, align 4
  br label %ifcont

else:                                             ; preds = %"Func entry"
  store i32 1, ptr %result, align 4
  br label %ifcont

ifcont:                                           ; preds = %else, %if
  %result5 = load i32, ptr %result, align 4
  ret i32 %result5
}

define i32 @rfact(i32 %n) {
"Func entry":
  %n1 = alloca i32, align 4
  store i32 %n, ptr %n1, align 4
  %n2 = load i32, ptr %n1, align 4
  %multiplyNumbers = call i32 @multiplyNumbers(i32 %n2)
  ret i32 %multiplyNumbers
}
