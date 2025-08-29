; ModuleID = 'mini-c'
source_filename = "mini-c"

declare i32 @print_int(i32)

define i32 @fibonacci(i32 %n) {
"Func entry":
  %total = alloca i32, align 4
  %c = alloca i32, align 4
  %next = alloca i32, align 4
  %second = alloca i32, align 4
  %first = alloca i32, align 4
  %n1 = alloca i32, align 4
  store i32 %n, ptr %n1, align 4
  store i32 0, ptr %first, align 4
  store i32 0, ptr %second, align 4
  store i32 0, ptr %next, align 4
  store i32 0, ptr %c, align 4
  store i32 0, ptr %total, align 4
  %n2 = load i32, ptr %n1, align 4
  %print_int = call i32 @print_int(i32 %n2)
  store i32 0, ptr %first, align 4
  store i32 1, ptr %second, align 4
  store i32 1, ptr %c, align 4
  store i32 0, ptr %total, align 4
  br label %condwhile

condwhile:                                        ; preds = %ifcont, %"Func entry"
  %c3 = load i32, ptr %c, align 4
  %n4 = load i32, ptr %n1, align 4
  %iLTtmp = icmp slt i32 %c3, %n4
  %whilecond = icmp ne i1 %iLTtmp, false
  br i1 %whilecond, label %loopwhile, label %afterwhile

loopwhile:                                        ; preds = %condwhile
  %c5 = load i32, ptr %c, align 4
  %icmpetmp = icmp sle i32 %c5, 1
  %ifcond = icmp ne i1 %icmpetmp, false
  br i1 %ifcond, label %if, label %else

afterwhile:                                       ; preds = %condwhile
  %total18 = load i32, ptr %total, align 4
  %print_int19 = call i32 @print_int(i32 %total18)
  %total20 = load i32, ptr %total, align 4
  ret i32 %total20

if:                                               ; preds = %loopwhile
  %c6 = load i32, ptr %c, align 4
  store i32 %c6, ptr %next, align 4
  br label %ifcont

else:                                             ; preds = %loopwhile
  %first7 = load i32, ptr %first, align 4
  %second8 = load i32, ptr %second, align 4
  %addtmp = add i32 %first7, %second8
  store i32 %addtmp, ptr %next, align 4
  %second9 = load i32, ptr %second, align 4
  store i32 %second9, ptr %first, align 4
  %next10 = load i32, ptr %next, align 4
  store i32 %next10, ptr %second, align 4
  br label %ifcont

ifcont:                                           ; preds = %else, %if
  %next11 = load i32, ptr %next, align 4
  %print_int12 = call i32 @print_int(i32 %next11)
  %c13 = load i32, ptr %c, align 4
  %addtmp14 = add i32 %c13, 1
  store i32 %addtmp14, ptr %c, align 4
  %total15 = load i32, ptr %total, align 4
  %next16 = load i32, ptr %next, align 4
  %addtmp17 = add i32 %total15, %next16
  store i32 %addtmp17, ptr %total, align 4
  br label %condwhile
}
