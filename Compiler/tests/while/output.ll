; ModuleID = 'mini-c'
source_filename = "mini-c"

@test = common global i32 0
@f = common global float 0.000000e+00
@b = common global i1 false

declare i32 @print_int(i32)

define i32 @While(i32 %n) {
"Func entry":
  %result = alloca i32, align 4
  %n1 = alloca i32, align 4
  store i32 %n, ptr %n1, align 4
  store i32 0, ptr %result, align 4
  store i32 12, ptr @test, align 4
  %test = load i32, ptr @test, align 4
  store i32 0, ptr %result, align 4
  %test2 = load i32, ptr @test, align 4
  %print_int = call i32 @print_int(i32 %test2)
  br label %condwhile

condwhile:                                        ; preds = %loopwhile, %"Func entry"
  %result3 = load i32, ptr %result, align 4
  %iLTtmp = icmp slt i32 %result3, 10
  %whilecond = icmp ne i1 %iLTtmp, false
  br i1 %whilecond, label %loopwhile, label %afterwhile

loopwhile:                                        ; preds = %condwhile
  %result4 = load i32, ptr %result, align 4
  %addtmp = add i32 %result4, 1
  store i32 %addtmp, ptr %result, align 4
  br label %condwhile

afterwhile:                                       ; preds = %condwhile
  %result5 = load i32, ptr %result, align 4
  ret i32 %result5
}
