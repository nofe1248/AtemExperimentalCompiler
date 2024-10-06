!si64 = !atemhir.int<s, 64>
!si32 = !atemhir.int<s, 32>
!si128 = !atemhir.int<s, 128>
!ui64 = !atemhir.int<u, 64>
!f64 = !atemhir.fp64

atemhir.function @main() -> !atemhir.int<s, 64> {
    %6 = atemhir.allocate_var !atemhir.fp80, !atemhir.ptr<!atemhir.fp80>, ["some_local_var"]
    %0 = atemhir.constant #atemhir.int<-42> : !si64
    %1 = atemhir.constant #atemhir.int<42> : !si64
    %2 = atemhir.constant #atemhir.int<42> : !si64
    %3 = atemhir.constant #atemhir.fp<-4.2e3> : !atemhir.fp128
    %4 = atemhir.zeroinit : !atemhir.bool
    %5 = atemhir.compare lt, %0, %1 : !atemhir.int<s, 64>

    %11 = atemhir.if %5 {
        atemhir.yield %0 : !si64
    } 
    else {
        atemhir.yield %1 : !si64
    } : !si64

    %7 = atemhir.constant #atemhir.int<114514> : !si64
    %8 = atemhir.cast int_promotion, %7 : !si64 to !si128
    %9 = atemhir.cast int_narrowing, %7 : !si64 to !si32

    %10 = atemhir.scope {
        atemhir.yield %0 : !si64
    } : !si64

    %14 = atemhir.constant #atemhir.int<0> : !si64

    %13 = atemhir.while (%14 : !si64)
    cond {
    ^bb0(%arg_0 : !si64):
        atemhir.condition(%5) %arg_0 : !si64
    }
    body {
    ^bb0(%arg_0 : !si64):
        atemhir.break %0 : !si64
    } 
    else {
    ^bb0(%arg_0 : !si64):
        atemhir.break %0 : !si64
    }: !si64

    %15 = atemhir.for (%14 : !si64)
    cond {
    ^bb0(%arg_0 : !si64):
        atemhir.condition(%5) %arg_0 : !si64
    }
    body {
    ^bb0(%arg_0 : !si64):
        atemhir.break %0 : !si64
    }
    step {
    ^bb0(%arg_0 : !si64):
        atemhir.break %0 : !si64
    }
    else {
    ^bb0(%arg_0 : !si64):
        atemhir.break %0 : !si64
    } : !si64

    atemhir.return %2 : !si64
}
