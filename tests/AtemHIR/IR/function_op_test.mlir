!si64 = !atemhir.int<s, 64>
!ui64 = !atemhir.int<u, 64>
!f64 = !atemhir.fp64

atemhir.function @main() -> !si64 {
    %0 = atemhir.constant #atemhir.int<-42> : !si64
    %1 = atemhir.constant #atemhir.int<42> : !si64
    %2 = atemhir.constant #atemhir.int<42> : !si64
    %3 = atemhir.constant #atemhir.fp<-4.2e3> : !atemhir.fp128
    %4 = atemhir.zeroinit : !atemhir.bool
    %5 = atemhir.compare lt, %0, %1: !atemhir.int<s, 64>

    atemhir.if %5 {
        atemhir.yield
    } 
    else {
        atemhir.yield
    } : !atemhir.unit

    atemhir.while {
        atemhir.condition %5
    } 
    do {
        %6 = atemhir.allocate_var !atemhir.fp80, !atemhir.ptr<!atemhir.fp80>, ["some_local_var"]
        atemhir.yield
    } : !atemhir.unit
    atemhir.return %2 : !si64
}
