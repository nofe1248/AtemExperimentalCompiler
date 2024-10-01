!si64 = !atemhir.int<s, 64>
!ui64 = !atemhir.int<u, 64>
!f64 = !atemhir.fp64

atemhir.function @main() -> !si64 {
    %0 = atemhir.constant #atemhir.int<-42> : !si64
    %1 = atemhir.constant #atemhir.int<42> : !ui64
    %2 = atemhir.constant #atemhir.int<42> : !si64
    %3 = atemhir.constant #atemhir.fp<-4.2e3> : !atemhir.fp128
    %4 = atemhir.zeroinit : !atemhir.bool
    atemhir.return %2 : !si64
}
