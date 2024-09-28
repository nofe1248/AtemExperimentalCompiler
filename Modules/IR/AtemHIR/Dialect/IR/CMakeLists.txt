add_mlir_dialect(AtemHIR atemhir)
add_mlir_doc(AtemHIRDialect AtemHIRDialect Dialect/ -gen-dialect-doc)
add_mlir_doc(AtemHIROps AtemHIROps Dialect/ -gen-op-doc)
add_mlir_doc(AtemHIRAttrs AtemHIRAttrs Dialect/ -gen-attrdef-doc)

set(LLVM_TARGET_DEFINITIONS AtemHIRAttrs.td)
mlir_tablegen(AtemHIRAttrDefs.h.inc -gen-attrdef-decls
        -attrdefs-dialect=atemhir)
mlir_tablegen(AtemHIRAttrDefs.cpp.inc -gen-attrdef-defs
        -attrdefs-dialect=atemhir)
add_public_tablegen_target(MLIRAtemHIRAttrDefsIncGen)

add_mlir_dialect_library(
    AtemHIR
        AtemHIR.cpp
        AtemHIRAttrs.cpp
        AtemHIRTypes.cpp
        AtemHIROps.cpp
        AtemHIRAttrs.hpp
        AtemHIRDialect.hpp
        AtemHIROps.hpp
        AtemHIRTypes.hpp


    DEPENDS
        MLIRAtemHIRIncGen
        AtemHIRInterfaces

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSupport
)