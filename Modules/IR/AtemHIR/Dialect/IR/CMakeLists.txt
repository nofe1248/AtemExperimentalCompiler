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

set(LLVM_TARGET_DEFINITIONS AtemHIROps.td)
mlir_tablegen(AtemHIROpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(AtemHIROpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAtemHIREnumDefsIncGen)

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
        AtemHIROpsEnums.cpp
        AtemHIROpsEnums.hpp


    DEPENDS
        MLIRAtemHIRIncGen
        MLIRAtemHIRAttrDefsIncGen
        MLIRAtemHIREnumDefsIncGen
        AtemHIRInterfaces

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSupport
)