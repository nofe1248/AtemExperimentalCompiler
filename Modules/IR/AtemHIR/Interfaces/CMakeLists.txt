function(add_atemhir_mlir_type_interface interface)
    set(LLVM_TARGET_DEFINITIONS ${interface}.td)
    mlir_tablegen(${interface}.h.inc -gen-type-interface-decls)
    mlir_tablegen(${interface}.cpp.inc -gen-type-interface-defs)
    add_public_tablegen_target(MLIR${interface}IncGen)
    add_dependencies(mlir-generic-headers MLIR${interface}IncGen)
endfunction()
add_atemhir_mlir_type_interface(AtemHIRFPTypeInterface)

add_mlir_dialect_library(
    AtemHIRInterfaces
        AtemHIRFPTypeInterface.cpp
        AtemHIRFPTypeInterface.hpp

    ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

    DEPENDS
        MLIRAtemHIRFPTypeInterfaceIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        ${dialect_libs}
        MLIRIR
        MLIRParser
        MLIRSupport
)