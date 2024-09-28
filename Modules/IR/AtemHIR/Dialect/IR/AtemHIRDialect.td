#ifndef ATEM_HIR_DIALECT
#define ATEM_HIR_DIALECT

include "mlir/IR/OpBase.td"

def AtemHIR_Dialect : Dialect {
    let name = "atemhir";
    let summary = "A high-level dialect for Atem";
    let description = [{
        A high-level IR of Atem which represents almost all information in the original AST
    }];

    let cppNamespace = "::mlir::atemhir";

    let extraClassDeclaration = [{
        auto registerTypes() -> void;
        auto registerAttributes() -> void;
        auto registerOperations() -> void;

        auto parseType(::mlir::DialectAsmParser &parser) const -> ::mlir::Type override;
        auto printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const -> void override;

        auto parseAttribute(::mlir::DialectAsmParser &parser, ::mlir::Type type) const -> ::mlir::Attribute override;
        auto printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter &os) const -> void override;
    }];
}

#endif  //ATEM_HIR_DIALECT