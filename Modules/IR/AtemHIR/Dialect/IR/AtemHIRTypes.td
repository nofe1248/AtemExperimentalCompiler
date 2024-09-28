#ifndef ATEM_HIR_TYPES
#define ATEM_HIR_TYPES

include "AtemHIRDialect.td"
include "IR/AtemHIR/Interfaces/AtemHIRFPTypeInterface.td"

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class AtemHIR_Type<string name, string type_mnemonic, 
    list<Trait> traits = [], string base_cpp_class = "::mlir::Type"> :
    TypeDef<AtemHIR_Dialect, name, traits, base_cpp_class> 
{
    let mnemonic = type_mnemonic;
}

//========================================================
// Atem HIR Integer Type
//========================================================

def AtemHIR_IntType : AtemHIR_Type<"Int", "int", 
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]>
{
    let summary = "Integer type with arbitrary precision";
    let description = [{
        Atem HIR type that represents Atem integer types.
    }];
    
    let parameters = (ins
        "unsigned":$width,
        "bool":$isSigned
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        auto isSigned() const -> bool { return this->getIsSigned(); }
        auto isUnsigned() const -> bool { return !this->getIsSigned(); }
    }];

    let genVerifyDecl = 1;
}

class UnsignedInt<int width> :
    Type<And<[
        CPred<"::mlir::isa<::mlir::atemhir::IntType>($_self)">,
        CPred<"::mlir::cast<::mlir::atemhir::IntType>($_self).isUnsigned()">,
        CPred<"::mlir::cast<::mlir::atemhir::IntType($_self).getWidth() == " # width>
    ]>, width # "-bit unsigned integer", "::mlir::atemhir::IntType">,
    BuildableType<
        "::mlir::atemhir::IntType::get($_builder.getContext(), "
        # width # ", false)"
    >
{
    int bitwidth = width;
}

class SignedInt<int width> :
    Type<And<[
        CPred<"::mlir::isa<::mlir::atemhir::IntType>($_self)">,
        CPred<"::mlir::cast<::mlir::atemhir::IntType>($_self).isSigned()">,
        CPred<"::mlir::cast<::mlir::atemhir::IntType($_self).getWidth() == " # width>
    ]>, width # "-bit signed integer", "::mlir::atemhir::IntType">,
    BuildableType<
        "::mlir::atemhir::IntType::get($_builder.getContext(), "
        # width # ", true)"
    >
{
    int bitwidth = width;
}

def SInt8 : SignedInt<8>;
def SInt16 : SignedInt<16>;
def SInt32 : SignedInt<32>;
def SInt64 : SignedInt<64>;
def SInt128 : SignedInt<128>;

def UInt8 : UnsignedInt<8>;
def UInt16 : UnsignedInt<16>;
def UInt32 : UnsignedInt<32>;
def UInt64 : UnsignedInt<64>;
def UInt128 : UnsignedInt<128>;

//========================================================
// Atem HIR Floating-point Type
//========================================================

class AtemHIR_FloatType<string name, string mnemonic>
    : AtemHIR_Type<name, mnemonic, [
        DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
        DeclareTypeInterfaceMethods<AtemHIRFPTypeInterface>
    ]>
{}

def AtemHIR_FP16 :
    AtemHIR_FloatType<"FP16", "fp16">
{
    let summary = "Atem HIR IEEE-754 binary16 floating-point type";
}

def AtemHIR_FP32 :
    AtemHIR_FloatType<"FP32", "fp32">
{
    let summary = "Atem HIR IEEE-754 binary32 floating-point type";
}

def AtemHIR_FP64 :
    AtemHIR_FloatType<"FP64", "fp64">
{
    let summary = "Atem HIR IEEE-754 binary64 floating-point type";
}

def AtemHIR_FP80 :
    AtemHIR_FloatType<"FP80", "fp80">
{
    let summary = "Atem HIR x87 80-bit floating-point type";
}

def AtemHIR_FP128 :
    AtemHIR_FloatType<"FP128", "fp128">
{
    let summary = "Atem HIR IEEE-754 binary128 floating-point type";
}
#endif  //ATEM_HIR_TYPES