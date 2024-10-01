#ifndef ATEM_HIR_ATTRS
#define ATEM_HIR_ATTRS

include "AtemHIRDialect.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"

class AtemHIR_Attr<string name, string attr_mnemonic, list<Trait> traits = []> :
    AttrDef<AtemHIR_Dialect, name, traits>
{
    let mnemonic = attr_mnemonic;
}

//========================================================
// Atem HIR Boolean Attribute
//========================================================

def AtemHIR_BoolAttr :
    AtemHIR_Attr<"Bool", "bool", [TypedAttrInterface]>
{
    let summary = "Represent true/false for !atemhir.bool type";

    let parameters = (ins
        AttributeSelfTypeParameter<"", "::mlir::atemhir::BoolType">:$type,
        "bool":$value
    );

    let assemblyFormat = [{
        `<` $value `>`
    }];
}

//========================================================
// Atem HIR Integer Attribute
//========================================================

def AtemHIR_IntAttr :
    AtemHIR_Attr<"Int", "int", [TypedAttrInterface]>
{
    let summary = "An attribute represents an integer value";

    let parameters = (ins 
        AttributeSelfTypeParameter<"">:$type,
        "::llvm::APInt":$value
    );

    let builders = [
        AttrBuilderWithInferredContext<(ins "::mlir::Type":$type, "::llvm::APInt const &":$value), [{
            return $_get(type.getContext(), type, value);
        }]>,
        AttrBuilderWithInferredContext<(ins "::mlir::Type":$type, "int64_t":$value), [{
            IntType int_type = ::mlir::cast<IntType>(type);
            ::llvm::APInt ap_value(int_type.getWidth(), value, int_type.isSigned());
            return $_get(int_type.getContext(), int_type, ap_value);
        }]>
    ];

    let extraClassDeclaration = [{
        auto getSignedInt() const -> int64_t { return this->getValue().getSExtValue(); }
        auto getUnsignedInt() const -> uint64_t { return this->getValue().getZExtValue(); }
        auto getSignedIntString() const -> ::std::string { 
            return ::llvm::toString(this->getValue(), 10, true); 
        }
        auto getUnsignedIntString() const -> ::std::string { 
            return ::llvm::toString(this->getValue(), 10, false); 
        }
        auto isNullValue() const -> bool { return this->getValue() == 0; }
        auto getBitWidth() const -> uint64_t { return ::mlir::cast<IntType>(this->getType()).getWidth(); }
    }];

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//========================================================
// Atem HIR Floating-point Type
//========================================================

def FPAttr :
    AtemHIR_Attr<"FP", "fp", [TypedAttrInterface]>
{
    let summary = "An attribute containing a floating-point value";

    let parameters = (ins 
        AttributeSelfTypeParameter<"">:$type,
        "APFloat":$value
    );

    let builders = [
        AttrBuilderWithInferredContext<(ins "::mlir::Type":$type, "::llvm::APFloat":$value), 
            [{
                return $_get(type.getContext(), type, value);
            }]
        >
    ];

    let extraClassDeclaration = [{
        static auto getZero(::mlir::Type type) -> FPAttr;
    }];

    let genVerifyDecl = 1;

    let assemblyFormat = "`<` custom<FloatLiteral>($value, ref($type)) `>`";
}

//========================================================
// Atem HIR Constant Array Attribute
//========================================================

def AtemHIR_ConstantArrayAttr :
    AtemHIR_Attr<"ConstantArray", "constant_array", [TypedAttrInterface]>
{
    let summary = "Attribute represents a constant array";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "::mlir::Attribute":$elements
    );

    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = 1;
}

#endif  //ATEM_HIR_ATTRS