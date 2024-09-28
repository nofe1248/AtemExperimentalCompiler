#ifndef ATEM_HIR_OPS
#define ATEM_HIR_OPS

include "AtemHIRDialect.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"

class AtemHIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AtemHIR_Dialect, mnemonic, traits>;

def AtemHIR_ConstantOp :
    AtemHIR_Op<"constant", [ConstantLike, Pure, AllTypesMatch<["value", "res"]>]>
{
    let summary = "constant operation";

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs AnyType:$res);

    let assemblyFormat = "$value attr-dict";

    let hasVerifier = 1;

    let hasFolder = 1;
}

def AtemHIR_ReturnOp :
    AtemHIR_Op<"return", [Terminator, ReturnLike]>
{
    let summary = "return operation";

    let arguments = (ins AnyType:$data);

    let assemblyFormat = "$data attr-dict `:` type($data)";
}

def AtemHIR_FunctionOp :
    AtemHIR_Op<"function", [
        IsolatedFromAbove,
        FunctionOpInterface,
        RegionKindInterface
    ]>
{
    let summary = "function operation";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    dag regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        auto getCallableRegion() -> ::mlir::Region * { return &this->getBody(); }

        auto getArgumentTypes() -> ::llvm::ArrayRef<::mlir::Type> { return this->getFunctionType().getInputs(); }
        auto getResultTypes() -> ::llvm::ArrayRef<::mlir::Type> { return this->getFunctionType().getResults(); }

        static auto getRegionKind(unsigned index) -> ::mlir::RegionKind { return mlir::RegionKind::SSACFG; }
    }];
}

def AtemHIR_CallOp :
    AtemHIR_Op<"call", [CallOpInterface]>
{
    let summary = "call operation";
    
    let arguments = (ins
        SymbolRefAttr:$callee,
        Variadic<AnyType>:$arg_operands
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = "$callee `(` $arg_operands `)` attr-dict `:` functional-type($arg_operands, results)";

    let extraClassDeclaration = [{
        auto getCallableForCallee() -> ::mlir::CallInterfaceCallable { return getCalleeAttr(); }
        auto setCalleeFromCallable(::mlir::CallInterfaceCallable callee) -> void { setCalleeAttr(callee.get<::mlir::SymbolRefAttr>()); }
    }];
}

#endif  //ATEM_HIR_OPS