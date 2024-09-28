#ifndef ATEM_FP_TYPE_INTERFACE_TD
#define ATEM_FP_TYPE_INTERFACE_TD

include "mlir/IR/OpBase.td"

def AtemHIRFPTypeInterface : 
    TypeInterface<"AtemHIRFPTypeInterface">
{
    let description = [{
        Contains helper function to query properties about a floating-point type.
    }];

    let cppNamespace = "::mlir::atemhir";

    let methods = [
        InterfaceMethod<
            [{
                Returns the bit width of this floating-point type.
            }],
            "unsigned",
            "getWidth",
            (ins),
            "",
            [{
                return ::llvm::APFloat::semanticsSizeInBits($_type.getFloatSemantics());
            }]
        >,
        InterfaceMethod<
            [{
                Returns the mantissa width of this floating-point type.
            }],
            "unsigned",
            "getFPMantissaWidth",
            (ins),
            "",
            [{
                return ::llvm::APFloat::semanticsPrecision($_type.getFloatSemantics());
            }]
        >,
        InterfaceMethod<
            [{
                Returns the float semantics of this floating-point type.
            }],
            "::llvm::fltSemantics const &",
            "getFloatSemantics"
        >
    ];
}

#endif  //ATEM_FP_TYPE_INTERFACE_TD