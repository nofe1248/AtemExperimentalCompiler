#ifndef ATEM_UTIL_TYPE_INTERFACE_TD
#define ATEM_UTIL_TYPE_INTERFACE_TD

include "mlir/IR/OpBase.td"

def AtemHIRUtilTypeInterface :
    TypeInterface<"AtemHIRUtilTypeInterface">
{
    let description = [{
        Contains helper function to query debug info of Atem HIR Types
    }];

    let cppNamespace = "::mlir::atemhir";

    let methods = [
        InterfaceMethod<
            [{
                Returns the string representation of this Atem HIR type.
            }],
            "::std::string",
            "toAtemTypeString",
            (ins),
            ""
        >
    ];
}

#endif  //ATEM_FP_TYPE_INTERFACE_TD