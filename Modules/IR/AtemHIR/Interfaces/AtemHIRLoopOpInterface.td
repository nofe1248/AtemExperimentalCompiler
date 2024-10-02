#ifndef ATEM_HIR_LOOP_OP_INTERFACE
#define ATEM_HIR_LOOP_OP_INTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

def AtemHIRLoopOpInterface :
    OpInterface<
        "AtemHIRLoopOpInterface", 
        [
            DeclareOpInterfaceMethods<RegionBranchOpInterface>,
            DeclareOpInterfaceMethods<LoopLikeOpInterface>
        ]
    >
{
    let description = [{
        Contains helper function to query properties and perform transformations on a loop
    }];

    let cppNamespace = "::mlir::atemhir";

    let methods = [
        InterfaceMethod<
            [{
                Return the loop's conditional region.
            }],
            "::mlir::Region &",
            "getCond"
        >,
        InterfaceMethod<
            [{
                Return the loop's body region.
            }],
            "::mlir::Region &",
            "getBody"
        >,
        InterfaceMethod<
            [{
                Return the loop's step region or nullptr.
            }],
            "::mlir::Region *",
            "maybeGetStep",
            (ins),
            "",
            "return nullptr;"
        >,
        InterfaceMethod<
            [{
                Returns the first region to be executed.
            }],
            "::mlir::Region &",
            "getEntry",
            (ins),
            "",
            [{
                return $_op.getCond();
            }]
        >,
        InterfaceMethod<
            [{
                Returns a list of regions in order of execution
            }],
            "::llvm::SmallVector<::mlir::Region *>",
            "getRegionsInExecutionOrder",
            (ins),
            "",
            [{
                return ::llvm::SmallVector<::mlir::Region *, 2>{&($_op.getRegion(0)), &($_op.getRegion(1))};
            }]
        >,
        InterfaceMethod<
            [{
                Recursively walks the body of the loop in pre-order while skipping nested loops
                and executing a callback on every other operation
            }],
            "::mlir::WalkResult",
            "walkBodySkippingNestedLoops",
            (ins "::llvm::function_ref<::mlir::WalkResult (::mlir::Operation *)>":$callback),
            "",
            [{
                return $_op.getBody().template walk<::mlir::WalkOrder::PreOrder>([&](::mlir::Operation *op) {
                    if (::mlir::isa<::mlir::atemhir::AtemHIRLoopOpInterface>(op)) {
                        return ::mlir::WalkResult::skip();
                    }
                    return callback(op);
                });
            }]
        >
    ];

    let extraClassDeclaration = [{
        static auto getLoopOrSuccessorRegions(
            ::mlir::atemhir::AtemHIRLoopOpInterface op,
            ::mlir::RegionBranchPoint point,
            ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> &regions
        ) -> void;
    }];

    let verify = [{
        return detail::verifyLoopOpInterface($_op);
    }];
}
#endif  //ATEM_HIR_LOOP_OP_INTERFACE