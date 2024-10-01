#ifndef ATEM_HIR_LOOP_OP_INTERFACE_HPP
#define ATEM_HIR_LOOP_OP_INTERFACE_HPP

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir
{
namespace atemhir
{
namespace detail
{
auto verifyLoopOpInterface(::mlir::Operation *op) -> ::mlir::LogicalResult;
}
}
}

#include "Modules/IR/AtemHIR/Interfaces/AtemHIRLoopOpInterface.h.inc"

#endif //ATEM_HIR_LOOP_OP_INTERFACE_HPP
