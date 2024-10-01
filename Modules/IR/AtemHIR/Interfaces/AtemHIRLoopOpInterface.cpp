#include "AtemHIRLoopOpInterface.hpp"

#include "Modules/IR/AtemHIR/Interfaces/AtemHIRLoopOpInterface.cpp.inc"

auto mlir::atemhir::detail::verifyLoopOpInterface(mlir::Operation *op) -> mlir::LogicalResult
{
    return success();
}

