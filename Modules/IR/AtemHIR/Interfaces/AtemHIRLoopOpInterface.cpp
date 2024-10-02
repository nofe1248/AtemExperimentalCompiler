#include "AtemHIRLoopOpInterface.hpp"

#include "Modules/IR/AtemHIR/Interfaces/AtemHIRLoopOpInterface.cpp.inc"

auto mlir::atemhir::detail::verifyLoopOpInterface(mlir::Operation *op) -> mlir::LogicalResult
{
    return success();
}

auto mlir::atemhir::AtemHIRLoopOpInterface::getLoopOpSuccessorRegions(AtemHIRLoopOpInterface op, RegionBranchPoint point,
                                                                      SmallVectorImpl<RegionSuccessor> &successors) -> void
{
    assert(point.isParent() or point.getRegionOrNull());

    if (point.isParent())
    {
        successors.emplace_back(&op.getEntry(), op.getEntry().getArguments());
    }
    else if (&op.getCond() == point.getRegionOrNull())
    {
        successors.emplace_back(RegionSuccessor{op->getResults()});
        successors.emplace_back(&op.getBody(), op.getBody().getArguments());
    }
    else if (&op.getBody() == point.getRegionOrNull())
    {
        auto *after_body = (op.maybeGetStep() ? op.maybeGetStep() : &op.getCond());
        successors.emplace_back(after_body, after_body->getArguments());
    }
    else if (op.maybeGetStep() == point.getRegionOrNull())
    {
        successors.emplace_back(&op.getCond(), op.getCond().getArguments());
    }
    else
    {
        llvm_unreachable("unexpected branch origin");
    }
}