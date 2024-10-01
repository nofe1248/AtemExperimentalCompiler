#include "IR/AtemHIR/Dialect/IR/AtemHIROps.hpp"
#include "IR/AtemHIR/Dialect/IR/AtemHIRDialect.hpp"

#define GET_OP_CLASSES
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIR.cpp.inc"

//========================================================
// Dialect Operation Initialization
//========================================================

auto mlir::atemhir::AtemHIRDialect::registerOperations() -> void
{
    this->addOperations<
#define GET_OP_LIST
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIR.cpp.inc"

        >();
}

auto mlir::atemhir::buildTerminatedBody(OpBuilder &builder, Location loc) -> void
{

}

//========================================================
// ConstantOp Definitions
//========================================================

auto mlir::atemhir::ConstantOp::verify() -> LogicalResult
{
    return success();
}

auto mlir::atemhir::ConstantOp::fold(FoldAdaptor adaptor) -> OpFoldResult
{
    return this->getValue();
}

//========================================================
// FunctionOp Definitions
//========================================================

auto mlir::atemhir::FunctionOp::parse(OpAsmParser &parser, OperationState &result) -> ParseResult
{
    auto build_func_type = [](auto &builder, auto arg_types, auto results, auto, auto) { return builder.getFunctionType(arg_types, results); };

    return function_interface_impl::parseFunctionOp(parser, result, false, getFunctionTypeAttrName(result.name), build_func_type,
                                                    getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

auto mlir::atemhir::FunctionOp::print(OpAsmPrinter &printer) -> void
{
    function_interface_impl::printFunctionOp(printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName());
}

//========================================================
// AllocateVarOp Definitions
//========================================================

auto mlir::atemhir::AllocateVarOp::build(OpBuilder &ods_builder, OperationState &ods_state, Type addr, Type allocation_type, StringRef name,
                                         IntegerAttr alignment) -> void
{
    ods_state.addAttribute(getAllocationTypeAttrName(ods_state.name), TypeAttr::get(allocation_type));
    ods_state.addAttribute(getNameAttrName(ods_state.name), ods_builder.getStringAttr(name));

    if (alignment)
    {
        ods_state.addAttribute(getAlignmentAttrName(ods_state.name), alignment);
    }

    ods_state.addTypes(addr);
}

auto mlir::atemhir::AllocateVarOp::getPromotableSlots() -> SmallVector<MemorySlot>
{
    return {
        MemorySlot{this->getResult(), this->getAllocationType()}
    };
}

auto mlir::atemhir::AllocateVarOp::getDefaultValue(MemorySlot const &slot, OpBuilder &builder) -> Value
{
    return builder.create<atemhir::ZeroInitOp>(this->getLoc(), slot.elemType);
}

auto mlir::atemhir::AllocateVarOp::handleBlockArgument(MemorySlot const &slot, BlockArgument argument, OpBuilder &builder) -> void
{
}

auto mlir::atemhir::AllocateVarOp::handlePromotionComplete(MemorySlot const &slot, Value default_value, OpBuilder &builder)
    -> std::optional<PromotableAllocationOpInterface>
{
    if (default_value && default_value.use_empty())
    {
        default_value.getDefiningOp()->erase();
    }
    this->erase();
    return std::nullopt;
}

//========================================================
// LoadOp Definitions
//========================================================

auto mlir::atemhir::LoadOp::loadsFrom(MemorySlot const &slot) -> bool
{
}

auto mlir::atemhir::LoadOp::storesTo(MemorySlot const &slot) -> bool
{
}

auto mlir::atemhir::LoadOp::getStored(MemorySlot const &slot, OpBuilder &builder, Value reaching_def, DataLayout const &data_layout) -> Value
{
}

auto mlir::atemhir::LoadOp::canUsesBeRemoved(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses,
                                             SmallVectorImpl<OpOperand *> &new_blocking_uses, DataLayout const &data_layout) -> bool
{
}

auto mlir::atemhir::LoadOp::removeBlockingUses(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses, OpBuilder &builder,
                                               Value reaching_def, DataLayout const &data_layout) -> DeletionKind
{
}

//========================================================
// StoreOp Definitions
//========================================================

auto mlir::atemhir::StoreOp::loadsFrom(MemorySlot const &slot) -> bool
{
}

auto mlir::atemhir::StoreOp::storesTo(MemorySlot const &slot) -> bool
{
}

auto mlir::atemhir::StoreOp::getStored(MemorySlot const &slot, OpBuilder &builder, Value reaching_def, DataLayout const &data_layout) -> Value
{
}

auto mlir::atemhir::StoreOp::canUsesBeRemoved(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses,
                                              SmallVectorImpl<OpOperand *> &new_blocking_uses, DataLayout const &data_layout) -> bool
{
}

auto mlir::atemhir::StoreOp::removeBlockingUses(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses, OpBuilder &builder,
                                                Value reaching_def, DataLayout const &data_layout) -> DeletionKind
{
}

//========================================================
// CopyOp Definitions
//========================================================

auto mlir::atemhir::CopyOp::verify() -> LogicalResult
{

}

auto mlir::atemhir::CopyOp::loadsFrom(MemorySlot const &slot) -> bool
{
}

auto mlir::atemhir::CopyOp::storesTo(MemorySlot const &slot) -> bool
{
}

auto mlir::atemhir::CopyOp::getStored(MemorySlot const &slot, OpBuilder &builder, Value reaching_def, DataLayout const &data_layout) -> Value
{
}

auto mlir::atemhir::CopyOp::canUsesBeRemoved(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses,
                                             SmallVectorImpl<OpOperand *> &new_blocking_uses, DataLayout const &data_layout) -> bool
{
}

auto mlir::atemhir::CopyOp::removeBlockingUses(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses, OpBuilder &builder,
                                               Value reaching_def, DataLayout const &data_layout) -> DeletionKind
{
}

//========================================================
// CastOp Definitions
//========================================================

auto mlir::atemhir::CastOp::verify() -> LogicalResult
{

}

auto mlir::atemhir::CastOp::fold(FoldAdaptor adaptor) -> OpFoldResult
{

}

auto mlir::atemhir::CastOp::canUsesBeRemoved(SmallPtrSetImpl<OpOperand *> const &blocking_uses, SmallVectorImpl<OpOperand *> &new_blocking_uses,
                                             DataLayout const &data_layout) -> bool
{
}

auto mlir::atemhir::CastOp::removeBlockingUses(SmallPtrSetImpl<OpOperand *> const &blocking_uses, OpBuilder &builder) -> DeletionKind
{
}

//========================================================
// IfOp Definitions
//========================================================

auto mlir::atemhir::IfOp::verify() -> LogicalResult
{

}