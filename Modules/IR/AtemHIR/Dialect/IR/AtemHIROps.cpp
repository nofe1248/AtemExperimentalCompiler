#include "IR/AtemHIR/Dialect/IR/AtemHIROps.hpp"
#include "IR/AtemHIR/Dialect/IR/AtemHIRDialect.hpp"

auto ensureRegionTerminator(mlir::OpAsmParser &parser, mlir::Region &region, llvm::SMLoc error_loc) -> llvm::LogicalResult;
auto canOmitRegionTerminator(mlir::Region &region) -> bool;
static auto parseOmittedTerminatorRegion(mlir::OpAsmParser &parser, mlir::Region &region) -> mlir::ParseResult;
static auto printOmittedTerminatorRegion(mlir::OpAsmPrinter &printer, mlir::atemhir::ScopeOp &op, mlir::Region &region) -> void;

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

auto ensureRegionTerminator(mlir::OpAsmParser &parser, mlir::Region &region, llvm::SMLoc error_loc) -> llvm::LogicalResult
{
    using namespace mlir;
    Location error_location = parser.getEncodedSourceLoc(parser.getCurrentLocation());
    OpBuilder builder(parser.getBuilder().getContext());

    if (region.empty() or (region.back().mightHaveTerminator() and region.back().getTerminator()))
    {
        return success();
    }

    if (not region.hasOneBlock())
    {
        return parser.emitError(error_loc, "multi-block region must not omit terminator");
    }

    if (region.back().empty())
    {
        return parser.emitError(error_loc, "empty region must not omit terminator");
    }

    region.back().push_back(builder.create<atemhir::YieldOp>(error_location));
    return success();
}

auto canOmitRegionTerminator(mlir::Region &region) -> bool
{
    auto const single_non_empty_block = region.hasOneBlock() and not region.back().empty();
    auto const yields_nothing = [&region]() {
        mlir::atemhir::YieldOp yield_op = mlir::dyn_cast<mlir::atemhir::YieldOp>(region.back().getTerminator());
        return yield_op and yield_op.getArgs().empty();
    }();
    return single_non_empty_block and yields_nothing;
}

static auto parseOmittedTerminatorRegion(mlir::OpAsmParser &parser, mlir::Region &region) -> mlir::ParseResult
{
    auto region_loc = parser.getCurrentLocation();
    if (parser.parseRegion(region))
    {
        return mlir::failure();
    }
    if (ensureRegionTerminator(parser, region, region_loc).failed())
    {
        return mlir::failure();
    }
    return mlir::success();
}

static auto printOmittedTerminatorRegion(mlir::OpAsmPrinter &printer, mlir::atemhir::ScopeOp &op, mlir::Region &region) -> void
{
    printer.printRegion(region, false, not canOmitRegionTerminator(region));
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
    return this->getAddr() == slot.ptr;
}

auto mlir::atemhir::LoadOp::storesTo(MemorySlot const &slot) -> bool
{
    return false;
}

auto mlir::atemhir::LoadOp::getStored(MemorySlot const &slot, OpBuilder &builder, Value reaching_def, DataLayout const &data_layout) -> Value
{
    llvm_unreachable("getStored() shouldn't be called on atemhir::LoadOp");
}

auto mlir::atemhir::LoadOp::canUsesBeRemoved(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses,
                                             SmallVectorImpl<OpOperand *> &new_blocking_uses, DataLayout const &data_layout) -> bool
{
    if (blocking_uses.size() != 1)
    {
        return false;
    }
    Value const blocking_use = (*blocking_uses.begin())->get();
    return blocking_use == slot.ptr and this->getAddr() == slot.ptr and this->getResult().getType() == slot.elemType;
}

auto mlir::atemhir::LoadOp::removeBlockingUses(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses, OpBuilder &builder,
                                               Value reaching_def, DataLayout const &data_layout) -> DeletionKind
{
    this->getResult().replaceAllUsesWith(reaching_def);
    return DeletionKind::Delete;
}

//========================================================
// StoreOp Definitions
//========================================================

auto mlir::atemhir::StoreOp::loadsFrom(MemorySlot const &slot) -> bool
{
    return false;
}

auto mlir::atemhir::StoreOp::storesTo(MemorySlot const &slot) -> bool
{
    return this->getAddr() == slot.ptr;
}

auto mlir::atemhir::StoreOp::getStored(MemorySlot const &slot, OpBuilder &builder, Value reaching_def, DataLayout const &data_layout) -> Value
{
    return this->getValue();
}

auto mlir::atemhir::StoreOp::canUsesBeRemoved(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses,
                                              SmallVectorImpl<OpOperand *> &new_blocking_uses, DataLayout const &data_layout) -> bool
{
    if (blocking_uses.size() != 1)
    {
        return false;
    }
    Value const blocking_use = (*blocking_uses.begin())->get();
    return blocking_use == slot.ptr and this->getAddr() == slot.ptr and this->getValue().getType() == slot.elemType;
}

auto mlir::atemhir::StoreOp::removeBlockingUses(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses, OpBuilder &builder,
                                                Value reaching_def, DataLayout const &data_layout) -> DeletionKind
{
    return DeletionKind::Delete;
}

//========================================================
// CopyOp Definitions
//========================================================

auto mlir::atemhir::CopyOp::verify() -> LogicalResult
{
    if (not this->getType().getPointeeType().hasTrait<DataLayoutTypeInterface::Trait>())
    {
        return emitError() << "missing data layout for pointee type";
    }

    if (this->getSource() == this->getDestination())
    {
        return emitError() << "source and destination addresses are the same";
    }
}

auto mlir::atemhir::CopyOp::loadsFrom(MemorySlot const &slot) -> bool
{
    return this->getSource() == slot.ptr;
}

auto mlir::atemhir::CopyOp::storesTo(MemorySlot const &slot) -> bool
{
    return this->getDestination() == slot.ptr;
}

auto mlir::atemhir::CopyOp::getStored(MemorySlot const &slot, OpBuilder &builder, Value reaching_def, DataLayout const &data_layout) -> Value
{
    return builder.create<LoadOp>(this->getLoc(), slot.elemType, this->getSource());
}

auto mlir::atemhir::CopyOp::canUsesBeRemoved(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses,
                                             SmallVectorImpl<OpOperand *> &new_blocking_uses, DataLayout const &data_layout) -> bool
{
    if (this->getSource() == this->getDestination())
    {
        return false;
    }
    return this->getLength() == data_layout.getTypeSize(slot.elemType);
}

auto mlir::atemhir::CopyOp::removeBlockingUses(MemorySlot const &slot, SmallPtrSetImpl<OpOperand *> const &blocking_uses, OpBuilder &builder,
                                               Value reaching_def, DataLayout const &data_layout) -> DeletionKind
{
    if (this->loadsFrom(slot))
    {
        builder.create<StoreOp>(this->getLoc(), reaching_def, this->getDestination(), false, IntegerAttr{}, MemoryOrderAttr{});
    }
    return DeletionKind::Delete;
}

//========================================================
// CastOp Definitions
//========================================================

auto mlir::atemhir::CastOp::verify() -> LogicalResult
{
    auto result_type = this->getResult().getType();
    auto source_type = this->getSource().getType();

    switch (this->getKind())
    {
    case CastKind::bitcast: {
        return success();
    }
    case CastKind::bool_to_int: {
        if (not isa<IntType>(result_type))
        {
            return emitError() << "bool_to_int cast requires the result type be a !atemhir.int type";
        }
        if (not isa<BoolType>(source_type))
        {
            return emitError() << "bool_to_int cast requires the source type to be a !atemhir.bool type";
        }
        return success();
    }
    case CastKind::float_promotion: {
        if (not isa<AtemHIRFPTypeInterface>(source_type) or not isa<AtemHIRFPTypeInterface>(result_type))
        {
            return emitError() << "int_promotion cast requires both the source and result type be !atemhir.int type";
        }
        auto source_type_fp = cast<AtemHIRFPTypeInterface>(source_type);
        auto result_type_fp = cast<AtemHIRFPTypeInterface>(result_type);
        if (source_type_fp.getWidth() > result_type_fp.getWidth())
        {
            return emitError() << "float_promotion requires that the width of source fp is smaller than result fp, use float_narrowing instead";
        }
        return success();
    }
    case CastKind::float_to_int: {
        if (not isa<IntType>(result_type))
        {
            return emitError() << "float_to_int cast requires the result type be a !atemhir.int type";
        }
        if (not isa<AtemHIRFPTypeInterface>(source_type))
        {
            return emitError() << "float_to_int cast requires the source type be a Atem HIR FP type";
        }
        return success();
    }
    case CastKind::int_promotion: {
        if (not isa<IntType>(source_type) or not isa<IntType>(result_type))
        {
            return emitError() << "int_promotion cast requires both the source and result type be !atemhir.int type";
        }
        if (cast<IntType>(source_type).getWidth() > cast<IntType>(result_type).getWidth())
        {
            return emitError()
                   << "int_promotion requires that the width of source integer type is smaller than result integer type, use int_narrowing instead";
        }
        return success();
    }
    case CastKind::int_to_bool: {
        if (not isa<BoolType>(result_type))
        {
            return emitError() << "int_to_bool cast requires the result type be a !atemhir.bool type";
        }
        if (not isa<IntType>(source_type))
        {
            return emitError() << "int_to_bool cast requires the source type to be a !atemhir.int type";
        }
        return success();
    }
    case CastKind::int_to_float: {
        if (not isa<AtemHIRFPTypeInterface>(result_type))
        {
            return emitError() << "int_to_float cast requires the result type be a Atem HIR FP type";
        }
        if (not isa<IntType>(source_type))
        {
            return emitError() << "int_to_float cast requires the source type be a !atem.int type";
        }
        return success();
    }
    case CastKind::int_narrowing: {
        if (not isa<IntType>(result_type) or not isa<IntType>(source_type))
        {
            return emitError() << "int_narrowing cast requires both the source type and result type be !atem.int type";
        }
        if (cast<IntType>(source_type).getWidth() <= cast<IntType>(result_type).getWidth())
        {
            return emitError() << "int_narrowing cast requires the width of source type is larger than result type, use int_promotion instead";
        }
        return success();
    }
    case CastKind::float_narrowing: {
        if (not isa<AtemHIRFPTypeInterface>(source_type) or not isa<AtemHIRFPTypeInterface>(result_type))
        {
            return emitError() << "float_narrowing cast requires both the source type and result type be Atem HIR FP type";
        }
        if (cast<AtemHIRFPTypeInterface>(source_type).getWidth() <= cast<AtemHIRFPTypeInterface>(result_type).getWidth())
        {
            return emitError() << "float_narrowing cast requires the width of source type is larger than result type, use float_promotion instead";
        }
        return success();
    }
    default:
        llvm_unreachable("unexpected cast kind");
    }
}

auto mlir::atemhir::CastOp::fold(FoldAdaptor adaptor) -> OpFoldResult
{
    auto is_int_or_bool_cast = [](CastOp op) {
        auto kind = op.getKind();
        return kind == CastKind::int_to_bool or kind == CastKind::bool_to_int or kind == CastKind::int_promotion or kind == CastKind::int_narrowing;
    };
    auto try_fold_cast_chain = [&is_int_or_bool_cast](CastOp op) -> Value {
        CastOp head = op, tail = op;

        while (op)
        {
            if (not is_int_or_bool_cast)
            {
                break;
            }
            head = op;
            op = dyn_cast_or_null<CastOp>(head.getSource().getDefiningOp());
        }

        if (head == tail)
        {
            return {};
        }

        if (head.getKind() == CastKind::bool_to_int and tail.getKind() == CastKind::int_to_bool)
        {
            return head.getSource();
        }

        if (head.getKind() == CastKind::int_to_bool and tail.getKind() == CastKind::int_to_bool)
        {
            return head.getResult();
        }

        return {};
    };

    if (this->getSource().getType() == this->getResult().getType())
    {
        switch (this->getKind())
        {
        case CastKind::int_promotion:
        case CastKind::int_narrowing: {
            SmallVector<OpFoldResult, 1> fold_results;
            auto fold_order = this->getSource().getDefiningOp()->fold(fold_results);
            if (fold_order.succeeded() and fold_results[0].is<Attribute>())
            {
                return fold_results[0].get<Attribute>();
            }
            return {};
        }
        case CastKind::bitcast: {
            return this->getSource();
        }
        default: {
            return {};
        }
        }
    }
    return try_fold_cast_chain(*this);
}

auto mlir::atemhir::CastOp::canUsesBeRemoved(SmallPtrSetImpl<OpOperand *> const &blocking_uses, SmallVectorImpl<OpOperand *> &new_blocking_uses,
                                             DataLayout const &data_layout) -> bool
{
    if (this->getKind() == CastKind::bitcast)
    {
        for (Value result : (*this)->getResults())
        {
            for (OpOperand &use : result.getUses())
            {
                new_blocking_uses.push_back(&use);
            }
        }
        return true;
    }
    return false;
}

auto mlir::atemhir::CastOp::removeBlockingUses(SmallPtrSetImpl<OpOperand *> const &blocking_uses, OpBuilder &builder) -> DeletionKind
{
    return DeletionKind::Delete;
}

//========================================================
// IfOp Definitions
//========================================================

auto mlir::atemhir::IfOp::verify() -> LogicalResult
{
    // need to verify types here
    return success();
}

auto mlir::atemhir::IfOp::getSuccessorRegions(RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &successors) -> void
{
    if (not point.isParent())
    {
        successors.push_back(RegionSuccessor(this->getODSResults(0)));
        return;
    }

    successors.push_back(RegionSuccessor(&this->getThenRegion()));
    if (Region *else_region = &this->getElseRegion())
    {
        successors.push_back(RegionSuccessor(else_region));
    }
}

auto mlir::atemhir::IfOp::parse(OpAsmParser &parser, OperationState &result) -> ParseResult
{
    result.regions.reserve(2);
    Region *then_region = result.addRegion();
    Region *else_region = result.addRegion();

    auto &builder = parser.getBuilder();

    OpAsmParser::UnresolvedOperand condition;

    Type bool_type = BoolType::get(builder.getContext());

    if (parser.parseOperand(condition) or parser.resolveOperand(condition, bool_type, result.operands))
    {
        return failure();
    }

    auto parse_then_loc = parser.getCurrentLocation();
    if (parser.parseRegion(*then_region))
    {
        return failure();
    }
    if (ensureRegionTerminator(parser, *then_region, parse_then_loc).failed())
    {
        return failure();
    }

    if (not parser.parseOptionalKeyword("else"))
    {
        auto parse_else_loc = parser.getCurrentLocation();
        if (parser.parseRegion(*else_region))
        {
            return failure();
        }
        if (ensureRegionTerminator(parser, *else_region, parse_else_loc).failed())
        {
            return failure();
        }
    }

    if (not parser.parseOptionalColon())
    {
        Type result_type;
        if (parser.parseType(result_type))
        {
            return failure();
        }
        result.addTypes(result_type);
    }

    if (parser.parseOptionalAttrDict(result.attributes))
    {
        return failure();
    }
    return success();
}

auto mlir::atemhir::IfOp::print(OpAsmPrinter &printer) -> void
{
    printer << " " << this->getCondition() << " ";
    auto &then_region = this->getThenRegion();
    printer.printRegion(then_region, false, not canOmitRegionTerminator(then_region));

    auto &else_region = this->getElseRegion();
    if (not else_region.empty())
    {
        printer << " else ";
        printer.printRegion(else_region, false, not canOmitRegionTerminator(else_region));
    }

    if (this->getResult())
    {
        printer << " : " << this->getResult().getType();
    }

    printer.printOptionalAttrDict(this->getOperation()->getAttrs());
}

auto mlir::atemhir::IfOp::build(OpBuilder &builder, OperationState &result, Value condition, bool has_else,
                                function_ref<void(OpBuilder &, Location)> then_builder, function_ref<void(OpBuilder &, Location)> else_builder)
    -> void
{
    assert(then_builder && "the builder for 'then' branch must be present");

    result.addOperands(condition);

    OpBuilder::InsertionGuard guard(builder);
    Region *then_region = result.addRegion();
    builder.createBlock(then_region);
    then_builder(builder, result.location);

    Region *else_region = result.addRegion();
    if (not has_else)
    {
        return;
    }

    builder.createBlock(else_region);
    else_builder(builder, result.location);
}

auto mlir::atemhir::IfOp::build(OpBuilder &builder, OperationState &result, Value condition, bool has_else,
                                function_ref<void(OpBuilder &, Type &, Location)> then_builder,
                                function_ref<void(OpBuilder &, Type &, Location)> else_builder) -> void
{
    assert(then_builder && "the builder for 'then' branch must be present");

    result.addOperands(condition);

    OpBuilder::InsertionGuard guard(builder);
    Region *then_region = result.addRegion();
    builder.createBlock(then_region);
    Type then_type;
    then_builder(builder, then_type, result.location);

    Region *else_region = result.addRegion();
    if (not has_else)
    {
        return;
    }

    builder.createBlock(else_region);
    Type else_type;
    else_builder(builder, else_type, result.location);

    if (then_type and else_type)
    {
        if (then_type != else_type)
        {
            llvm_unreachable("the result type of then branch and else branch must be the same");
        }
        result.addTypes(TypeRange{then_type});
    }
}

//========================================================
// YieldOp Definitions
//========================================================

auto mlir::atemhir::YieldOp::build(OpBuilder &builder, OperationState &result) -> void
{
}

//========================================================
// BreakOp Definitions
//========================================================

auto mlir::atemhir::BreakOp::verify() -> LogicalResult
{
    if (not this->getOperation()->getParentOfType<AtemHIRLoopOpInterface>())
    {
        return emitOpError("'break' must be within a loop");
    }
    return success();
}

//========================================================
// ContinueOp Definitions
//========================================================

auto mlir::atemhir::ContinueOp::verify() -> LogicalResult
{
    if (not this->getOperation()->getParentOfType<AtemHIRLoopOpInterface>())
    {
        return emitOpError("'continue' must be within a loop");
    }
    return success();
}

//========================================================
// ConditionOp Definitions
//========================================================

auto mlir::atemhir::ConditionOp::verify() -> LogicalResult
{
    if (not mlir::isa<AtemHIRLoopOpInterface>(this->getOperation()->getParentOp()))
    {
        return emitOpError("'condition' must be within a loop condition region");
    }
    return success();
}

auto mlir::atemhir::ConditionOp::getSuccessorRegions(ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &successors) -> void
{
    if (auto loop_op = mlir::dyn_cast<AtemHIRLoopOpInterface>(this->getOperation()->getParentOp()))
    {
        successors.emplace_back(&loop_op.getBody(), loop_op.getBody().getArguments());
        successors.emplace_back(loop_op->getResults());
    }
}

auto mlir::atemhir::ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) -> MutableOperandRange
{
    return this->getArgsMutable();
}

//========================================================
// WhileOp Definitions
//========================================================

auto mlir::atemhir::WhileOp::getSuccessorRegions(RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &successors) -> void
{
    assert(point.isParent() or point.getRegionOrNull());

    if (point.isParent())
    {
        successors.emplace_back(&this->getCond(), this->getCond().getArguments());
    }
    else if (&this->getBody() == point.getRegionOrNull())
    {
        successors.emplace_back(RegionSuccessor{this->getResults()});
        successors.emplace_back(&this->getCond(), this->getCond().getArguments());
    }
    else if (&this->getCond() == point.getRegionOrNull())
    {
        successors.emplace_back(&this->getBody(), this->getBody().getArguments());
        successors.emplace_back(&this->getElse(), this->getElse().getArguments());
    }
    else if (&this->getElse() == point.getRegionOrNull())
    {
        successors.emplace_back(RegionSuccessor{this->getResults()});
    }
    else
    {
        llvm_unreachable("unexpected branch origin");
    }
}

auto mlir::atemhir::WhileOp::getLoopRegions() -> SmallVector<Region *>
{
    return {&this->getCond(), &this->getBody()};
}

auto mlir::atemhir::WhileOp::getRegionIterArgs() -> Block::BlockArgListType
{
    return this->getBody().getArguments();
}

auto mlir::atemhir::WhileOp::getEntrySuccessorOperands(RegionBranchPoint point) -> OperandRange
{
    return this->getInits();
}

//========================================================
// DoWhileOp Definitions
//========================================================

auto mlir::atemhir::DoWhileOp::getSuccessorRegions(RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &successors) -> void
{
    assert(point.isParent() or point.getRegionOrNull());

    if (point.isParent())
    {
        successors.emplace_back(&this->getBody(), this->getBody().getArguments());
    }
    else if (&this->getBody() == point.getRegionOrNull())
    {
        successors.emplace_back(RegionSuccessor{this->getResults()});
        successors.emplace_back(&this->getCond(), this->getCond().getArguments());
    }
    else if (&this->getCond() == point.getRegionOrNull())
    {
        successors.emplace_back(&this->getBody(), this->getBody().getArguments());
        successors.emplace_back(&this->getElse(), this->getElse().getArguments());
    }
    else if (&this->getElse() == point.getRegionOrNull())
    {
        successors.emplace_back(RegionSuccessor{this->getResults()});
    }
    else
    {
        llvm_unreachable("unexpected branch origin");
    }
}

auto mlir::atemhir::DoWhileOp::getLoopRegions() -> SmallVector<Region *>
{
    return {&this->getBody()};
}

auto mlir::atemhir::DoWhileOp::getEntrySuccessorOperands(RegionBranchPoint point) -> OperandRange
{
    return this->getInits();
}

auto mlir::atemhir::DoWhileOp::getRegionIterArgs() -> Block::BlockArgListType
{
    return this->getBody().getArguments();
}

//========================================================
// ForOp Definitions
//========================================================

auto mlir::atemhir::ForOp::getSuccessorRegions(RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &successors) -> void
{
    assert(point.isParent() or point.getRegionOrNull());

    if (point.isParent())
    {
        successors.emplace_back(&this->getEntry(), this->getEntry().getArguments());
    }
    else if (&this->getCond() == point.getRegionOrNull())
    {
        successors.emplace_back(&this->getElse(), this->getElse().getArguments());
        successors.emplace_back(&this->getBody(), this->getBody().getArguments());
    }
    else if (&this->getBody() == point.getRegionOrNull())
    {
        auto *after_body = (this->maybeGetStep() ? this->maybeGetStep() : &this->getCond());
        successors.emplace_back(after_body, after_body->getArguments());
        successors.emplace_back(RegionSuccessor{this->getResults()});
    }
    else if (this->maybeGetStep() == point.getRegionOrNull())
    {
        successors.emplace_back(&this->getCond(), this->getCond().getArguments());
    }
    else if (&this->getElse() == point.getRegionOrNull())
    {
        successors.emplace_back(RegionSuccessor{this->getResults()});
    }
    else
    {
        llvm_unreachable("unexpected branch origin");
    }
}

auto mlir::atemhir::ForOp::getLoopRegions() -> SmallVector<Region *>
{
    return {&this->getBody()};
}

auto mlir::atemhir::ForOp::build(OpBuilder &builder, OperationState &result, llvm::function_ref<void(OpBuilder &, Location)> cond_builder,
                                 llvm::function_ref<void(OpBuilder &, Location)> body_builder,
                                 llvm::function_ref<void(OpBuilder &, Location)> step_builder,
                                 llvm::function_ref<void(OpBuilder &, Type &, Location)> else_builder) -> void
{
    Type for_type;

    OpBuilder::InsertionGuard guard(builder);

    builder.createBlock(result.addRegion());
    cond_builder(builder, result.location);

    builder.createBlock(result.addRegion());
    body_builder(builder, result.location);

    builder.createBlock(result.addRegion());
    step_builder(builder, result.location);

    if (for_type)
    {
        result.addTypes(TypeRange{for_type});
    }
}

auto mlir::atemhir::ForOp::build(OpBuilder &builder, OperationState &result, llvm::function_ref<void(OpBuilder &, Location)> cond_builder,
                                 llvm::function_ref<void(OpBuilder &, Location)> body_builder,
                                 llvm::function_ref<void(OpBuilder &, Location)> step_builder,
                                 llvm::function_ref<void(OpBuilder &, Location)> else_builder) -> void
{
    OpBuilder::InsertionGuard guard(builder);

    builder.createBlock(result.addRegion());
    cond_builder(builder, result.location);

    builder.createBlock(result.addRegion());
    body_builder(builder, result.location);

    builder.createBlock(result.addRegion());
    step_builder(builder, result.location);
}

auto mlir::atemhir::ForOp::getEntrySuccessorOperands(RegionBranchPoint point) -> OperandRange
{
    return this->getInits();
}

auto mlir::atemhir::ForOp::getRegionIterArgs() -> Block::BlockArgListType
{
    return this->getBody().getArguments();
}

//========================================================
// UnaryOp Definitions
//========================================================

auto mlir::atemhir::UnaryOp::verify() -> LogicalResult
{
    return success();
}

auto mlir::atemhir::UnaryOp::fold(FoldAdaptor adaptor) -> OpFoldResult
{
    if (isa<BoolType>(this->getResult().getType()) and this->getKind() == UnaryOpKind::Not)
    {
        if (auto previous = dyn_cast_or_null<UnaryOp>(this->getInput().getDefiningOp()))
        {
            if (isa<BoolType>(previous.getResult().getType()) and previous.getKind() == UnaryOpKind::Not)
            {
                return previous.getInput();
            }
        }
    }
    return {};
}

//========================================================
// BinaryOp Definitions
//========================================================

auto mlir::atemhir::BinaryOp::verify() -> LogicalResult
{
    bool is_no_wrap = this->getNoSignedWrap() and this->getNoUnsignedWrap();

    if (not isa<IntType>(this->getType()) and is_no_wrap)
    {
        return emitError() << "nsw/nuw flags are only permitted on integer values";
    }

    bool is_no_wrap_ops = this->getKind() == BinaryOpKind::Add or this->getKind() == BinaryOpKind::Sub or this->getKind() == BinaryOpKind::Mul;

    if (is_no_wrap and not is_no_wrap_ops)
    {
        return emitError() << "nsw/nuw flags are only applicable to opcodes: 'add', 'sub', and 'mul'";
    }

    return success();
}

//========================================================
// ScopeOp Definitions
//========================================================

auto mlir::atemhir::ScopeOp::verify() -> LogicalResult
{
    return success();
}

auto mlir::atemhir::ScopeOp::getSuccessorRegions(RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &successors) -> void
{
    if (not point.isParent())
    {
        successors.push_back(RegionSuccessor{this->getODSResults(0)});
        return;
    }
    successors.push_back(RegionSuccessor{&this->getScopeRegion()});
}

auto mlir::atemhir::ScopeOp::build(OpBuilder &builder, OperationState &result, function_ref<void(OpBuilder &, Type &, Location)> scope_builder)
    -> void
{
    assert(scope_builder && "the callback 'scope_builder' must be present");

    OpBuilder::InsertionGuard guard(builder);
    Region *scope_region = result.addRegion();
    builder.createBlock(scope_region);

    Type yield_type;
    scope_builder(builder, yield_type, result.location);

    if (yield_type)
    {
        result.addTypes(TypeRange{yield_type});
    }
}

auto mlir::atemhir::ScopeOp::build(OpBuilder &builder, OperationState &result, function_ref<void(OpBuilder &, Location)> scope_builder) -> void
{
    assert(scope_builder && "the callback 'scope_builder' must be present");
    OpBuilder::InsertionGuard guard(builder);
    Region *scope_region = result.addRegion();
    builder.createBlock(scope_region);
    scope_builder(builder, result.location);
}