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