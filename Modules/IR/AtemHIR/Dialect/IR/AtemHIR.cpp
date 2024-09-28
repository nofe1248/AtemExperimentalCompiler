#include "IR/AtemHIR/Dialect/IR/AtemHIRDialect.hpp"

#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIRDialect.cpp.inc"

using namespace mlir;

void atemhir::AtemHIRDialect::initialize()
{
    this->registerOperations();
    this->registerTypes();
    this->registerAttributes();
}