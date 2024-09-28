#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "IR/AtemHIR/Dialect/IR/AtemHIRDialect.hpp"

using namespace mlir;
using namespace llvm;

auto main(int const argc, char **argv) -> int
{
    DialectRegistry registry;
    registry.insert<func::FuncDialect, atemhir::AtemHIRDialect, scf::SCFDialect, arith::ArithDialect, async::AsyncDialect>();
    registerCSEPass();
    registerCanonicalizerPass();
    return asMainReturnCode(MlirOptMain(argc, argv, "atemir-opt", registry));
}