module;

#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "IR/AtemHIR/Dialect/IR/AtemHIRDialect.hpp"

#include "AtemLexer.h"
#include "AtemParser.h"

export module Atem.Main;

import Atem.Parser;

namespace atem
{
namespace cl = llvm::cl;

static cl::opt<std::string> input_file_name{cl::Positional, cl::desc("<input atem file>"), cl::init("-"), cl::value_desc("filename")};

enum class InputType
{
    AtemSource,
    AtemHIR,
    MLIRStandard
};

static cl::opt<InputType> input_type{
    "x",
    cl::init(InputType::AtemSource),
    cl::desc("input file kind"),
    cl::values(clEnumValN(InputType::AtemSource, "atem_source", "load the input file as an Atem source.")),
    cl::values(clEnumValN(InputType::AtemHIR, "atem_hir", "load the input file as an Atem HIR source.")),
    cl::values(clEnumValN(InputType::MLIRStandard, "mlir_standard", "load the input file as a MLIR Standard source."))};

enum class Action
{
    None,
    DumpAST,
    DumpAtemHIR,
    DumpMLIRStandard,
    DumpMLIRLLVM,
    DumpLLVMIR,
    RunJIT
};

static cl::opt<Action> emit_action{
    "emit",
    cl::desc("select the kind of output desired"),
    cl::values(clEnumValN(Action::DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(Action::DumpAtemHIR, "atemhir", "output the Atem HIR after AST lowering")),
    cl::values(clEnumValN(Action::DumpMLIRStandard, "mlir", "output the MLIR Standard dialect after Atem IR lowering")),
    cl::values(clEnumValN(Action::DumpMLIRLLVM, "mlir-llvm", "output the MLIR LLVM dialect after llvm lowering")),
    cl::values(clEnumValN(Action::DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(Action::RunJIT, "jit", "run the code by invoking the main function through JIT"))};

enum class OptimizationLevel
{
    None,
    O1,
    O2,
    O3
};

static cl::opt<OptimizationLevel> optimization_level{"opt",
                                                     cl::init(OptimizationLevel::None),
                                                     cl::desc("optimization level"),
                                                     cl::values(clEnumValN(OptimizationLevel::None, "none", "disable optimization")),
                                                     cl::values(clEnumValN(OptimizationLevel::O1, "o1", "apply conservative optimization")),
                                                     cl::values(clEnumValN(OptimizationLevel::O2, "o2", "apply aggressive optimization")),
                                                     cl::values(clEnumValN(OptimizationLevel::O3, "o3", "apply every possible optimization"))};
} // namespace atem

export namespace atem
{
namespace cl = llvm::cl;

auto dumpAST() -> int
{
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_file_name);

    if (std::error_code ec = file_or_err.getError())
    {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return 1;
    }

    auto const buffer = std::string_view{file_or_err.get()->getBuffer()};
    antlr4::ANTLRInputStream input_stream(buffer);
    atem_antlr::AtemLexer lexer(&input_stream);
    antlr4::CommonTokenStream tokens(&lexer);
    atem_antlr::AtemParser parser(&tokens);

    auto *ast_root = parser.program();
    auto ast_str = ast_root->toStringTree(true);
    llvm::outs() << "Dumping Abstract Syntax Tree:\n";
    llvm::outs() << ast_str << "\n";
    return 0;
}

auto dumpAtemHIR(mlir::MLIRContext &context) -> int
{
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_file_name);

    if (std::error_code ec = file_or_err.getError())
    {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return 1;
    }

    auto const buffer = std::string_view{file_or_err.get()->getBuffer()};
    antlr4::ANTLRInputStream input_stream(buffer);
    atem_antlr::AtemLexer lexer(&input_stream);
    antlr4::CommonTokenStream tokens(&lexer);
    atem_antlr::AtemParser parser(&tokens);

    auto *ast_root = parser.program();

    parser::AtemHIRGenerator generator(context, input_file_name);

    if (mlir::ModuleOp hir_module = generator.buildAtemHIRModuleFromAST(ast_root))
    {
        llvm::outs() << "Dumping Atem HIR:\n";
        hir_module->dump();
    }
    else
    {
        llvm::errs() << "Compilation failed\n";
    }

    return 0;
}

auto AtemMain(int argc, char *argv[]) -> int
{
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "Atem Experimental Compiler\n");

    if (emit_action == Action::DumpAST)
    {
        return dumpAST();
    }

    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::arith::ArithDialect, mlir::affine::AffineDialect,
                    mlir::atemhir::AtemHIRDialect>();

    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::atemhir::AtemHIRDialect>();

    context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
        diag.print(llvm::errs());
    });

    if (emit_action == Action::DumpAtemHIR)
    {
        return dumpAtemHIR(context);
    }

    return 0;
}
} // namespace atem

export auto main(int argc, char *argv[]) -> int
{
    return atem::AtemMain(argc, argv);
}