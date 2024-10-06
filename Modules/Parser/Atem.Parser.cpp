
module;

#include <any>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "IR/AtemHIR/Dialect/IR/AtemHIRAttrs.hpp"
#include "IR/AtemHIR/Dialect/IR/AtemHIRDialect.hpp"
#include "IR/AtemHIR/Dialect/IR/AtemHIROps.hpp"

#include "AtemParser.h"
#include "AtemParserBaseVisitor.h"

export module Atem.Parser;

export namespace atem::parser
{
using namespace atem_antlr;

auto toString(mlir::Type type) -> std::string
{
    std::string result;
    llvm::raw_string_ostream os(result);
    type.print(os);
    return os.str();
}

class AtemHIRGenerator final : public AtemParserBaseVisitor
{
private:
    mlir::MLIRContext &context;
    mlir::ModuleOp hir_module;
    mlir::OpBuilder builder;

    std::string file_name;
    std::vector<std::string> file_buf;

    std::size_t error_count = 0, warning_count = 0;

    struct VariableDeclarationInfo
    {
        std::variant<AtemParser::Variable_declContext *, AtemParser::Constant_declContext *> decl_ast;
        AtemParser::ExprContext *init_expr;
    };
    using ScopeHashTableScopeT = llvm::ScopedHashTable<llvm::StringRef, VariableDeclarationInfo>;

    ScopeHashTableScopeT symbol_table;
    llvm::StringMap<mlir::atemhir::FunctionOp> function_map;

    auto generateMLIRLocationFromASTLocation(antlr4::ParserRuleContext const *ast_node) -> mlir::Location
    {
        return mlir::FileLineColLoc::get(builder.getStringAttr(this->file_name), ast_node->getStart()->getLine(),
                                         ast_node->getStart()->getCharPositionInLine());
    }

    auto declareVariable(AtemParser::Variable_declContext *var_decl_ast, AtemParser::ExprContext *init_expr_ast) -> llvm::LogicalResult
    {
        if (this->symbol_table.count(var_decl_ast->Identifier()->getText()) > 0)
        {
            return mlir::failure();
        }
        this->symbol_table.insert(var_decl_ast->Identifier()->getText(), {var_decl_ast, init_expr_ast});
        return mlir::success();
    }

    template <typename... Args>
    auto formatDiagnostics(antlr4::ParserRuleContext const *ast_node, mlir::InFlightDiagnostic &diag, std::string_view fmt_str, Args &&...fmt_args)
        -> void
    {
        auto line_start = ast_node->getStart()->getLine();
        auto col_start = ast_node->getStart()->getCharPositionInLine();
        auto line_end = ast_node->getStop()->getLine();
        auto col_end = ast_node->getStop()->getCharPositionInLine() + ast_node->getStop()->getText().length();
        diag << std::format("in file: {} (line: {} col: {})\n", this->file_name, line_start, col_start);

        std::string severity_str;
        switch (diag.getUnderlyingDiagnostic()->getSeverity())
        {
        case mlir::DiagnosticSeverity::Error:
            severity_str = "error";
            break;
        case mlir::DiagnosticSeverity::Warning:
            severity_str = "warn";
            break;
        case mlir::DiagnosticSeverity::Remark:
            severity_str = "remark";
            break;
        case mlir::DiagnosticSeverity::Note:
            severity_str = "note";
            break;
        default:
            llvm_unreachable("unknown diagnostic severity");
        }

        diag << std::format("  {}: {}\n", severity_str, std::format(std::runtime_format(fmt_str), std::forward<Args>(fmt_args)...));
        if (line_start == line_end)
        {
            auto length = col_end - col_start;
            std::string underline_ident(col_start, ' ');
            std::string underline(length, '^');
            diag << std::format("    {}\n", this->file_buf[line_start - 1]);
            diag << std::format("    {}{}\n", underline_ident, underline);
        }
        else
        {
            for (std::size_t i = line_start - 1; i < line_end; ++i)
            {
                diag << std::format("    {}\n", this->file_buf[i]);

                auto length = this->file_buf[i].length();
                if (i == line_start - 1)
                {
                    diag << std::format("    {}{}\n", std::string(col_start, ' '), std::string(length - col_start, '^'));
                    continue;
                }
                if (i == line_end - 1)
                {
                    diag << std::format("    {}\n", std::string(length - col_end, '^'));
                    continue;
                }
                diag << std::format("    {}\n", std::string(length, '^'));
            }
        }
    }

    template <typename... Args>
    auto emitError(antlr4::ParserRuleContext const *ast_node, std::string_view fmt_str, Args &&...fmt_args) -> mlir::InFlightDiagnostic
    {
        auto diag = mlir::emitError(this->generateMLIRLocationFromASTLocation(ast_node));
        this->formatDiagnostics(ast_node, diag, fmt_str, std::forward<Args>(fmt_args)...);
        ++this->error_count;
        return diag;
    }

    template <typename... Args>
    auto emitWarn(antlr4::ParserRuleContext const *ast_node, std::string_view fmt_str, Args &&...fmt_args) -> mlir::InFlightDiagnostic
    {
        auto diag = mlir::emitWarning(this->generateMLIRLocationFromASTLocation(ast_node));
        this->formatDiagnostics(ast_node, diag, fmt_str, std::forward<Args>(fmt_args)...);
        ++this->warning_count;
        return diag;
    }

    template <typename... Args>
    auto emitRemark(antlr4::ParserRuleContext const *ast_node, std::string_view fmt_str, Args &&...fmt_args) -> mlir::InFlightDiagnostic
    {
        auto diag = mlir::emitRemark(this->generateMLIRLocationFromASTLocation(ast_node));
        this->formatDiagnostics(ast_node, diag, fmt_str, std::forward<Args>(fmt_args)...);
        return diag;
    }

    [[noreturn]] auto unreachable(antlr4::ParserRuleContext const *ast_node, llvm::StringRef info) -> void
    {
        auto const str = std::format("file {} line {}:col {} {}", this->file_name, ast_node->getStart()->getLine(),
                                     ast_node->getStart()->getCharPositionInLine(), std::string_view{info});
        llvm_unreachable(str.c_str());
    }

public:
    explicit AtemHIRGenerator(mlir::MLIRContext &context, llvm::StringRef source)
        : context(context), builder(&context), file_name(std::filesystem::absolute(std::string_view{source}).string())
    {
        std::ifstream source_file(std::string{source});
        std::string line;

        if (source_file.is_open())
        {
            while (std::getline(source_file, line))
            {
                this->file_buf.push_back(line);
                line.clear();
            }
        }
        else
        {
            llvm_unreachable("Cannot open input file");
        }
    }
    auto buildAtemHIRModuleFromAST(AtemParser::ProgramContext *ast_root) -> mlir::ModuleOp
    {
        this->hir_module = mlir::ModuleOp::create(this->builder.getUnknownLoc());

        for (auto *decl_ptr : ast_root->decls()->decl())
        {
            if (auto *func_ptr = decl_ptr->function_decl())
            {
                auto func_op = std::any_cast<mlir::atemhir::FunctionOp>(this->visitFunction_decl(func_ptr));
                if (func_op)
                {
                    if (this->function_map.count(func_op.getSymName()) > 0)
                    {
                        mlir::emitError(this->generateMLIRLocationFromASTLocation(func_ptr))
                            << "function declaration already exists : " << func_op.getSymName();
                    }
                    else
                    {
                        this->function_map.insert({func_op.getSymName(), func_op});
                    }
                }
            }
            else if (auto *var_ptr = decl_ptr->variable_decl())
            {
                this->emitError(var_ptr, "global variable declarations are not supported yet");
            }
            else if (auto *const_ptr = decl_ptr->constant_decl())
            {
                this->emitError(const_ptr, "global constant declarations are not supported yet");
            }
            else if (auto *struct_ptr = decl_ptr->struct_decl())
            {
                this->emitError(struct_ptr, "structs are not supported yet");
            }
            else
            {
                this->unreachable(decl_ptr, "unknown declaration type in AST");
            }
        }

        llvm::outs() << std::format("{} errors, {} warning generated\n", this->error_count, this->warning_count);

        if (mlir::verify(this->hir_module).failed())
        {
            return nullptr;
        }

        if (this->error_count > 0)
        {
            return nullptr;
        }

        return this->hir_module;
    }

    auto visitLiteral_expr(AtemParser::Literal_exprContext *ast_node) -> std::any override
    {
        if (auto *ptr = ast_node->integer_literal())
        {
            return this->visitInteger_literal(ptr);
        }
        if (auto *ptr = ast_node->string_literal())
        {
            this->emitError(ptr, "string literals are not supported yet");
            return mlir::Value{nullptr};
        }
        if (auto *ptr = ast_node->FloatingPointLiteral())
        {
            llvm::APFloat fp64(llvm::APFloat::IEEEdouble()), fp80(llvm::APFloat::x87DoubleExtended()), fp128(llvm::APFloat::IEEEquad()), final(0.0);
            uint8_t which_to_use = 64;
            auto raw_fp = ast_node->getText();
            if ((void)fp64.convertFromString(raw_fp, llvm::APFloat::roundingMode::NearestTiesToAway); fp64.isInfinity())
            {
                which_to_use = 80;
                if ((void)fp80.convertFromString(raw_fp, llvm::APFloat::roundingMode::NearestTiesToAway); fp80.isInfinity())
                {
                    which_to_use = 128;
                    if ((void)fp128.convertFromString(raw_fp, llvm::APFloat::roundingMode::NearestTiesToAway); fp128.isInfinity())
                    {
                        final = fp128;
                        this->emitWarn(ast_node, "floating-point literal cannot fit in floating-point types, precision loss is expected");
                    }
                    else
                    {
                        final = fp128;
                    }
                }
                else
                {
                    final = fp80;
                }
            }
            else
            {
                final = fp64;
            }
            mlir::Type fp_type;
            if (which_to_use == 64)
            {
                fp_type = mlir::atemhir::FP64Type::get(&this->context);
            }
            if (which_to_use == 80)
            {
                fp_type = mlir::atemhir::FP80Type::get(&this->context);
            }
            if (which_to_use == 128)
            {
                fp_type = mlir::atemhir::FP128Type::get(&this->context);
            }
            auto fp_attr = mlir::atemhir::FPAttr::get(&this->context, fp_type, final);
            auto const_op = this->builder.create<mlir::atemhir::ConstantOp>(this->generateMLIRLocationFromASTLocation(ast_node), fp_type, fp_attr);
            return mlir::Value{const_op.getRes()};
        }
        if (auto *ptr = ast_node->KeywordFalse())
        {
            auto bool_type = mlir::atemhir::BoolType::get(&this->context);
            auto bool_attr = mlir::atemhir::BoolAttr::get(&this->context, bool_type, false);
            auto const_op =
                this->builder.create<mlir::atemhir::ConstantOp>(this->generateMLIRLocationFromASTLocation(ast_node), bool_type, bool_attr);
            return mlir::Value{const_op.getRes()};
        }
        if (auto *ptr = ast_node->KeywordTrue())
        {
            auto bool_type = mlir::atemhir::BoolType::get(&this->context);
            auto bool_attr = mlir::atemhir::BoolAttr::get(&this->context, bool_type, true);
            auto const_op =
                this->builder.create<mlir::atemhir::ConstantOp>(this->generateMLIRLocationFromASTLocation(ast_node), bool_type, bool_attr);
            return mlir::Value{const_op.getRes()};
        }
        if (auto *ptr = ast_node->KeywordNull())
        {
            auto unit_type = mlir::atemhir::UnitType::get(&this->context);
            auto zeroinit_op = this->builder.create<mlir::atemhir::ZeroInitOp>(this->generateMLIRLocationFromASTLocation(ast_node), unit_type);
            return mlir::Value{zeroinit_op.getRes()};
        }
        if (auto *ptr = ast_node->KeywordUndefined())
        {
            auto unit_type = mlir::atemhir::UnitType::get(&this->context);
            auto undefined_op = this->builder.create<mlir::atemhir::UndefinedOp>(this->generateMLIRLocationFromASTLocation(ast_node), unit_type);
            return mlir::Value{undefined_op.getRes()};
        }
        this->unreachable(ast_node, "unknown literal in AST");
    }

    auto visitInteger_literal(AtemParser::Integer_literalContext *ast_node) -> std::any override
    {
        auto raw_int = ast_node->getText();
        uint8_t radix = 10;
        if (ast_node->BinaryLiteral())
        {
            radix = 2;
        }
        else if (ast_node->OctalLiteral())
        {
            radix = 8;
        }
        else if (ast_node->DecimalDigits() or ast_node->DecimalLiteral())
        {
            radix = 10;
        }
        else if (ast_node->HexadecimalLiteral())
        {
            radix = 16;
        }
        else
        {
            this->unreachable(ast_node, "unknown integer literal in AST");
        }
        if (radix != 10)
        {
            raw_int.erase(0, 2);
        }
        std::size_t bit_width = pow(2, ceil(log2(llvm::APInt::getBitsNeeded(raw_int, radix))));
        if (bit_width < 8)
        {
            bit_width = 8;
        }
        llvm::APInt int_value(bit_width, raw_int, radix);
        auto int_type = mlir::atemhir::IntType::get(&this->context, bit_width, true);
        auto int_attr = mlir::atemhir::IntAttr::get(&this->context, int_type, int_value);
        auto const_op = this->builder.create<mlir::atemhir::ConstantOp>(this->generateMLIRLocationFromASTLocation(ast_node), int_type, int_attr);
        return mlir::Value{const_op.getRes()};
    }

    auto tryImplicitCast(antlr4::ParserRuleContext const *ast_node, mlir::Value source, mlir::Type result_type) -> mlir::Value
    {
        using enum mlir::atemhir::CastKind;
        auto source_type = source.getType();
        if (source_type == result_type)
        {
            return source;
        }
        auto loc = this->generateMLIRLocationFromASTLocation(ast_node);
        if (mlir::isa<mlir::atemhir::BoolType>(source_type) and mlir::isa<mlir::atemhir::IntType>(result_type))
        {
            auto cast_op = this->builder.create<mlir::atemhir::CastOp>(loc, result_type, bool_to_int, source);
            return mlir::Value{cast_op.getResult()};
        }
        if (mlir::isa<mlir::atemhir::IntType>(source_type) and mlir::isa<mlir::atemhir::BoolType>(result_type))
        {
            auto cast_op = this->builder.create<mlir::atemhir::CastOp>(loc, result_type, int_to_bool, source);
            return mlir::Value{cast_op.getResult()};
        }
        if (auto source_int = mlir::dyn_cast<mlir::atemhir::IntType>(source_type), result_int = mlir::dyn_cast<mlir::atemhir::IntType>(result_type);
            source_int and result_int)
        {
            if (source_int.isSigned() and result_int.isUnsigned())
            {
                this->emitError(ast_node, "cannot implicitly cast a signed integer to an unsigned integer, use an explicit cast instead");
            }
            if (source_int.getWidth() <= result_int.getWidth())
            {
                auto cast_op = this->builder.create<mlir::atemhir::CastOp>(loc, result_type, int_promotion, source);
                return mlir::Value{cast_op.getResult()};
            }
            this->emitError(ast_node, "cannot implicitly narrow {} to {}, use an explicit cast instead", toString(source_type),
                            toString(result_type));
            return mlir::Value{nullptr};
        }
        if (auto source_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(source_type),
            result_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(result_type);
            source_fp and result_fp)
        {
            if (source_fp.getWidth() <= result_fp.getWidth())
            {
                auto cast_op = this->builder.create<mlir::atemhir::CastOp>(loc, result_type, float_promotion, source);
                return mlir::Value{cast_op.getResult()};
            }
            this->emitError(ast_node, "cannot implicitly narrow {} to {}, use an explicit cast instead", toString(source_type),
                            toString(result_type));
            return mlir::Value{nullptr};
        }
        {
            auto source_int = mlir::dyn_cast<mlir::atemhir::IntType>(source_type);
            auto result_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(result_type);
            if (source_int and result_fp)
            {
                llvm::APSInt fp_max;
                bool is_exact;
                llvm::APFloat::getLargest(result_fp.getFloatSemantics())
                    .convertToInteger(fp_max, llvm::APFloat::roundingMode::NearestTiesToAway, &is_exact);
                if (llvm::APSInt::getMaxValue(source_int.getWidth(), source_int.isUnsigned()) > fp_max)
                {
                    this->emitError(ast_node, "cannot implicitly narrow {} to {}, data loss possible", toString(source_type), toString(result_type));
                    return mlir::Value{nullptr};
                }
                auto cast_op = this->builder.create<mlir::atemhir::CastOp>(loc, result_type, int_to_float, source);
                return mlir::Value{cast_op.getResult()};
            }
        }
        {
            auto source_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(source_type);
            auto result_int = mlir::dyn_cast<mlir::atemhir::IntType>(result_type);
            if (source_fp and result_int)
            {
                this->emitError(ast_node, "cannot implicitly cast floating-point type {} to integer type {}, use an explicit cast instead",
                                toString(source_type), toString(result_type));
                return mlir::Value{nullptr};
            }
        }
        this->emitError(ast_node, "unsupported implicit cast from {} to {}", toString(source_type), toString(result_type));
        return mlir::Value{nullptr};
    }

    auto visitType_expr(AtemParser::Type_exprContext *ctx) -> mlir::Type
    {
        if (auto *ptr = dynamic_cast<AtemParser::Simple_type_expressionContext *>(ctx))
        {
            return std::any_cast<mlir::Type>(this->visitSimple_type_expression(ptr));
        }
        if (auto *ptr = dynamic_cast<AtemParser::Array_type_expressionContext *>(ctx))
        {
            return std::any_cast<mlir::Type>(this->visitArray_type_expression(ptr));
        }
        if (auto *ptr = dynamic_cast<AtemParser::Pointer_type_expressionContext *>(ctx))
        {
            return std::any_cast<mlir::Type>(this->visitPointer_type_expression(ptr));
        }
        if (auto *ptr = dynamic_cast<AtemParser::Struct_type_expressionContext *>(ctx))
        {
            return std::any_cast<mlir::Type>(this->visitStruct_type_expression(ptr));
        }
        if (auto *ptr = dynamic_cast<AtemParser::Function_type_expressionContext *>(ctx))
        {
            return std::any_cast<mlir::Type>(this->visitFunction_type_expression(ptr));
        }
        if (auto *ptr = dynamic_cast<AtemParser::Identifier_type_expressionContext *>(ctx))
        {
            this->emitError(ptr, "user-defined types are not supported yet");
            return mlir::Type{nullptr};
        }
        this->unreachable(ctx, "unknown type in AST");
    }

    auto visitFunction_decl(AtemParser::Function_declContext *ast_node) -> std::any override
    {
        llvm::SmallVector<mlir::Type> param_types{};
        llvm::SmallVector<mlir::Type, 1> result_types{};

        if (auto *func_type_ptr = ast_node->function_type_expr())
        {
            for (auto *func_param_ptr : func_type_ptr->function_parameter_list()->function_parameter())
            {
                auto type = this->visitType_expr(func_param_ptr->type_expr());

                if (type)
                {
                    param_types.push_back(type);
                }
            }
            if (auto *func_ret_ptr = func_type_ptr->type_expr())
            {
                auto ret_type = this->visitType_expr(func_ret_ptr);

                if (ret_type)
                {
                    result_types.push_back(ret_type);
                }
            }
            else
            {
                result_types.push_back(mlir::atemhir::UnitType::get(&this->context));
            }
        }

        auto func_type = mlir::atemhir::FunctionType::get(&this->context, param_types, result_types);

        auto func_name = ast_node->Identifier()->getText();

        this->builder.setInsertionPointToEnd(this->hir_module.getBody());

        auto func = this->builder.create<mlir::atemhir::FunctionOp>(this->generateMLIRLocationFromASTLocation(ast_node), func_name, func_type,
                                                                    mlir::ArrayAttr{}, mlir::ArrayAttr{});

        func.addEntryBlock();
        mlir::Block &entry_block = func.front();

        this->builder.setInsertionPointToEnd(&entry_block);

        if (auto *scope_ptr = ast_node->block_or_expr()->scope_expr())
        {
            for (auto *stmt_ptr : scope_ptr->stmts()->stmt())
            {
                this->visitStmt(stmt_ptr);
            }
        }
        if (auto *expr_ptr = ast_node->block_or_expr()->expr())
        {
            auto expr_result = std::any_cast<mlir::Value>(this->visit(expr_ptr));
            if (expr_result.getType() != result_types.front())
            {
                auto casted_expr = this->tryImplicitCast(expr_ptr, expr_result, result_types.front());
                if (casted_expr)
                {
                    this->builder.create<mlir::atemhir::ReturnOp>(this->generateMLIRLocationFromASTLocation(expr_ptr), mlir::ValueRange{casted_expr});
                }
                else
                {
                    this->emitError(expr_ptr, "fail to implicitly cast from {} to {}", toString(expr_result.getType()),
                                    toString(result_types.front()));
                }
            }
            else
            {
                this->builder.create<mlir::atemhir::ReturnOp>(this->generateMLIRLocationFromASTLocation(expr_ptr), mlir::ValueRange{expr_result});
            }
        }

        mlir::atemhir::ReturnOp return_op{};
        if (not entry_block.empty())
        {
            return_op = mlir::dyn_cast<mlir::atemhir::ReturnOp>(entry_block.back());
        }
        if (not return_op)
        {
            if (func_type.isReturningUnit())
            {
                auto unit_const = this->builder.create<mlir::atemhir::ConstantOp>(
                    this->generateMLIRLocationFromASTLocation(ast_node), func_type.getResults().front(),
                    mlir::atemhir::UnitAttr::get(&this->context, mlir::dyn_cast<mlir::atemhir::UnitType>(func_type.getResults().front())));
                this->builder.create<mlir::atemhir::ReturnOp>(this->generateMLIRLocationFromASTLocation(ast_node),
                                                              mlir::ValueRange{unit_const.getRes()});
            }
            else
            {
                this->emitError(ast_node, "function {} returning non-Unit must contain a return expression at the end", func_name);
                return mlir::atemhir::FunctionOp{nullptr};
            }
        }
        else
        {
            if (return_op.getOperandTypes().front() != result_types.front())
            {
                auto return_operand = return_op.getOperands().front();
                this->builder.setInsertionPointAfterValue(return_operand);
                auto casted_return_operand = this->tryImplicitCast(ast_node, return_operand, result_types.front());
                if (not casted_return_operand)
                {
                    this->emitError(ast_node, "cannot convert {} to return type {}", toString(return_operand.getType()),
                                    toString(result_types.front()));
                }
                else
                {
                    return_op.setOperand(0, casted_return_operand);
                }
            }
        }

        if (func_name != "main")
        {
            func.setPrivate();
        }

        this->builder.setInsertionPointToEnd(this->hir_module.getBody());

        return func;
    }

    auto visitReturn_expr(AtemParser::Return_exprContext *ast_node) -> std::any override
    {
        if (auto *expr_ptr = ast_node->expr())
        {
            auto return_value = std::any_cast<mlir::Value>(this->visit(expr_ptr));
            auto return_op =
                this->builder.create<mlir::atemhir::ReturnOp>(this->generateMLIRLocationFromASTLocation(expr_ptr), mlir::ValueRange{return_value});
        }
        else
        {
            auto unit_type = mlir::atemhir::UnitType::get(&this->context);
            auto unit_const = this->builder.create<mlir::atemhir::ConstantOp>(this->generateMLIRLocationFromASTLocation(ast_node), unit_type,
                                                                              mlir::atemhir::UnitAttr::get(&this->context, unit_type));
            auto return_op = this->builder.create<mlir::atemhir::ReturnOp>(this->generateMLIRLocationFromASTLocation(ast_node),
                                                                           mlir::ValueRange{unit_const.getRes()});
        }
        return mlir::Value{nullptr};
    }

    auto visitSimple_type_expression(AtemParser::Simple_type_expressionContext *ast_node) -> std::any override
    {
        if (auto *ptr = dynamic_cast<AtemParser::Bool_type_expressionContext *>(ast_node->simple_type_expr()))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::BoolType>(this->visitBool_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::Int_type_expressionContext *>(ast_node->simple_type_expr()))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::IntType>(this->visitInt_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::Float_type_expressionContext *>(ast_node->simple_type_expr()))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::AtemHIRFPTypeInterface>(this->visitFloat_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::String_type_expressionContext *>(ast_node->simple_type_expr()))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::StringType>(this->visitString_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::Uint_type_expressionContext *>(ast_node->simple_type_expr()))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::IntType>(this->visitUint_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::Noreturn_type_expressionContext *>(ast_node->simple_type_expr()))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::NoreturnType>(this->visitNoreturn_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::Rune_type_expressionContext *>(ast_node->simple_type_expr()))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::RuneType>(this->visitRune_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::Unit_type_expressionContext *>(ast_node->simple_type_expr()))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::UnitType>(this->visitUnit_type_expression(ptr))};
        }
        this->unreachable(ast_node, "unknown simple type in AST");
    }

    auto visitBool_type_expression(AtemParser::Bool_type_expressionContext *ast_node) -> std::any override
    {
        return mlir::atemhir::BoolType::get(&this->context);
    }

    auto visitInt_type_expression(AtemParser::Int_type_expressionContext *ast_node) -> std::any override
    {
        const auto width = std::stoi(ast_node->getText().substr(3));
        return mlir::atemhir::IntType::get(&this->context, width, true);
    }

    auto visitUint_type_expression(AtemParser::Uint_type_expressionContext *ast_node) -> std::any override
    {
        const auto width = std::stoi(ast_node->getText().substr(4));
        return mlir::atemhir::IntType::get(&this->context, width, false);
    }

    auto visitFloat_type_expression(AtemParser::Float_type_expressionContext *ast_node) -> std::any override
    {
        if (ast_node->getText() == "Float16")
        {
            return mlir::atemhir::AtemHIRFPTypeInterface{mlir::atemhir::FP16Type::get(&this->context)};
        }
        if (ast_node->getText() == "Float32")
        {
            return mlir::atemhir::AtemHIRFPTypeInterface{mlir::atemhir::FP32Type::get(&this->context)};
        }
        if (ast_node->getText() == "Float64")
        {
            return mlir::atemhir::AtemHIRFPTypeInterface{mlir::atemhir::FP64Type::get(&this->context)};
        }
        if (ast_node->getText() == "Float80")
        {
            return mlir::atemhir::AtemHIRFPTypeInterface{mlir::atemhir::FP80Type::get(&this->context)};
        }
        if (ast_node->getText() == "Float128")
        {
            return mlir::atemhir::AtemHIRFPTypeInterface{mlir::atemhir::FP128Type::get(&this->context)};
        }
        this->unreachable(ast_node, "unknown floating-point type in AST");
    }

    auto visitString_type_expression(AtemParser::String_type_expressionContext *ast_node) -> std::any override
    {
        this->emitError(ast_node, "String types are not supported yet");
        return mlir::atemhir::StringType::get(&this->context);
    }

    auto visitRune_type_expression(AtemParser::Rune_type_expressionContext *ast_node) -> std::any override
    {
        this->emitError(ast_node, "Rune types are not supported yet");
        return mlir::atemhir::RuneType::get(&this->context);
    }

    auto visitNoreturn_type_expression(AtemParser::Noreturn_type_expressionContext *ast_node) -> std::any override
    {
        return mlir::atemhir::NoreturnType::get(&this->context);
    }

    auto visitUnit_type_expression(AtemParser::Unit_type_expressionContext *ast_node) -> std::any override
    {
        return mlir::atemhir::UnitType::get(&this->context);
    }
};
} // namespace atem::parser