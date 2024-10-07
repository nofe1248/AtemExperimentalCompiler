
module;

#include <any>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <ranges>
#include <string>
#include <utility>
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
#include "IR/AtemHIR/Dialect/IR/AtemHIRTypes.hpp"

#include "AtemParser.h"
#include "AtemParserBaseVisitor.h"

#include <AtemLexer.h>

export module Atem.Parser;

export namespace llvm
{
template <>
struct DenseMapInfo<std::string, void>
{
    static inline auto getEmptyKey() -> std::string
    {
        return std::string(reinterpret_cast<const char *>(~static_cast<uintptr_t>(0)), 0);
    }

    static inline auto getTombstoneKey() -> std::string
    {
        return std::string(reinterpret_cast<const char *>(~static_cast<uintptr_t>(1)), 0);
    }

    static auto getHashValue(std::string const &Val) -> unsigned
    {
        return std::hash<std::string>()(Val);
    }

    static auto isEqual(std::string const &LHS, std::string const &RHS) -> bool
    {
        if (RHS.data() == getEmptyKey().data())
        {
            return LHS.data() == getEmptyKey().data();
        }
        if (RHS.data() == getTombstoneKey().data())
        {
            return LHS.data() == getTombstoneKey().data();
        }
        return LHS == RHS;
    }
};
} // namespace llvm

export namespace atem::parser
{
using namespace atem_antlr;

auto toString(mlir::Type type) -> std::string
{
    if (auto atem_hir_type = mlir::dyn_cast<mlir::atemhir::AtemHIRUtilTypeInterface>(type))
    {
        return atem_hir_type.toAtemTypeString();
    }
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
        mlir::Value var_ssa_value;
        std::optional<antlr4::ParserRuleContext const *> var_decl_ast = std::nullopt;
    };
    using SymbolTableScopeT = llvm::ScopedHashTable<std::string, VariableDeclarationInfo>::ScopeTy;

    llvm::ScopedHashTable<std::string, VariableDeclarationInfo> symbol_table;
    llvm::StringMap<mlir::atemhir::FunctionType> function_map;

    auto generateMLIRLocationFromASTLocation(antlr4::ParserRuleContext const *ast_node) -> mlir::Location
    {
        return mlir::FileLineColLoc::get(builder.getStringAttr(this->file_name), ast_node->getStart()->getLine(),
                                         ast_node->getStart()->getCharPositionInLine());
    }

    auto declareVariable(std::string const &var_name, mlir::Value var_ssa_value,
                         std::optional<antlr4::ParserRuleContext const *> var_decl_ast = std::nullopt) -> llvm::LogicalResult
    {
        if (this->symbol_table.count(var_name))
        {
            return mlir::failure();
        }
        this->symbol_table.insert(var_name, VariableDeclarationInfo{var_ssa_value, var_decl_ast});
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

        SymbolTableScopeT var_scope(this->symbol_table);
        for (auto *decl_ptr : ast_root->decls()->decl())
        {
            if (auto *func_ptr = decl_ptr->function_decl())
            {
                auto func_name = func_ptr->Identifier()->getText();
                auto func_type =
                    std::any_cast<mlir::atemhir::FunctionType>(this->visitFunction_type_expr(decl_ptr->function_decl()->function_type_expr()));
                if (this->function_map.count(func_name) > 0)
                {
                    mlir::emitError(this->generateMLIRLocationFromASTLocation(func_ptr)) << "function declaration already exists : " << func_name;
                }
                else
                {
                    this->function_map.insert({func_name, func_type});
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

        for (auto *decl_ptr : ast_root->decls()->decl())
        {
            if (auto *func_ptr = decl_ptr->function_decl())
            {
                auto func_op = std::any_cast<mlir::atemhir::FunctionOp>(this->visitFunction_decl(func_ptr));
                this->builder.setInsertionPointToEnd(this->hir_module.getBody());
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

    auto getIntAttrFromLiteral(AtemParser::Integer_literalContext *ast_node) -> mlir::atemhir::IntAttr
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
        return int_attr;
    }

    auto visitInteger_literal(AtemParser::Integer_literalContext *ast_node) -> std::any override
    {
        auto int_attr = this->getIntAttrFromLiteral(ast_node);
        auto const_op =
            this->builder.create<mlir::atemhir::ConstantOp>(this->generateMLIRLocationFromASTLocation(ast_node), int_attr.getType(), int_attr);
        return mlir::Value{const_op.getRes()};
    }

    auto implicitlyCastable(mlir::Type source_type, mlir::Type result_type) const -> std::optional<mlir::atemhir::CastKind>
    {
        using enum mlir::atemhir::CastKind;

        if (source_type == result_type)
        {
            return static_cast<mlir::atemhir::CastKind>(-1);
        }
        if (mlir::isa<mlir::atemhir::BoolType>(source_type) and mlir::isa<mlir::atemhir::IntType>(result_type))
        {
            return bool_to_int;
        }
        if (mlir::isa<mlir::atemhir::IntType>(source_type) and mlir::isa<mlir::atemhir::BoolType>(result_type))
        {
            return std::nullopt;
        }
        if (auto source_int = mlir::dyn_cast<mlir::atemhir::IntType>(source_type), result_int = mlir::dyn_cast<mlir::atemhir::IntType>(result_type);
            source_int and result_int)
        {
            if (source_int.isSigned() and result_int.isUnsigned())
            {
                return std::nullopt;
            }
            if (source_int.getWidth() <= result_int.getWidth())
            {
                return int_promotion;
            }
            return std::nullopt;
        }
        if (auto source_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(source_type),
            result_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(result_type);
            source_fp and result_fp)
        {
            if (source_fp.getWidth() <= result_fp.getWidth())
            {
                return float_promotion;
            }
            return std::nullopt;
        }
        {
            auto source_int = mlir::dyn_cast<mlir::atemhir::IntType>(source_type);
            auto result_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(result_type);
            if (source_int and result_fp)
            {
                auto fp = llvm::APFloat(result_fp.getFloatSemantics(), llvm::APSInt::getMaxValue(source_int.getWidth(), source_int.isUnsigned()));

                if (fp.isInfinity())
                {
                    return std::nullopt;
                }
                return int_to_float;
            }
        }
        {
            auto source_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(source_type);
            auto result_int = mlir::dyn_cast<mlir::atemhir::IntType>(result_type);
            if (source_fp and result_int)
            {
                return std::nullopt;
            }
        }
        return std::nullopt;
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
            this->emitError(ast_node, "cannot implicitly narrow '{}' to '{}', use an explicit cast instead", toString(source_type),
                            toString(result_type));
            return mlir::Value{nullptr};
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
            this->emitError(ast_node, "cannot implicitly narrow '{}' to '{}', use an explicit cast instead", toString(source_type),
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
            this->emitError(ast_node, "cannot implicitly narrow '{}' to '{}', use an explicit cast instead", toString(source_type),
                            toString(result_type));
            return mlir::Value{nullptr};
        }
        {
            auto source_int = mlir::dyn_cast<mlir::atemhir::IntType>(source_type);
            auto result_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(result_type);
            if (source_int and result_fp)
            {
                auto fp = llvm::APFloat(result_fp.getFloatSemantics(), llvm::APSInt::getMaxValue(source_int.getWidth(), source_int.isUnsigned()));

                if (fp.isInfinity())
                {
                    this->emitError(ast_node, "cannot implicitly narrow '{}' to '{}', data loss possible", toString(source_type),
                                    toString(result_type));
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
                this->emitError(ast_node, "cannot implicitly cast floating-point type '{}' to integer type '{}', use an explicit cast instead",
                                toString(source_type), toString(result_type));
                return mlir::Value{nullptr};
            }
        }
        this->emitError(ast_node, "unsupported implicit cast from '{}' to '{}'", toString(source_type), toString(result_type));
        return mlir::Value{nullptr};
    }

    auto resolveCommonType(mlir::Type lhs, mlir::Type rhs) -> std::optional<mlir::Type>
    {
        if (lhs == rhs)
        {
            return lhs;
        }
        if (this->implicitlyCastable(rhs, lhs).has_value())
        {
            return lhs;
        }
        if (this->implicitlyCastable(lhs, rhs).has_value())
        {
            return rhs;
        }
        return std::nullopt;
    }

    auto resolveCommonType(mlir::TypeRange types) -> std::optional<mlir::Type>
    {
        assert(types.size() >= 1);
        if (types.size() == 1)
        {
            return types.front();
        }

        auto common_type_opt = this->resolveCommonType(types[0], types[1]);
        if (not common_type_opt.has_value())
        {
            return std::nullopt;
        }
        for (std::size_t i = 2; i < types.size(); ++i)
        {
            if (common_type_opt.has_value())
            {
                common_type_opt = this->resolveCommonType(common_type_opt.value(), types[i]);
            }
            else
            {
                return std::nullopt;
            }
        }
        return common_type_opt;
    }

    auto visitType_expr(AtemParser::Type_exprContext *ctx) -> mlir::Type
    {
        if (auto *ptr = dynamic_cast<AtemParser::Simple_type_expressionContext *>(ctx))
        {
            return std::any_cast<mlir::Type>(this->visitSimple_type_expression(ptr));
        }
        if (auto *ptr = dynamic_cast<AtemParser::Array_type_expressionContext *>(ctx))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::ArrayType>(this->visitArray_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::Pointer_type_expressionContext *>(ctx))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::PointerType>(this->visitPointer_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::Struct_type_expressionContext *>(ctx))
        {
            return mlir::Type{std::any_cast<mlir::Type>(this->visitStruct_type_expression(ptr))};
        }
        if (auto *ptr = dynamic_cast<AtemParser::Function_type_expressionContext *>(ctx))
        {
            return mlir::Type{std::any_cast<mlir::atemhir::FunctionType>(this->visitFunction_type_expression(ptr))};
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
        SymbolTableScopeT var_scope(this->symbol_table);
        llvm::SmallVector<mlir::Type> param_types{};
        llvm::SmallVector<mlir::Type, 1> result_types{};

        if (auto *func_type_ptr = ast_node->function_type_expr())
        {
            for (auto *func_param_ptr : func_type_ptr->function_parameter_list()->function_parameter())
            {
                if (auto type = this->visitType_expr(func_param_ptr->type_expr()))
                {
                    param_types.push_back(type);
                }
            }
            if (auto *func_ret_ptr = func_type_ptr->type_expr())
            {
                if (auto ret_type = this->visitType_expr(func_ret_ptr))
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

        if (not func_type.getInputs().empty())
        {
            for (auto const func_arg :
                 llvm::zip(entry_block.getArguments(), ast_node->function_type_expr()->function_parameter_list()->function_parameter()))
            {
                auto block_arg = std::get<0>(func_arg);
                auto arg_type = block_arg.getType();
                auto arg_name = std::get<1>(func_arg)->Identifier()->getText();
                auto ptr_type = mlir::atemhir::PointerType::get(&this->context, arg_type);

                mlir::DataLayout layout;
                mlir::DataLayoutEntryList entries;
                mlir::IntegerAttr alignment = nullptr;

                if (auto data_layout = mlir::dyn_cast<mlir::DataLayoutTypeInterface>(arg_type); data_layout)
                {
                    alignment =
                        mlir::IntegerAttr::get(mlir::IntegerType::get(&this->context, 64), data_layout.getPreferredAlignment(layout, entries));
                }

                auto var_op = this->builder.create<mlir::atemhir::AllocateVarOp>(this->generateMLIRLocationFromASTLocation(ast_node), ptr_type,
                                                                                 arg_type, arg_name, alignment);
                this->builder.create<mlir::atemhir::StoreOp>(
                    this->generateMLIRLocationFromASTLocation(ast_node), mlir::Value{block_arg}, var_op.getResult(), nullptr, alignment,
                    mlir::atemhir::MemoryOrderAttr::get(&this->context, mlir::atemhir::MemoryOrder::SequentiallyConsistent));

                if (mlir::failed(this->declareVariable(arg_name, var_op.getResult())))
                {
                    this->emitError(std::get<1>(func_arg), "function parameter redeclaration");
                }
            }
        }

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
                    func.erase();
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
                func.erase();
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
                    func.erase();
                    return mlir::atemhir::FunctionOp{nullptr};
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

        return func;
    }

    auto visitVariable_decl(AtemParser::Variable_declContext *ast_node) -> std::any override
    {
        auto var_name = ast_node->Identifier()->getText();
        auto var_init_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr()));
        auto var_type = this->visitType_expr(ast_node->type_expr());

        if (not var_init_expr)
        {
            this->emitError(ast_node, "variable must be initialized on declaration");
            return mlir::Value{nullptr};
        }

        auto var_ptr_type = mlir::atemhir::PointerType::get(&this->context, var_type);

        mlir::DataLayout layout;
        mlir::DataLayoutEntryList entries;
        mlir::IntegerAttr alignment = nullptr;

        if (auto data_layout = mlir::dyn_cast<mlir::DataLayoutTypeInterface>(var_type); data_layout)
        {
            alignment = mlir::IntegerAttr::get(mlir::IntegerType::get(&this->context, 64), data_layout.getPreferredAlignment(layout, entries));
        }

        auto var_op = this->builder.create<mlir::atemhir::AllocateVarOp>(this->generateMLIRLocationFromASTLocation(ast_node), var_ptr_type, var_type,
                                                                         var_name, alignment);
        if (var_type != var_init_expr.getType())
        {
            if (auto init_expr_def_op = var_init_expr.getDefiningOp(); mlir::isa<mlir::atemhir::UndefinedOp>(init_expr_def_op))
            {
                auto undef_op = this->builder.create<mlir::atemhir::UndefinedOp>(this->generateMLIRLocationFromASTLocation(ast_node), var_type);
                var_init_expr.replaceAllUsesWith(undef_op.getRes());
                init_expr_def_op->erase();
            }
            else if (mlir::isa<mlir::atemhir::ZeroInitOp>(init_expr_def_op))
            {
                auto zeroinit_op = this->builder.create<mlir::atemhir::ZeroInitOp>(this->generateMLIRLocationFromASTLocation(ast_node), var_type);
                var_init_expr.replaceAllUsesWith(zeroinit_op.getRes());
                init_expr_def_op->erase();
            }
            else
            {
                auto casted_init_expr = this->tryImplicitCast(ast_node, var_init_expr, var_type);
                if (not casted_init_expr)
                {
                    this->emitError(ast_node, "cannot initialize variable {} of type '{}' with a value {} of type '{}'", var_name, toString(var_type),
                                    ast_node->expr()->getText(), toString(var_init_expr.getType()));
                    var_op.erase();
                    return mlir::Value{nullptr};
                }
                this->builder.create<mlir::atemhir::StoreOp>(
                    this->generateMLIRLocationFromASTLocation(ast_node), casted_init_expr, var_op.getResult(), nullptr, alignment,
                    mlir::atemhir::MemoryOrderAttr::get(&this->context, mlir::atemhir::MemoryOrder::SequentiallyConsistent));
            }
        }
        else
        {
            this->builder.create<mlir::atemhir::StoreOp>(
                this->generateMLIRLocationFromASTLocation(ast_node), var_init_expr, var_op.getResult(), nullptr, alignment,
                mlir::atemhir::MemoryOrderAttr::get(&this->context, mlir::atemhir::MemoryOrder::SequentiallyConsistent));
        }

        if (mlir::failed(this->declareVariable(var_name, var_op.getResult(), ast_node)))
        {
            this->emitError(ast_node, "variable {} redeclaration", var_name);
            return mlir::Value{nullptr};
        }
        return var_op.getResult();
    }

    auto visitIdentifier_expression(AtemParser::Identifier_expressionContext *ast_node) -> std::any override
    {
        auto identifier = ast_node->Identifier()->getText();

        if (this->function_map.contains(identifier))
        {
            auto func_type = this->function_map[identifier];
            //::mlir::Type res, ::mlir::TypedAttr value
            auto const_op = this->builder.create<mlir::atemhir::ConstantOp>(
                this->generateMLIRLocationFromASTLocation(ast_node), func_type,
                mlir::atemhir::FunctionAttr::get(&this->context, func_type,
                                                 mlir::SymbolRefAttr::get(mlir::StringAttr::get(&this->context, identifier))));
            return mlir::Value{const_op.getResult()};
        }

        if (this->symbol_table.count(identifier) == 0)
        {
            this->emitError(ast_node, "variable {} does not exist", identifier);
            return mlir::Value{nullptr};
        }

        auto var_addr = this->symbol_table.lookup(identifier).var_ssa_value;
        auto var_type = mlir::dyn_cast<mlir::atemhir::PointerType>(var_addr.getType()).getPointeeType();

        mlir::DataLayout layout;
        mlir::DataLayoutEntryList entries;
        mlir::IntegerAttr alignment = nullptr;

        if (auto data_layout = mlir::dyn_cast<mlir::DataLayoutTypeInterface>(var_type); data_layout)
        {
            alignment = mlir::IntegerAttr::get(mlir::IntegerType::get(&this->context, 64), data_layout.getPreferredAlignment(layout, entries));
        }

        auto memory_order = mlir::atemhir::MemoryOrderAttr::get(&this->context, mlir::atemhir::MemoryOrder::SequentiallyConsistent);

        //::mlir::Type result, ::mlir::Value addr, /*optional*/bool is_deref, /*optional*/bool is_volatile, /*optional*/::mlir::IntegerAttr alignment,
        //:/*optional*/::mlir::atemhir::MemoryOrderAttr memory_order
        auto load_op = this->builder.create<mlir::atemhir::LoadOp>(this->generateMLIRLocationFromASTLocation(ast_node), var_type, var_addr, false,
                                                                   false, alignment, memory_order);
        return mlir::Value{load_op.getResult()};
    }

    auto visitPrefix_expression(AtemParser::Prefix_expressionContext *ast_node) -> std::any override
    {
        auto prev_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr()));
        for (auto *prefix_op : std::ranges::views::reverse(ast_node->prefix_operator()))
        {
            if (auto try_expr = this->visitSinglePrefix_expression(prefix_op, prev_expr); try_expr)
            {
                prev_expr = try_expr;
            }
        }
        return prev_expr;
    }

    auto visitSinglePrefix_expression(AtemParser::Prefix_operatorContext *ast_node, mlir::Value operand) -> mlir::Value
    {
        auto op_type = operand.getType();
        auto loc = this->generateMLIRLocationFromASTLocation(ast_node);
        if (ast_node->Minus())
        {
            if (mlir::isa<mlir::atemhir::IntType>(op_type) or mlir::isa<mlir::atemhir::AtemHIRFPTypeInterface>(op_type))
            {
                //::mlir::Type result, ::mlir::atemhir::UnaryOpKind kind, ::mlir::Value input
                auto neg_op = this->builder.create<mlir::atemhir::UnaryOp>(loc, op_type, mlir::atemhir::UnaryOpKind::Neg, operand);
                return neg_op.getResult();
            }

            this->emitError(ast_node, "invalid argument type '{}' to unary expression '-'", toString(op_type));
            return {nullptr};
        }
        if (ast_node->KeywordNot())
        {
            if (mlir::isa<mlir::atemhir::BoolType>(op_type))
            {
                //::mlir::Type result, ::mlir::atemhir::UnaryOpKind kind, ::mlir::Value input
                auto neg_op = this->builder.create<mlir::atemhir::UnaryOp>(loc, op_type, mlir::atemhir::UnaryOpKind::Not, operand);
                return neg_op.getResult();
            }

            this->emitError(ast_node, "invalid argument type '{}' to unary expression 'not'", toString(op_type));
            return {nullptr};
        }
        if (ast_node->BitNot())
        {
            //::mlir::Type result, ::mlir::atemhir::UnaryOpKind kind, ::mlir::Value input
            auto bit_not_op = this->builder.create<mlir::atemhir::UnaryOp>(loc, op_type, mlir::atemhir::UnaryOpKind::BitNot, operand);
            return bit_not_op.getResult();
        }
        this->unreachable(ast_node, "unknown prefix operator in AST");
    }

    auto visitPostfix_expression(AtemParser::Postfix_expressionContext *ast_node) -> std::any override
    {
        auto prev_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr()));

        for (auto *postfix_op : ast_node->postfix_operator())
        {
            if (auto try_expr = this->visitSinglePostfix_expression(postfix_op, prev_expr))
            {
                prev_expr = try_expr;
            }
        }
        return prev_expr;
    }

    auto visitSinglePostfix_expression(AtemParser::Postfix_operatorContext *ast_node, mlir::Value operand) -> mlir::Value
    {
        if (auto *func_call_op = ast_node->function_call_operator())
        {
            if (auto func_attr_def_op = mlir::dyn_cast<mlir::atemhir::ConstantOp>(operand.getDefiningOp()))
            {
                auto func_type = mlir::dyn_cast<mlir::atemhir::FunctionType>(operand.getType());
                auto func_attr = mlir::dyn_cast<mlir::atemhir::FunctionAttr>(func_attr_def_op.getValue());
                std::string func_name = mlir::dyn_cast<mlir::SymbolRefAttr>(func_attr.getValue()).getLeafReference().getValue().data();
                func_attr_def_op->erase();

                if (not this->function_map.contains(func_name))
                {
                    this->emitError(ast_node, "use of undeclared function '{}'", func_name);
                    return mlir::Value{nullptr};
                }

                auto target_func_type = this->function_map[func_name];
                llvm::SmallVector<mlir::Value, 8> provided_args;
                auto target_arg_count = target_func_type.getInputs().size();
                auto provided_arg_count = ast_node->function_call_operator()->expr().size();

                if (target_arg_count != provided_arg_count)
                {
                    this->emitError(ast_node, "function '{}' needs {} arguments, but {} were provided", func_name, target_arg_count,
                                    provided_arg_count);
                    return mlir::Value{nullptr};
                }

                for (auto i = 0; i < provided_arg_count; ++i)
                {
                    auto *expr_ptr = ast_node->function_call_operator()->expr()[i];
                    auto arg = std::any_cast<mlir::Value>(this->visit(expr_ptr));
                    if (not arg)
                    {
                        this->emitError(ast_node, "failed to generate Atem HIR for the {}th argument", i + 1);
                        return mlir::Value{nullptr};
                    }

                    auto target_arg_type = target_func_type.getInputs()[i];
                    if (arg.getType() == target_arg_type)
                    {
                        provided_args.push_back(arg);
                    }
                    else
                    {
                        auto casted_expr = this->tryImplicitCast(expr_ptr, arg, target_arg_type);
                        if (not casted_expr)
                        {
                            this->emitError(expr_ptr, "no known conversion from '{}' to '{}' for {}th argument", toString(arg.getType()),
                                            toString(target_arg_type), i + 1);
                            return mlir::Value{nullptr};
                        }
                        provided_args.push_back(casted_expr);
                    }
                }

                auto func_name_sym = mlir::SymbolRefAttr::get(mlir::StringAttr::get(&this->context, func_name.data()));
                //::mlir::Type result, ::mlir::SymbolRefAttr callee, ::mlir::ValueRange arg_operands
                auto call_op = this->builder.create<mlir::atemhir::CallOp>(this->generateMLIRLocationFromASTLocation(ast_node),
                                                                           target_func_type.getResults().front(), func_name_sym, provided_args);
                return mlir::Value{call_op.getResult()};
            }
            this->emitError(ast_node, "the lhs of function call operator must be a name of a function");
            return mlir::Value{nullptr};
        }
        if (ast_node->PointerDeref())
        {
            this->emitError(ast_node, "pointer dereference are not yet supported");
            return {nullptr};
        }
        if (ast_node->ObjectAddress())
        {
            this->emitError(ast_node, "address taking operation are not yet supported");
            return {nullptr};
        }
        if (auto *int_ptr = ast_node->integer_literal())
        {
            this->emitError(ast_node, "array subscripts are not yet supported");
            return {nullptr};
        }
        if (auto *member_access_op = ast_node->member_access_operator())
        {
            this->emitError(ast_node, "member access operation is not yet supported");
            return {nullptr};
        }
        this->unreachable(ast_node, "unknown postfix operator in AST");
    }

    auto visitMultiplicative_expression(AtemParser::Multiplicative_expressionContext *ast_node) -> std::any override
    {
        using enum mlir::atemhir::BinaryOpKind;
        mlir::atemhir::BinaryOpKind op_kind;

        if (ast_node->mul_operator()->Mul())
        {
            op_kind = Mul;
        }
        else if (ast_node->mul_operator()->Divide())
        {
            op_kind = Div;
        }
        else if (ast_node->mul_operator()->Remainder())
        {
            op_kind = Rem;
        }
        else
        {
            this->unreachable(ast_node, "unknown multiplicative expression in AST");
        }

        auto lhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().front()));
        auto rhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().back()));
        auto lhs_type = lhs_expr.getType();
        auto rhs_type = rhs_expr.getType();

        auto common_type = lhs_type;

        if (lhs_type != rhs_type)
        {
            if (auto common_type_opt = this->resolveCommonType(lhs_type, rhs_type); common_type_opt.has_value())
            {
                common_type = common_type_opt.value();
                lhs_expr = this->tryImplicitCast(ast_node->expr().front(), lhs_expr, common_type);
                rhs_expr = this->tryImplicitCast(ast_node->expr().back(), rhs_expr, common_type);
            }
            else
            {
                this->emitError(ast_node, "cannot resolve common type for lhs '{}' and rhs '{}'", toString(lhs_type), toString(rhs_type));
                return mlir::Value{nullptr};
            }
        }

        if (not mlir::isa<mlir::atemhir::IntType>(common_type) and not mlir::isa<mlir::atemhir::AtemHIRFPTypeInterface>(common_type))
        {
            this->emitError(ast_node, "invalid operand types for binary multiplicative expression ('{}' and '{}')", toString(lhs_type),
                            toString(rhs_type));
            return mlir::Value{nullptr};
        }

        //::mlir::Type result, ::mlir::atemhir::BinaryOpKind kind, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/bool no_unsigned_wrap = false,
        //:/*optional*/bool no_signed_wrap = false
        auto bin_op = this->builder.create<mlir::atemhir::BinaryOp>(this->generateMLIRLocationFromASTLocation(ast_node), common_type, op_kind,
                                                                    lhs_expr, rhs_expr);
        return mlir::Value{bin_op.getResult()};
    }

    auto visitAdditive_expression(AtemParser::Additive_expressionContext *ast_node) -> std::any override
    {
        using enum mlir::atemhir::BinaryOpKind;
        mlir::atemhir::BinaryOpKind op_kind;

        if (ast_node->add_operator()->Add())
        {
            op_kind = Add;
        }
        else if (ast_node->add_operator()->Minus())
        {
            op_kind = Sub;
        }
        else
        {
            this->unreachable(ast_node, "unknown additive expression in AST");
        }

        auto lhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().front()));
        auto rhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().back()));
        auto lhs_type = lhs_expr.getType();
        auto rhs_type = rhs_expr.getType();

        auto common_type = lhs_type;

        if (lhs_type != rhs_type)
        {
            if (auto common_type_opt = this->resolveCommonType(lhs_type, rhs_type); common_type_opt.has_value())
            {
                common_type = common_type_opt.value();
                lhs_expr = this->tryImplicitCast(ast_node->expr().front(), lhs_expr, common_type);
                rhs_expr = this->tryImplicitCast(ast_node->expr().back(), rhs_expr, common_type);
            }
            else
            {
                this->emitError(ast_node, "cannot resolve common type for lhs '{}' and rhs '{}'", toString(lhs_type), toString(rhs_type));
                return mlir::Value{nullptr};
            }
        }

        if (not mlir::isa<mlir::atemhir::IntType>(common_type) and not mlir::isa<mlir::atemhir::AtemHIRFPTypeInterface>(common_type))
        {
            this->emitError(ast_node, "invalid operand types for binary additive expression ('{}' and '{}')", toString(lhs_type), toString(rhs_type));
            return mlir::Value{nullptr};
        }

        //::mlir::Type result, ::mlir::atemhir::BinaryOpKind kind, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/bool no_unsigned_wrap = false,
        //:/*optional*/bool no_signed_wrap = false
        auto bin_op = this->builder.create<mlir::atemhir::BinaryOp>(this->generateMLIRLocationFromASTLocation(ast_node), common_type, op_kind,
                                                                    lhs_expr, rhs_expr);
        return mlir::Value{bin_op.getResult()};
    }
    auto visitBitshift_expression(AtemParser::Bitshift_expressionContext *ast_node) -> std::any override
    {
        bool is_shifting_left = true;
        if (ast_node->bitshift_operator()->BitRightShift())
        {
            is_shifting_left = false;
        }

        auto lhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().front()));
        auto rhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().back()));
        auto lhs_type = lhs_expr.getType();
        auto rhs_type = rhs_expr.getType();

        auto common_type = lhs_type;

        if (lhs_type != rhs_type)
        {
            if (auto common_type_opt = this->resolveCommonType(lhs_type, rhs_type); common_type_opt.has_value())
            {
                common_type = common_type_opt.value();
                lhs_expr = this->tryImplicitCast(ast_node->expr().front(), lhs_expr, common_type);
                rhs_expr = this->tryImplicitCast(ast_node->expr().back(), rhs_expr, common_type);
            }
            else
            {
                this->emitError(ast_node, "cannot resolve common type for lhs '{}' and rhs '{}'", toString(lhs_type), toString(rhs_type));
                return mlir::Value{nullptr};
            }
        }

        if (not mlir::isa<mlir::atemhir::IntType>(common_type))
        {
            this->emitError(ast_node, "invalid operand types for bit-shifting expression ('{}' and '{}')", toString(lhs_type), toString(rhs_type));
            return mlir::Value{nullptr};
        }

        //::mlir::Type result, ::mlir::Value value, ::mlir::Value amount, /*optional*/bool is_shifting_left = false
        auto bin_op = this->builder.create<mlir::atemhir::ShiftOp>(this->generateMLIRLocationFromASTLocation(ast_node), common_type, lhs_expr,
                                                                   rhs_expr, is_shifting_left);
        return mlir::Value{bin_op.getResult()};
    }

    auto visitBitwise_expression(AtemParser::Bitwise_expressionContext *ast_node) -> std::any override
    {
        using enum mlir::atemhir::BinaryOpKind;
        mlir::atemhir::BinaryOpKind op_kind;

        if (ast_node->bitwise_operator()->BitAnd())
        {
            op_kind = BitAnd;
        }
        else if (ast_node->bitwise_operator()->BitOr())
        {
            op_kind = BitOr;
        }
        else if (ast_node->bitwise_operator()->BitXor())
        {
            op_kind = BitXor;
        }
        else
        {
            this->unreachable(ast_node, "unknown bitwise expression in AST");
        }

        auto lhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().front()));
        auto rhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().back()));
        auto lhs_type = lhs_expr.getType();
        auto rhs_type = rhs_expr.getType();

        auto common_type = lhs_type;

        if (lhs_type != rhs_type)
        {
            if (auto common_type_opt = this->resolveCommonType(lhs_type, rhs_type); common_type_opt.has_value())
            {
                common_type = common_type_opt.value();
                lhs_expr = this->tryImplicitCast(ast_node->expr().front(), lhs_expr, common_type);
                rhs_expr = this->tryImplicitCast(ast_node->expr().back(), rhs_expr, common_type);
            }
            else
            {
                this->emitError(ast_node, "cannot resolve common type for lhs '{}' and rhs '{}'", toString(lhs_type), toString(rhs_type));
                return mlir::Value{nullptr};
            }
        }

        if (not mlir::isa<mlir::atemhir::IntType>(common_type))
        {
            this->emitError(ast_node, "invalid operand types for binary bitwise expression ('{}' and '{}')", toString(lhs_type), toString(rhs_type));
            return mlir::Value{nullptr};
        }

        //::mlir::Type result, ::mlir::atemhir::BinaryOpKind kind, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/bool no_unsigned_wrap = false,
        //:/*optional*/bool no_signed_wrap = false
        auto bin_op = this->builder.create<mlir::atemhir::BinaryOp>(this->generateMLIRLocationFromASTLocation(ast_node), common_type, op_kind,
                                                                    lhs_expr, rhs_expr);
        return mlir::Value{bin_op.getResult()};
    }

    auto visitComparison_expression(AtemParser::Comparison_expressionContext *ast_node) -> std::any override
    {
        using enum mlir::atemhir::CompareOpKind;

        mlir::atemhir::CompareOpKind op_kind;

        if (ast_node->comparison_operator()->Equal())
        {
            op_kind = eq;
        }
        else if (ast_node->comparison_operator()->NotEqual())
        {
            op_kind = ne;
        }
        else if (ast_node->comparison_operator()->GreaterThan())
        {
            op_kind = gt;
        }
        else if (ast_node->comparison_operator()->GreaterThanOrEqual())
        {
            op_kind = ge;
        }
        else if (ast_node->comparison_operator()->LessThan())
        {
            op_kind = lt;
        }
        else if (ast_node->comparison_operator()->LessThanOrEqual())
        {
            op_kind = le;
        }
        else
        {
            this->unreachable(ast_node, "unknown comparison operator in AST");
        }

        auto lhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().front()));
        auto rhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().back()));
        auto lhs_type = lhs_expr.getType();
        auto rhs_type = rhs_expr.getType();

        auto common_type = lhs_type;
        auto bool_result_type = mlir::atemhir::BoolType::get(&this->context);

        if (lhs_type != rhs_type)
        {
            if (auto common_type_opt = this->resolveCommonType(lhs_type, rhs_type); common_type_opt.has_value())
            {
                common_type = common_type_opt.value();
                lhs_expr = this->tryImplicitCast(ast_node->expr().front(), lhs_expr, common_type);
                rhs_expr = this->tryImplicitCast(ast_node->expr().back(), rhs_expr, common_type);
            }
            else
            {
                this->emitError(ast_node, "cannot resolve common type for lhs '{}' and rhs '{}'", toString(lhs_type), toString(rhs_type));
                return mlir::Value{nullptr};
            }
        }

        if (not mlir::isa<mlir::atemhir::IntType>(common_type) and not mlir::isa<mlir::atemhir::AtemHIRFPTypeInterface>(common_type))
        {
            this->emitError(ast_node, "invalid operand types for binary comparison expression ('{}' and '{}')", toString(lhs_type),
                            toString(rhs_type));
            return mlir::Value{nullptr};
        }

        //::mlir::Type resultType0, ::mlir::atemhir::CompareOpKindAttr kind, ::mlir::Value lhs, ::mlir::Value rhs
        auto comp_op = this->builder.create<mlir::atemhir::CompareOp>(this->generateMLIRLocationFromASTLocation(ast_node), bool_result_type, op_kind,
                                                                      lhs_expr, rhs_expr);
        return mlir::Value{comp_op.getResult()};
    }

    auto visitBinary_logical_expression(AtemParser::Binary_logical_expressionContext *ast_node) -> std::any override
    {
        using enum mlir::atemhir::BinaryOpKind;
        mlir::atemhir::BinaryOpKind op_kind;

        if (ast_node->binary_logical_operator()->KeywordAnd())
        {
            op_kind = And;
        }
        else if (ast_node->binary_logical_operator()->KeywordOr())
        {
            op_kind = Or;
        }
        else
        {
            this->unreachable(ast_node, "unknown binary logical expression in AST");
        }

        auto lhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().front()));
        auto rhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().back()));
        auto lhs_type = lhs_expr.getType();
        auto rhs_type = rhs_expr.getType();

        auto bool_result_type = mlir::atemhir::BoolType::get(&this->context);

        if (not mlir::isa<mlir::atemhir::BoolType>(lhs_type) or not mlir::isa<mlir::atemhir::BoolType>(rhs_type))
        {
            this->emitError(ast_node, "invalid operand types for binary logical expression ('{}' and '{}')", toString(lhs_type), toString(rhs_type));
            return mlir::Value{nullptr};
        }

        //::mlir::Type result, ::mlir::atemhir::BinaryOpKind kind, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/bool no_unsigned_wrap = false,
        //:/*optional*/bool no_signed_wrap = false
        auto bin_op = this->builder.create<mlir::atemhir::BinaryOp>(this->generateMLIRLocationFromASTLocation(ast_node), bool_result_type, op_kind,
                                                                    lhs_expr, rhs_expr);
        return mlir::Value{bin_op.getResult()};
    }

    auto visitAssignment_expression(AtemParser::Assignment_expressionContext *ast_node) -> std::any override
    {
        auto *var_identifier_ast = dynamic_cast<AtemParser::Identifier_expressionContext *>(ast_node->expr().front());
        if (not var_identifier_ast)
        {
            this->emitError(ast_node, "the left side of assignment expression must be a variable name");
            return mlir::Value{nullptr};
        }

        auto var_addr = this->symbol_table.lookup(var_identifier_ast->Identifier()->getText()).var_ssa_value;
        auto rhs_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr().back()));

        auto var_ptr_type = mlir::dyn_cast<mlir::atemhir::PointerType>(var_addr.getType());
        auto var_type = var_ptr_type.getPointeeType();
        auto rhs_type = rhs_expr.getType();

        auto loc = this->generateMLIRLocationFromASTLocation(ast_node);

        mlir::DataLayout layout;
        mlir::DataLayoutEntryList entries;
        mlir::IntegerAttr alignment = nullptr;

        if (auto data_layout = mlir::dyn_cast<mlir::DataLayoutTypeInterface>(var_type); data_layout)
        {
            alignment = mlir::IntegerAttr::get(mlir::IntegerType::get(&this->context, 64), data_layout.getPreferredAlignment(layout, entries));
        }

        auto memory_order_attr = mlir::atemhir::MemoryOrderAttr::get(&this->context, mlir::atemhir::MemoryOrder::SequentiallyConsistent);

        mlir::Value casted_rhs_expr = rhs_expr;

        if (rhs_type != var_type)
        {
            auto cast_expr = this->tryImplicitCast(ast_node, rhs_expr, var_type);
            if (not cast_expr)
            {
                this->emitError(ast_node, "failed to assign value of type '{}' to variable of type '{}'", toString(rhs_type), toString(var_type));
                return mlir::Value{nullptr};
            }
            casted_rhs_expr = cast_expr;
        }

        if (ast_node->assign_operator()->Assign())
        {
            //::mlir::Value value, ::mlir::Value addr, /*optional*/bool is_volatile, /*optional*/::mlir::IntegerAttr alignment,
            //:/*optional*/::mlir::atemhir::MemoryOrderAttr memory_order
            this->builder.create<mlir::atemhir::StoreOp>(loc, casted_rhs_expr, var_addr, false, alignment, memory_order_attr);
            auto var_load_op = this->builder.create<mlir::atemhir::LoadOp>(loc, var_type, var_addr, false, false, alignment, memory_order_attr);
            return mlir::Value{var_load_op.getResult()};
        }

        mlir::atemhir::BinaryOpKind op_kind = {};
        using enum mlir::atemhir::BinaryOpKind;
        bool is_bitshift = false;
        bool is_shifting_left = false;
        auto *assign_op_ptr = ast_node->assign_operator();

        if (assign_op_ptr->AddAssign())
        {
            op_kind = Add;
        }
        else if (assign_op_ptr->SubAssign())
        {
            op_kind = Sub;
        }
        else if (assign_op_ptr->MulAssign())
        {
            op_kind = Mul;
        }
        else if (assign_op_ptr->DivideAssign())
        {
            op_kind = Div;
        }
        else if (assign_op_ptr->RemainderDivideAssign())
        {
            op_kind = Rem;
        }
        else if (assign_op_ptr->BitAndAssign())
        {
            op_kind = BitAnd;
        }
        else if (assign_op_ptr->BitOrAssign())
        {
            op_kind = BitOr;
        }
        else if (assign_op_ptr->BitXorAssign())
        {
            op_kind = BitXor;
        }
        else if (assign_op_ptr->BitLeftShiftAssign())
        {
            is_shifting_left = true;
            is_bitshift = true;
        }
        else if (assign_op_ptr->BitRightShiftAssign())
        {
            is_bitshift = true;
        }
        else
        {
            this->unreachable(ast_node, "unknown assignment operator in AST");
        }

        //::mlir::Type result, ::mlir::Value addr, /*optional*/bool is_deref, /*optional*/bool is_volatile, /*optional*/::mlir::IntegerAttr alignment,
        //:/*optional*/::mlir::atemhir::MemoryOrderAttr memory_order
        auto var_load_op = this->builder.create<mlir::atemhir::LoadOp>(loc, var_type, var_addr, false, false, alignment, memory_order_attr);
        auto var_content = var_load_op.getResult();
        mlir::Value result_expr{nullptr};

        if (is_bitshift)
        {
            if (not mlir::isa<mlir::atemhir::IntType>(var_type))
            {
                this->emitError(ast_node, "invalid operand types for binary assignment expression ('{}' and '{}')", toString(var_type),
                                toString(rhs_type));
                return mlir::Value{nullptr};
            }

            //::mlir::Type result, ::mlir::Value value, ::mlir::Value amount, /*optional*/bool is_shifting_left = false
            auto bin_op = this->builder.create<mlir::atemhir::ShiftOp>(loc, var_type, var_content, casted_rhs_expr, is_shifting_left);
            result_expr = bin_op.getResult();
        }
        else
        {
            switch (op_kind)
            {
            case Add:
            case Sub:
            case Mul:
            case Div:
            case Rem: {
                if (not mlir::isa<mlir::atemhir::IntType>(var_type) and not mlir::isa<mlir::atemhir::AtemHIRFPTypeInterface>(var_type))
                {
                    this->emitError(ast_node, "invalid operand types for binary assignment expression ('{}' and '{}')", toString(var_type),
                                    toString(rhs_type));
                    return mlir::Value{nullptr};
                }
                break;
            }
            case BitAnd:
            case BitOr:
            case BitXor: {
                if (not mlir::isa<mlir::atemhir::IntType>(var_type))
                {
                    this->emitError(ast_node, "invalid operand types for binary assignment expression ('{}' and '{}')", toString(var_type),
                                    toString(rhs_type));
                    return mlir::Value{nullptr};
                }
                break;
            }
            default: {
                this->unreachable(ast_node, "unknown assignment operator in AST");
            }
            }

            //::mlir::Type result, ::mlir::atemhir::BinaryOpKind kind, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/bool no_unsigned_wrap =
            //: false,
            //:/*optional*/bool no_signed_wrap = false
            auto bin_op = this->builder.create<mlir::atemhir::BinaryOp>(this->generateMLIRLocationFromASTLocation(ast_node), var_type, op_kind,
                                                                        var_content, casted_rhs_expr);
            result_expr = bin_op.getResult();
        }

        //::mlir::Value value, ::mlir::Value addr, /*optional*/bool is_volatile, /*optional*/::mlir::IntegerAttr alignment,
        //:/*optional*/::mlir::atemhir::MemoryOrderAttr memory_order
        this->builder.create<mlir::atemhir::StoreOp>(loc, result_expr, var_addr, false, alignment, memory_order_attr);

        return result_expr;
    }

    auto visitConversion_expression(AtemParser::Conversion_expressionContext *ast_node) -> std::any override
    {
        auto source_expr = std::any_cast<mlir::Value>(this->visit(ast_node->expr()));
        auto source_type = source_expr.getType();
        auto result_type = std::any_cast<mlir::Type>(this->visitType_expr(ast_node->type_expr()));

        auto source_int = mlir::dyn_cast<mlir::atemhir::IntType>(source_type);
        auto result_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(result_type);

        auto source_fp = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(source_type);
        auto result_int = mlir::dyn_cast<mlir::atemhir::IntType>(result_type);

        using enum mlir::atemhir::CastKind;
        if (source_type == result_type)
        {
            return source_expr;
        }
        auto loc = this->generateMLIRLocationFromASTLocation(ast_node);
        mlir::atemhir::CastKind kind;
        if (mlir::isa<mlir::atemhir::BoolType>(source_type) and mlir::isa<mlir::atemhir::IntType>(result_type))
        {
            kind = bool_to_int;
        }
        else if (mlir::isa<mlir::atemhir::IntType>(source_type) and mlir::isa<mlir::atemhir::BoolType>(result_type))
        {
            kind = int_to_bool;
        }
        else if (source_int and result_int)
        {
            if (source_int.isSigned() and result_int.isUnsigned())
            {
                this->emitWarn(ast_node, "narrowing cast from '{}' to '{}'", toString(source_type), toString(result_type));
                kind = int_narrowing;
            }
            else
            {
                if (source_int.getWidth() <= result_int.getWidth())
                {
                    kind = int_promotion;
                }
                else
                {
                    this->emitWarn(ast_node, "narrowing cast from '{}' to '{}'", toString(source_type), toString(result_type));
                    kind = int_narrowing;
                }
            }
        }
        else if (source_fp and result_fp)
        {
            if (source_fp.getWidth() <= result_fp.getWidth())
            {
                kind = float_promotion;
            }
            else
            {
                this->emitWarn(ast_node, "narrowing cast from '{}' to '{}'", toString(source_type), toString(result_type));
                kind = float_narrowing;
            }
        }
        else if (source_int and result_fp)
        {
            llvm::APSInt fp_max;
            bool is_exact;
            llvm::APFloat::getLargest(result_fp.getFloatSemantics())
                .convertToInteger(fp_max, llvm::APFloat::roundingMode::NearestTiesToAway, &is_exact);
            if (llvm::APSInt::getMaxValue(source_int.getWidth(), source_int.isUnsigned()) > fp_max)
            {
                this->emitWarn(ast_node, "narrowing cast from '{}' to '{}'", toString(source_type), toString(result_type));
            }
            kind = int_to_float;
        }
        else if (source_fp and result_int)
        {
            this->emitWarn(ast_node, "narrowing cast from '{}' to '{}'", toString(source_type), toString(result_type));
            kind = float_to_int;
        }
        else
        {
            this->emitError(ast_node, "unsupported cast from '{}' to '{}', if a bitcast is intended, use @bitCast intrinsic instead",
                            toString(source_type), toString(result_type));
            return mlir::Value{nullptr};
        }

        auto cast_op = this->builder.create<mlir::atemhir::CastOp>(loc, result_type, kind, source_expr);
        return mlir::Value{cast_op.getResult()};
    }

    auto visitReturn_expr(AtemParser::Return_exprContext *ast_node) -> std::any override
    {
        if (auto *expr_ptr = ast_node->expr())
        {
            auto return_value = std::any_cast<mlir::Value>(this->visit(expr_ptr));
            this->builder.create<mlir::atemhir::ReturnOp>(this->generateMLIRLocationFromASTLocation(expr_ptr), mlir::ValueRange{return_value});
        }
        else
        {
            auto unit_type = mlir::atemhir::UnitType::get(&this->context);
            auto unit_const = this->builder.create<mlir::atemhir::ConstantOp>(this->generateMLIRLocationFromASTLocation(ast_node), unit_type,
                                                                              mlir::atemhir::UnitAttr::get(&this->context, unit_type));
            this->builder.create<mlir::atemhir::ReturnOp>(this->generateMLIRLocationFromASTLocation(ast_node), mlir::ValueRange{unit_const.getRes()});
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

    auto visitArray_type_expression(AtemParser::Array_type_expressionContext *ast_node) -> std::any override
    {
        auto elem_type = std::any_cast<mlir::Type>(this->visitType_expr(ast_node->type_expr()));
        auto length_attr = this->getIntAttrFromLiteral(ast_node->integer_literal()).getSignedInt();
        return mlir::atemhir::ArrayType::get(&this->context, elem_type, length_attr);
    }

    auto visitPointer_type_expression(AtemParser::Pointer_type_expressionContext *ast_node) -> std::any override
    {
        auto pointee_type = std::any_cast<mlir::Type>(this->visitType_expr(ast_node->type_expr()));
        return mlir::atemhir::PointerType::get(&this->context, pointee_type);
    }

    auto visitFunction_type_expression(AtemParser::Function_type_expressionContext *ast_node) -> std::any override
    {
        return this->visitFunction_type_expr(ast_node->function_type_expr());
    }

    auto visitFunction_type_expr(AtemParser::Function_type_exprContext *ast_node) -> std::any
    {
        llvm::SmallVector<mlir::Type, 8> arg_types;
        llvm::SmallVector<mlir::Type, 1> result_type;

        if (auto *func_param_list_ptr = ast_node->function_parameter_list())
        {
            for (auto *func_param_ptr : func_param_list_ptr->function_parameter())
            {
                arg_types.emplace_back(std::any_cast<mlir::Type>(this->visitType_expr(func_param_ptr->type_expr())));
            }
        }

        if (auto *func_return_ptr = ast_node->type_expr())
        {
            result_type.emplace_back(std::any_cast<mlir::Type>(this->visitType_expr(func_return_ptr)));
        }
        else
        {
            result_type.emplace_back(mlir::atemhir::UnitType::get(&this->context));
        }

        auto func_type = mlir::atemhir::FunctionType::get(&this->context, arg_types, result_type);
        return func_type;
    }
};
} // namespace atem::parser