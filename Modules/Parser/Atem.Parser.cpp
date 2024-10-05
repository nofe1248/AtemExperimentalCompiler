
module;

#include <any>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
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
class AtemHIRGenerator final : public AtemParserBaseVisitor
{
private:
    mlir::MLIRContext &context;
    mlir::ModuleOp hir_module;
    mlir::OpBuilder builder;

    std::string file_name;

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

    auto getDiagnosticError(antlr4::ParserRuleContext const *ast_node, llvm::StringRef info = "") -> mlir::Diagnostic
    {
        return mlir::Diagnostic(this->generateMLIRLocationFromASTLocation(ast_node), mlir::DiagnosticSeverity::Error);
    }

    auto getDiagnosticWarn(antlr4::ParserRuleContext const *ast_node, llvm::StringRef info = "") -> mlir::Diagnostic
    {
        return mlir::Diagnostic(this->generateMLIRLocationFromASTLocation(ast_node), mlir::DiagnosticSeverity::Warning);
    }

    auto getDiagnosticRemark(antlr4::ParserRuleContext const *ast_node, llvm::StringRef info = "") -> mlir::Diagnostic
    {
        return mlir::Diagnostic(this->generateMLIRLocationFromASTLocation(ast_node), mlir::DiagnosticSeverity::Remark);
    }

    auto getDiagnosticNote(antlr4::ParserRuleContext const *ast_node, llvm::StringRef info = "") -> mlir::Diagnostic
    {
        return mlir::Diagnostic(this->generateMLIRLocationFromASTLocation(ast_node), mlir::DiagnosticSeverity::Note);
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
                            << "Function declaration already exists : " << func_op.getSymName();
                    }
                    else
                    {
                        this->function_map.insert({func_op.getSymName(), func_op});
                    }
                }
            }
            else if (auto *var_ptr = decl_ptr->variable_decl())
            {
                mlir::emitError(this->generateMLIRLocationFromASTLocation(var_ptr)) << "Global variable declarations are not supported yet";
            }
            else if (auto *const_ptr = decl_ptr->constant_decl())
            {
                mlir::emitError(this->generateMLIRLocationFromASTLocation(const_ptr)) << "Global constant declarations are not supported yet";
            }
            else if (auto *struct_ptr = decl_ptr->struct_decl())
            {
                mlir::emitError(this->generateMLIRLocationFromASTLocation(struct_ptr)) << "Structs are not supported yet";
            }
            else
            {
                this->unreachable(decl_ptr, "unknown declaration type in AST");
            }
        }

        if (mlir::verify(this->hir_module).failed())
        {
            return nullptr;
        }

        return this->hir_module;
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
            mlir::emitError(this->generateMLIRLocationFromASTLocation(ptr)) << "User-defined types are not supported yet";
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
                mlir::emitError(this->generateMLIRLocationFromASTLocation(ast_node))
                    << "Function " << func_name << " returns non-Unit, thus the function must contains a return expression on the end";
                return mlir::atemhir::FunctionOp{nullptr};
            }
        }

        if (func_name != "main")
        {
            func.setPrivate();
        }

        this->builder.setInsertionPointToEnd(this->hir_module.getBody());

        return func;
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
            return mlir::atemhir::FP16Type::get(&this->context);
        }
        if (ast_node->getText() == "Float32")
        {
            return mlir::atemhir::FP32Type::get(&this->context);
        }
        if (ast_node->getText() == "Float64")
        {
            return mlir::atemhir::FP64Type::get(&this->context);
        }
        if (ast_node->getText() == "Float80")
        {
            return mlir::atemhir::FP80Type::get(&this->context);
        }
        if (ast_node->getText() == "Float128")
        {
            return mlir::atemhir::FP128Type::get(&this->context);
        }
        this->unreachable(ast_node, "unknown floating-point type in AST");
    }

    auto visitString_type_expression(AtemParser::String_type_expressionContext *ast_node) -> std::any override
    {
        mlir::emitError(this->generateMLIRLocationFromASTLocation(ast_node)) << "String types are not supported yet";
        return mlir::atemhir::StringType::get(&this->context);
    }

    auto visitRune_type_expression(AtemParser::Rune_type_expressionContext *ast_node) -> std::any override
    {
        mlir::emitError(this->generateMLIRLocationFromASTLocation(ast_node)) << "Rune types are not supported yet";
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