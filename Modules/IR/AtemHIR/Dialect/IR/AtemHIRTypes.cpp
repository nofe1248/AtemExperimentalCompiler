#include <format>

#include "IR/AtemHIR/Dialect/IR/AtemHIRTypes.hpp"
#include "IR/AtemHIR/Dialect/IR/AtemHIRDialect.hpp"

#include "mlir/Support/LLVM.h"

#define GET_TYPEDEF_CLASSES
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIRTypes.cpp.inc"

#include "mlir/IR/Builders.h"

//========================================================
// Dialect Type Initialization
//========================================================

auto mlir::atemhir::AtemHIRDialect::registerTypes() -> void
{
    this->addTypes<
#define GET_TYPEDEF_LIST
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIRTypes.cpp.inc"

        >();
}

//========================================================
// Dialect Type Parser & Printer
//========================================================

auto mlir::atemhir::AtemHIRDialect::parseType(DialectAsmParser &parser) const -> Type
{
    SMLoc type_loc = parser.getCurrentLocation();
    StringRef mnemonic;
    Type gen_type;

    OptionalParseResult result = generatedTypeParser(parser, &mnemonic, gen_type);
    if (result.has_value())
    {
        return gen_type;
    }

    return StringSwitch<function_ref<Type()>>(mnemonic).Default([&] {
        parser.emitError(type_loc) << "unknown Atem HIR type: " << mnemonic;
        return Type();
    })();
}
auto mlir::atemhir::AtemHIRDialect::printType(Type type, DialectAsmPrinter &printer) const -> void
{
    if (generatedTypePrinter(type, printer).succeeded())
    {
        return;
    }

    TypeSwitch<Type>(type).Default([](Type type) { llvm::report_fatal_error("printer is missing a handler for this type"); });
}

//========================================================
// Boolean Type Definitions
//========================================================

auto mlir::atemhir::BoolType::getTypeSizeInBits(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return llvm::TypeSize::getFixed(8);
}

auto mlir::atemhir::BoolType::getABIAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return 1;
}

auto mlir::atemhir::BoolType::getPreferredAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return 1;
}

auto mlir::atemhir::BoolType::toAtemTypeString() const -> std::string
{
    return "Bool";
}

//========================================================
// Integer Type Definitions
//========================================================

auto mlir::atemhir::IntType::toAtemTypeString() const -> std::string
{
    if (this->isSigned())
    {
        return std::string{"Int"}.append(std::to_string(this->getWidth()));
    }
    return std::string{"UInt"}.append(std::to_string(this->getWidth()));
}

auto mlir::atemhir::IntType::parse(AsmParser &parser) -> Type
{
    auto *context = parser.getBuilder().getContext();
    auto loc = parser.getCurrentLocation();
    bool is_signed;
    unsigned width;

    if (parser.parseLess())
    {
        return {};
    }

    StringRef sign;
    if (parser.parseKeyword(&sign))
    {
        return {};
    }
    if (sign == "s")
    {
        is_signed = true;
    }
    else if (sign == "u")
    {
        is_signed = false;
    }
    else
    {
        parser.emitError(loc, "Expected 's' or 'u'");
        return {};
    }

    if (parser.parseComma())
    {
        return {};
    }

    if (parser.parseInteger(width))
    {
        return {};
    }

    if (parser.parseGreater())
    {
        return {};
    }

    return IntType::get(context, width, is_signed);
}

auto mlir::atemhir::IntType::print(AsmPrinter &printer) const -> void
{
    printer << '<' << (this->isSigned() ? 's' : 'u') << ", " << this->getWidth() << '>';
}

auto mlir::atemhir::IntType::verify(function_ref<InFlightDiagnostic()> emit_error, unsigned width, bool is_signed) -> LogicalResult
{
    return success();
}

auto mlir::atemhir::IntType::getTypeSizeInBits(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return llvm::TypeSize::getFixed(this->getWidth());
}

auto mlir::atemhir::IntType::getABIAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::IntType::getPreferredAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

//========================================================
// Floating-point Type Definitions
//========================================================

auto mlir::atemhir::FP16Type::toAtemTypeString() const -> std::string
{
    return "Float16";
}

auto mlir::atemhir::FP16Type::getFloatSemantics() const -> llvm::fltSemantics const &
{
    return APFloat::IEEEhalf();
}

auto mlir::atemhir::FP16Type::getTypeSizeInBits(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return llvm::TypeSize::getFixed(this->getWidth());
}

auto mlir::atemhir::FP16Type::getABIAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::FP16Type::getPreferredAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::FP32Type::toAtemTypeString() const -> std::string
{
    return "Float32";
}

auto mlir::atemhir::FP32Type::getFloatSemantics() const -> llvm::fltSemantics const &
{
    return APFloat::IEEEsingle();
}

auto mlir::atemhir::FP32Type::getTypeSizeInBits(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return llvm::TypeSize::getFixed(this->getWidth());
}

auto mlir::atemhir::FP32Type::getABIAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::FP32Type::getPreferredAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::FP64Type::toAtemTypeString() const -> std::string
{
    return "Float64";
}

auto mlir::atemhir::FP64Type::getFloatSemantics() const -> llvm::fltSemantics const &
{
    return APFloat::IEEEdouble();
}

auto mlir::atemhir::FP64Type::getTypeSizeInBits(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return llvm::TypeSize::getFixed(this->getWidth());
}

auto mlir::atemhir::FP64Type::getABIAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::FP64Type::getPreferredAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::FP80Type::toAtemTypeString() const -> std::string
{
    return "Float80";
}

auto mlir::atemhir::FP80Type::getFloatSemantics() const -> llvm::fltSemantics const &
{
    return APFloat::x87DoubleExtended();
}

auto mlir::atemhir::FP80Type::getTypeSizeInBits(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return llvm::TypeSize::getFixed(this->getWidth());
}

auto mlir::atemhir::FP80Type::getABIAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::FP80Type::getPreferredAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::FP128Type::toAtemTypeString() const -> std::string
{
    return "Float128";
}

auto mlir::atemhir::FP128Type::getFloatSemantics() const -> llvm::fltSemantics const &
{
    return APFloat::IEEEquad();
}

auto mlir::atemhir::FP128Type::getTypeSizeInBits(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return llvm::TypeSize::getFixed(this->getWidth());
}

auto mlir::atemhir::FP128Type::getABIAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

auto mlir::atemhir::FP128Type::getPreferredAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return static_cast<uint64_t>(this->getWidth() / 8);
}

//========================================================
// Pointer Type Definitions
//========================================================

auto mlir::atemhir::PointerType::toAtemTypeString() const -> std::string
{
    if (auto atem_hir_type = mlir::dyn_cast<mlir::atemhir::AtemHIRUtilTypeInterface>(this->getPointeeType()))
    {
        return atem_hir_type.toAtemTypeString().append(".&");
    }
    std::string result;
    llvm::raw_string_ostream os(result);
    os << "!atem.ptr<";
    this->getPointeeType().print(os);
    os << ">";
    return os.str();
}

auto mlir::atemhir::PointerType::verify(function_ref<InFlightDiagnostic()> emit_error, Type pointee_type) -> LogicalResult
{
    return success();
}

auto mlir::atemhir::PointerType::getTypeSizeInBits(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return llvm::TypeSize::getFixed(64);
}

auto mlir::atemhir::PointerType::getABIAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return 8;
}

auto mlir::atemhir::PointerType::getPreferredAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return 8;
}

//========================================================
// Array Type Definitions
//========================================================

auto mlir::atemhir::ArrayType::toAtemTypeString() const -> std::string
{
    if (auto atem_hir_type = mlir::dyn_cast<mlir::atemhir::AtemHIRUtilTypeInterface>(this->getElementType()))
    {
        return atem_hir_type.toAtemTypeString().append("[").append(std::to_string(this->getSize())).append("]");
    }
    std::string result;
    llvm::raw_string_ostream os(result);
    os << "!atem.array<";
    this->getElementType().print(os);
    os << ", " << this->getSize() << ">";
    return os.str();
}

auto mlir::atemhir::ArrayType::getTypeSizeInBits(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return this->getSize() * data_layout.getTypeSizeInBits(this->getElementType());
}

auto mlir::atemhir::ArrayType::getABIAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return data_layout.getTypeABIAlignment(this->getElementType());
}

auto mlir::atemhir::ArrayType::getPreferredAlignment(DataLayout const &data_layout, DataLayoutEntryListRef params) const -> uint64_t
{
    return data_layout.getTypeABIAlignment(this->getElementType());
}

//========================================================
// Function Type Definitions
//========================================================

auto mlir::atemhir::FunctionType::toAtemTypeString() const -> std::string
{
    auto mlirTypeToString = [](Type type) -> std::string {
        if (auto atem_hir_type = mlir::dyn_cast<mlir::atemhir::AtemHIRUtilTypeInterface>(type))
        {
            return atem_hir_type.toAtemTypeString().append(".&");
        }
        std::string result;
        llvm::raw_string_ostream os(result);
        type.print(os);
        return os.str();
    };
    std::string arg_type_list;
    auto arg_list = this->getInputs();
    if (not arg_list.empty())
    {
        for (std::size_t i = 0; i < arg_list.size() - 1; ++i)
        {
            arg_type_list.append(mlirTypeToString(arg_list[i])).append(", ");
        }
        arg_type_list.append(mlirTypeToString(arg_list.back()));
    }
    std::string result_type = mlirTypeToString(this->getResults().front());
    return std::string{"("}.append(arg_type_list).append(") -> ").append(result_type);
}

auto mlir::atemhir::FunctionType::clone(TypeRange inputs, TypeRange outputs) const -> FunctionType
{
    return atemhir::FunctionType::get(llvm::to_vector(inputs), llvm::to_vector(outputs));
}

//========================================================
// Unit Type Definitions
//========================================================

auto mlir::atemhir::UnitType::toAtemTypeString() const -> std::string
{
    return "Unit";
}

//========================================================
// Noreturn Type Definitions
//========================================================

auto mlir::atemhir::NoreturnType::toAtemTypeString() const -> std::string
{
    return "Noreturn";
}

//========================================================
// String Type Definitions
//========================================================

auto mlir::atemhir::StringType::toAtemTypeString() const -> std::string
{
    return "String";
}

//========================================================
// Rune Type Definitions
//========================================================

auto mlir::atemhir::RuneType::toAtemTypeString() const -> std::string
{
    return "Rune";
}