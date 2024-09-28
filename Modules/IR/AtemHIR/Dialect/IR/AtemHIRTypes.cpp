#include "IR/AtemHIR/Dialect/IR/AtemHIRTypes.hpp"
#include "IR/AtemHIR/Dialect/IR/AtemHIRDialect.hpp"

#include "mlir/Support/LLVM.h"

#define GET_TYPEDEF_CLASSES
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIRTypes.cpp.inc"

#include <mlir/IR/Builders.h>

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

    return StringSwitch<function_ref<Type()>>(mnemonic)
        .Default([&] {
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

    TypeSwitch<Type>(type)
        .Default([](Type type) {
            llvm::report_fatal_error("printer is missing a handler for this type");
        });
}

//========================================================
// Integer Type Definitions
//========================================================

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
// Floating-point Types Definitions
//========================================================

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