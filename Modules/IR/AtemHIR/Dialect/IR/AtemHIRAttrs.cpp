#include "IR/AtemHIR/Dialect/IR/AtemHIRAttrs.hpp"
#include "IR/AtemHIR/Dialect/IR/AtemHIRDialect.hpp"
#include "IR/AtemHIR/Dialect/IR/AtemHIRTypes.hpp"

static auto printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value, mlir::Type type) -> void;
static auto parseFloatLiteral(mlir::AsmParser &parser, mlir::FailureOr<llvm::APFloat> &value, mlir::Type type) -> mlir::ParseResult;

#define GET_ATTRDEF_CLASSES
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIRAttrDefs.cpp.inc"

//========================================================
// Helper Function
//========================================================

static auto printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value, mlir::Type type) -> void
{
    p << value;
}
static auto parseFloatLiteral(mlir::AsmParser &parser, mlir::FailureOr<llvm::APFloat> &value, mlir::Type type) -> mlir::ParseResult
{
    auto fp_interface = mlir::dyn_cast<mlir::atemhir::AtemHIRFPTypeInterface>(type);
    if (not fp_interface)
    {
        return mlir::success();
    }

    llvm::APFloat raw_value(fp_interface.getFloatSemantics());
    bool info_loss = false;

    if (parser.parseFloat(fp_interface.getFloatSemantics(), raw_value))
    {
        return parser.emitError(parser.getCurrentLocation(), "expected floating-point value");
    }

    value.emplace(raw_value);

    value->convert(fp_interface.getFloatSemantics(), llvm::RoundingMode::TowardZero, &info_loss);

    return llvm::success();
}

//========================================================
// Dialect Attribute Initialization
//========================================================

auto mlir::atemhir::AtemHIRDialect::registerAttributes() -> void
{
    this->addAttributes<
#define GET_ATTRDEF_LIST
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIRAttrDefs.cpp.inc"

        >();
}

//========================================================
// Dialect Attribute Parser & Printer
//========================================================

auto mlir::atemhir::AtemHIRDialect::parseAttribute(::mlir::DialectAsmParser &parser, ::mlir::Type type) const -> ::mlir::Attribute
{
    SMLoc type_loc = parser.getCurrentLocation();
    StringRef mnemonic;
    Attribute attr;
    if (OptionalParseResult const parse_result = generatedAttributeParser(parser, &mnemonic, type, attr); parse_result.has_value())
    {
        return attr;
    }
    parser.emitError(type_loc, "unknown attribute in Atem HIR dialect");
    return {};
}
auto mlir::atemhir::AtemHIRDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter &os) const -> void
{
    if (failed(generatedAttributePrinter(attr, os)))
    {
        llvm_unreachable("unexpected Atem HIR type kind for attribute");
    }
}

//========================================================
// IntAttr Definitions
//========================================================

auto mlir::atemhir::IntAttr::parse(AsmParser &parser, Type ods_type) -> Attribute
{
    APInt ap_value, value;

    if (not mlir::isa<IntType>(ods_type))
    {
        return {};
    }

    auto type = mlir::cast<IntType>(ods_type);

    if (parser.parseLess())
    {
        return {};
    }

    if (type.isSigned())
    {
        if (parser.parseInteger(value))
        {
            parser.emitError(parser.getCurrentLocation(), "expected integer value");
        }
        ap_value = APInt(type.getWidth(), toString(value, 10, true), 10);
    }
    else
    {
        if (parser.parseInteger(value))
        {
            parser.emitError(parser.getCurrentLocation(), "expected integer value");
        }
        ap_value = APInt(type.getWidth(), toString(value, 10, false), 10);
    }

    if (parser.parseGreater())
    {
        return {};
    }

    return IntAttr::get(type, ap_value);
}

auto mlir::atemhir::IntAttr::print(AsmPrinter &printer) const -> void
{
    auto type = mlir::cast<IntType>(this->getType());
    printer << '<' << (type.isSigned() ? this->getSignedIntString() : this->getUnsignedIntString()) << '>';
}

auto mlir::atemhir::IntAttr::verify(function_ref<InFlightDiagnostic()> emit_error, Type type, APInt value) -> LogicalResult
{
    if (not mlir::isa<IntType>(type))
    {
        emit_error() << "expected 'atemhir.int' type";
        return failure();
    }

    auto int_type = mlir::cast<IntType>(type);
    if (value.getBitWidth() != int_type.getWidth())
    {
        emit_error() << "type and value bit width mismatch: "
                     << "value: " << value.getBitWidth() << " != "
                     << "type: " << int_type.getWidth();
        return failure();
    }
    return success();
}

//========================================================
// FPAttr Definitions
//========================================================

auto mlir::atemhir::FPAttr::getZero(Type type) -> FPAttr
{
    return get(type, APFloat::getZero(mlir::cast<AtemHIRFPTypeInterface>(type).getFloatSemantics()));
}

auto mlir::atemhir::FPAttr::verify(function_ref<InFlightDiagnostic()> emit_error, Type type, APFloat value) -> LogicalResult
{
    auto fp_type_interface = mlir::dyn_cast<AtemHIRFPTypeInterface>(type);
    if (not fp_type_interface)
    {
        emit_error() << "expected floating-point type";
        return failure();
    }
    if (APFloat::SemanticsToEnum(fp_type_interface.getFloatSemantics()) != APFloat::SemanticsToEnum(value.getSemantics()))
    {
        emit_error() << "floating-point semantics mismatch";
        return failure();
    }
    return success();
}

//========================================================
// ConstantArrayAttr Definitions
//========================================================

auto mlir::atemhir::ConstantArrayAttr::verify(function_ref<InFlightDiagnostic()> emit_error, Type type, Attribute elements) -> LogicalResult
{
    if (not mlir::isa<ArrayAttr>(elements))
    {
        return emit_error() << "constant array expects ArrayAttr";
    }

    auto array_attr = mlir::cast<ArrayAttr>(elements);
    auto array_type = mlir::cast<ArrayType>(type);

    if (array_type.getSize() != array_attr.size())
    {
        return emit_error() << "constant array size should match type size";
    }

    LogicalResult element_type_check = success();

    array_attr.walkImmediateSubElements(
        [&](Attribute element) {
            if (element_type_check.failed())
            {
                return ;
            }
            auto typed_attr = mlir::dyn_cast<TypedAttr>(element);
            if (not typed_attr or typed_attr.getType() != array_type.getElementType())
            {
                element_type_check = failure();
                emit_error() << "constant array element type mismatch";
            }
        },
        [&](Type type) {}
    );
    return element_type_check;
}

auto mlir::atemhir::ConstantArrayAttr::parse(AsmParser &parser, Type type) -> Attribute
{
    FailureOr<Type> result_type;
    FailureOr<Attribute> result_attr;
    SMLoc loc = parser.getCurrentLocation();

    if (parser.parseLess())
    {
        return {};
    }

    result_attr = FieldParser<Attribute>::parse(parser);
    if (failed(result_attr))
    {
        parser.emitError(
            parser.getCurrentLocation(),
            "failed to parse ConstantArrayAttr parameter 'value' which is to be a 'Attribute'"
        );
        return {};
    }

    if (mlir::dyn_cast<ArrayAttr>(*result_attr))
    {
        if (parser.parseOptionalColon().failed())
        {
            result_type = type;
        }
        else
        {
            result_type = mlir::FieldParser<Type>::parse(parser);
            if (failed(result_type))
            {
                parser.emitError(
                    parser.getCurrentLocation(),
                    "failed to parse ConstantArrayAttr parameter 'type' which is to be a 'Type'"
                );
                return {};
            }
        }
    }
    else
    {
        parser.emitError(
            parser.getCurrentLocation(),
            "failed to parse ConstantArrayAttr parameter 'value' which is to be a 'ArrayAttr'"
        );
        return {};
    }

    if (parser.parseGreater())
    {
        return {};
    }

    return parser.getChecked<ConstantArrayAttr>(loc, parser.getContext(), result_type.value(), result_attr.value());
}

auto mlir::atemhir::ConstantArrayAttr::print(AsmPrinter &printer) const -> void
{
    printer << "<";
    printer.printStrippedAttrOrType(this->getElements());
    printer << ">";
}