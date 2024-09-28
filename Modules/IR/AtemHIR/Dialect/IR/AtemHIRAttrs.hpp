#ifndef ATEM_HIR_ATTRS_HPP
#define ATEM_HIR_ATTRS_HPP

#include "./AtemHIRTypes.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

#define GET_ATTRDEF_CLASSES
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIRAttrDefs.h.inc"

#endif //ATEM_HIR_ATTRS_HPP
