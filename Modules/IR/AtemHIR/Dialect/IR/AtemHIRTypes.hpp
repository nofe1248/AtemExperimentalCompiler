#ifndef ATEM_HIR_TYPES_HPP
#define ATEM_HIR_TYPES_HPP

#include "IR/AtemHIR/Interfaces/AtemHIRFPTypeInterface.hpp"

#include "mlir/IR/Types.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/TypeSize.h"

#define GET_TYPEDEF_CLASSES
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIRTypes.h.inc"

#endif //ATEM_HIR_TYPES_HPP
