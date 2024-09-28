#ifndef ATEM_HIR_OPS_HPP
#define ATEM_HIR_OPS_HPP

#pragma once

#include "./AtemHIRDialect.hpp"
#include "./AtemHIRTypes.hpp"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeImplementation.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "Modules/IR/AtemHIR/Dialect/IR/AtemHIR.h.inc"

#endif