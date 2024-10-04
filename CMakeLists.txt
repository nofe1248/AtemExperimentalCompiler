cmake_minimum_required(VERSION 3.30)
project(AtemExperimentalCompiler)

set(CMAKE_CXX_STANDARD 26)
cmake_policy(SET CMP0135 NEW)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${STANDALONE_SOURCE_DIR}/include)
include_directories(${STANDALONE_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_definitions(-D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING)

add_executable(AtemExperimentalCompiler)

target_sources(
        AtemExperimentalCompiler PRIVATE
        FILE_SET atem_exp_cpp_modules TYPE CXX_MODULES FILES
        Modules/Lexer/Atem.Lexer.cpp
        Modules/Parser/Atem.Parser.cpp
        Modules/Utils/Result/Atem.Utils.Result.cpp
        Modules/Main/Atem.Main.cpp
)

target_link_libraries(
        AtemExperimentalCompiler PRIVATE

        AtemHIR
        AtemHIRInterfaces

        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRFunctionInterfaces
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
)

target_link_options(AtemExperimentalCompiler PRIVATE -pthread -femulated-tls)
target_compile_options(AtemExperimentalCompiler PRIVATE -pthread -femulated-tls)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Modules)
add_subdirectory(Modules/IR)

find_path(CPP_PEGLIB_INCLUDE_DIRS "peglib.h")
target_include_directories(AtemExperimentalCompiler PRIVATE ${CPP_PEGLIB_INCLUDE_DIRS})
