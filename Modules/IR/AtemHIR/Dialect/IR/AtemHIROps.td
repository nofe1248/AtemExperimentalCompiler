#ifndef ATEM_HIR_OPS
#define ATEM_HIR_OPS

include "AtemHIRDialect.td"
include "AtemHIRTypes.td"

include "IR/AtemHIR/Interfaces/AtemHIRLoopOpInterface.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"

class AtemHIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<AtemHIR_Dialect, mnemonic, traits>;

//========================================================
// Atem HIR Memory Order
//========================================================

def MemoryOrderRelaxed : I32EnumAttrCase<"Relaxed", 0, "relaxed">;
def MemoryOrderConsume : I32EnumAttrCase<"Comsume", 1, "consume">;
def MemoryOrderAcquire : I32EnumAttrCase<"Acquire", 2, "acquire">;
def MemoryOrderRelease : I32EnumAttrCase<"Release", 3, "release">;
def MemoryOrderAcqRel : I32EnumAttrCase<"AcquireRelease", 4, "acq_rel">;
def MemoryOrderSeqCst : I32EnumAttrCase<"SequentiallyConsistent", 5, "seq_cst">;

def MemoryOrder :
    I32EnumAttr<
        "MemoryOrder",
        "Memory order",
        [
            MemoryOrderRelaxed, MemoryOrderConsume, MemoryOrderAcquire, MemoryOrderRelease, MemoryOrderAcqRel, MemoryOrderSeqCst
        ]
    >
{
    let cppNamespace = "::mlir::atemhir";
}

//========================================================
// Atem HIR Constant Operation
//========================================================

def AtemHIR_ConstantOp :
    AtemHIR_Op<"constant", [ConstantLike, Pure, AllTypesMatch<["value", "res"]>]>
{
    let summary = "constant operation";

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs AnyType:$res);

    let assemblyFormat = "$value attr-dict";

    let hasVerifier = 1;

    let hasFolder = 1;
}

//========================================================
// Atem HIR Return Operation
//========================================================

def AtemHIR_ReturnOp :
    AtemHIR_Op<
        "return", 
        [
            Terminator, 
            ReturnLike, 
            ParentOneOf<["FunctionOp", "IfOp", "WhileOp", "ForOp", "DoWhileOp", "ScopeOp"]>
        ]
    >
{
    let summary = "return operation";

    let arguments = (ins Variadic<AnyType>:$data);

    let assemblyFormat = "attr-dict ($data^ `:` type($data))?";
}

//========================================================
// Atem HIR Function Operation
//========================================================

def AtemHIR_FunctionOp :
    AtemHIR_Op<"function", [
        IsolatedFromAbove,
        FunctionOpInterface,
        RegionKindInterface
    ]>
{
    let summary = "function operation";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<AtemHIR_FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    dag regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        auto getCallableRegion() -> ::mlir::Region * { return &this->getBody(); }

        auto getArgumentTypes() -> ::llvm::ArrayRef<::mlir::Type> { return this->getFunctionType().getInputs(); }
        auto getResultTypes() -> ::llvm::ArrayRef<::mlir::Type> { return this->getFunctionType().getResults(); }

        static auto getRegionKind(unsigned index) -> ::mlir::RegionKind { return mlir::RegionKind::SSACFG; }
    }];
}

//========================================================
// Atem HIR Call Operation
//========================================================

def AtemHIR_CallOp :
    AtemHIR_Op<"call", [CallOpInterface]>
{
    let summary = "call operation";
    
    let arguments = (ins
        SymbolRefAttr:$callee,
        Variadic<AnyType>:$arg_operands
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = "$callee `(` $arg_operands `)` attr-dict `:` functional-type($arg_operands, results)";

    let extraClassDeclaration = [{
        auto getCallableForCallee() -> ::mlir::CallInterfaceCallable { return getCalleeAttr(); }
        auto setCalleeFromCallable(::mlir::CallInterfaceCallable callee) -> void { setCalleeAttr(callee.get<::mlir::SymbolRefAttr>()); }
    }];
}

//========================================================
// Atem HIR Yield Operation
//========================================================

def AtemHIR_YieldOp :
    AtemHIR_Op<
        "yield",
        [
            ReturnLike,
            Terminator,
            ParentOneOf<[
                "IfOp", "WhileOp", "DoWhileOp", "ForOp", "ScopeOp"
            ]>
        ]
    >
{
    let summary = "Carry the result value of a region";

    let arguments = (ins Variadic<AnyType>:$args);
    let assemblyFormat = [{ ($args^ `:` type($args))? attr-dict }];

    let builders = [
        OpBuilder<(ins), [{}]>
    ];
}

//========================================================
// Atem HIR Break Operation
//========================================================

def AtemHIR_BreakOp :
    AtemHIR_Op<"break", [ReturnLike, Terminator]>
{
    let summary = "A break expression";

    let arguments = (ins Variadic<AnyType>:$args);
    let assemblyFormat = [{ ($args^ `:` type($args))? attr-dict }];

    let hasVerifier = 1;
}

//========================================================
// Atem HIR Continue Operation
//========================================================

def AtemHIR_ContinueOp :
    AtemHIR_Op<"continue", [Terminator]>
{
    let summary = "A continue expression";

    let arguments = (ins Variadic<AnyType>:$args);
    let assemblyFormat = [{ ($args^ `:` type($args))? attr-dict }];

    let hasVerifier = 1;
}

//========================================================
// Atem HIR Condition Operation
//========================================================

def AtemHIR_ConditionOp :
    AtemHIR_Op<
        "condition",
        [
            Pure,
            Terminator,
            ParentOneOf<[
                "WhileOp", "DoWhileOp", "ForOp"
            ]>,
            DeclareOpInterfaceMethods<
                RegionBranchTerminatorOpInterface,
                ["getSuccessorRegions"]
            >
        ]
    >
{
    let summary = "Loop continuation condition";

    let arguments = (ins 
        AtemHIR_BoolType:$condition,
        Variadic<AnyType>:$args
    );

    let assemblyFormat = [{ 
        `(` $condition `)` attr-dict ($args^ `:` type($args))?
    }];

    let hasVerifier = 1;
}

//========================================================
// Atem HIR If Operation
//========================================================

def AtemHIR_IfOp :
    AtemHIR_Op<
        "if",
        [
            DeclareOpInterfaceMethods<RegionBranchOpInterface>,
            RecursivelySpeculatable,
            AutomaticAllocationScope,
            NoRegionArguments
        ]
    >
{
    let summary = "Represents an if-then-else expression";

    let arguments = (ins AtemHIR_BoolType:$condition);
    let results = (outs Optional<AnyType>:$result);
    let regions = (region 
        AnyRegion:$then_region, 
        AnyRegion:$else_region
    );

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins 
            "::mlir::Value":$cond, 
            "bool":$has_else_branch, 
            CArg<"::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>">:$then_builder,
            CArg<"::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>", "nullptr">:$else_builder
        )>,
        OpBuilder<(ins 
            "::mlir::Value":$cond, 
            "bool":$has_else_branch, 
            CArg<"::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>", "::mlir::atemhir::buildTerminatedBody">:$then_builder,
            CArg<"::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>", "nullptr">:$else_builder
        )>
    ];
}

//========================================================
// Atem HIR While & Do-While Operation
//========================================================

class AtemHIR_WhileOpBase<string mnemonic> :
    AtemHIR_Op<mnemonic, [
        AtemHIRLoopOpInterface,
        RecursiveMemoryEffects,
        SingleBlock
    ]>
{
    defvar is_while = !eq(mnemonic, "while");
    let summary = "Atem HIR " # !if(is_while, "while", "do-while") # " loop";

    let builders = [
        OpBuilder<
            (ins
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$condition_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$body_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$else_builder
            ),
            [{
                ::mlir::OpBuilder::InsertionGuard guard($_builder);
                $_builder.createBlock($_state.addRegion());
            }] # 
            !if
            (
                is_while,
                [{
                    condition_builder($_builder, $_state.location);
                    $_builder.createBlock($_state.addRegion());
                    body_builder($_builder, $_state.location);
                }],
                [{
                    body_builder($_builder, $_state.location);                    
                    $_builder.createBlock($_state.addRegion());
                    condition_builder($_builder, $_state.location);
                }]
            ) #
            [{
                $_builder.createBlock($_state.addRegion());
                else_builder($_builder, $_state.location);
            }]
        >,
        OpBuilder<
            (ins
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$condition_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>":$body_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>":$else_builder
            ),
            [{
                ::mlir::Type body_type, else_type;
                ::mlir::OpBuilder::InsertionGuard guard($_builder);
                $_builder.createBlock($_state.addRegion());
            }] # 
            !if
            (
                is_while,
                [{
                    condition_builder($_builder, $_state.location);
                    $_builder.createBlock($_state.addRegion());
                    body_builder($_builder, body_type, $_state.location);
                }],
                [{
                    body_builder($_builder, body_type, $_state.location);                    
                    $_builder.createBlock($_state.addRegion());
                    condition_builder($_builder, $_state.location);
                }]
            ) #
            [{
                $_builder.createBlock($_state.addRegion());
                else_builder($_builder, else_type, $_state.location);

                if (body_type and else_type and body_type == else_type) {
                    $_state.addTypes(::mlir::TypeRange{body_type});
                }
            }]
        >,
        OpBuilder<
            (ins
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$condition_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$body_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>":$else_builder
            ),
            [{
                ::mlir::Type else_type;
                ::mlir::OpBuilder::InsertionGuard guard($_builder);
                $_builder.createBlock($_state.addRegion());
            }] # 
            !if
            (
                is_while,
                [{
                    condition_builder($_builder, $_state.location);
                    $_builder.createBlock($_state.addRegion());
                    body_builder($_builder, $_state.location);
                }],
                [{
                    body_builder($_builder, $_state.location);                    
                    $_builder.createBlock($_state.addRegion());
                    condition_builder($_builder, $_state.location);
                }]
            ) #
            [{
                $_builder.createBlock($_state.addRegion());
                else_builder($_builder, else_type, $_state.location);
                if (else_type) {
                    $_state.addTypes(::mlir::TypeRange{else_type});
                }
            }]
        >,
        OpBuilder<
            (ins
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$condition_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>":$body_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$else_builder
            ),
            [{
                ::mlir::Type body_type;
                ::mlir::OpBuilder::InsertionGuard guard($_builder);
                $_builder.createBlock($_state.addRegion());
            }] # 
            !if
            (
                is_while,
                [{
                    condition_builder($_builder, $_state.location);
                    $_builder.createBlock($_state.addRegion());
                    body_builder($_builder, body_type, $_state.location);
                }],
                [{
                    body_builder($_builder, body_type, $_state.location);                    
                    $_builder.createBlock($_state.addRegion());
                    condition_builder($_builder, $_state.location);
                }]
            ) #
            [{
                $_builder.createBlock($_state.addRegion());
                else_builder($_builder, $_state.location);
                if (body_type) {
                    $_state.addTypes(::mlir::TypeRange{body_type});
                }
            }]
        >,
    ];
}

def AtemHIR_WhileOp :
    AtemHIR_WhileOpBase<"while">
{
    let arguments = (ins Variadic<AnyType>:$inits);
    let results = (outs Variadic<AnyType>:$result);
    let regions = (region 
        SizedRegion<1>:$cond, 
        MinSizedRegion<1>:$body,
        MinSizedRegion<1>:$else
    );

    let assemblyFormat = [{ 
    (`(` $inits^ `:` type($inits) `)`)?
    `cond` $cond 
    `body` $body 
    `else` $else
    (`:` type($result)^)? attr-dict }];
}

def AtemHIR_DoWhileOp :
    AtemHIR_WhileOpBase<"do">
{
    let arguments = (ins Variadic<AnyType>:$inits);
    let results = (outs Variadic<AnyType>:$result);
    let regions = (region 
        MinSizedRegion<1>:$body,
        SizedRegion<1>:$cond,
        MinSizedRegion<1>:$else
    );

    let assemblyFormat = [{ 
    (`(` $inits^ `:` type($inits) `)`)?
    `body` $body 
    `cond` $cond 
    `else` $else
    (`:` type($result)^)? attr-dict }];

    let extraClassDeclaration = [{
        auto getEntry() -> ::mlir::Region & { return this->getBody(); }
    }];
}

//========================================================
// Atem HIR For Operation
//========================================================

def AtemHIR_ForOp :
    AtemHIR_Op<
        "for", 
        [
            AtemHIRLoopOpInterface,
            RecursiveMemoryEffects,
            SingleBlock
        ]
    >
{
    let summary = "Represents a for loop";

    let arguments = (ins Variadic<AnyType>:$inits);
    let results = (outs Variadic<AnyType>:$result);
    let regions = (region
        SizedRegion<1>:$cond,
        MinSizedRegion<1>:$body,
        SizedRegion<1>:$step,
        MinSizedRegion<1>:$else
    );

    let assemblyFormat = [{
        (`(` $inits^ `:` type($inits) `)`)?
        `cond` $cond
        `body` $body
        `step` $step
        `else` $else
        (`:` type($result)^)?
        attr-dict
    }];

    let builders = [
        OpBuilder<
            (ins
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$cond_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>":$body_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$step_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>":$else_builder
            )
        >,
        OpBuilder<
            (ins
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$cond_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>":$body_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$step_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$else_builder
            )
        >,
        OpBuilder<
            (ins
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$cond_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$body_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$step_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>":$else_builder
            )
        >,
        OpBuilder<
            (ins
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$cond_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$body_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$step_builder,
                "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$else_builder
            )
        >
    ];

    let extraClassDeclaration = [{
        auto maybeGetStep() -> ::mlir::Region * { return &this->getStep(); }
        auto getRegionsInExecutionOrder() -> ::llvm::SmallVector<::mlir::Region *> {
            return ::llvm::SmallVector<::mlir::Region *, 3>{&this->getCond(), &this->getBody(), &this->getStep()};
        }
    }];
}

//========================================================
// Atem HIR Variable Initialization Operation
//========================================================

def AtemHIR_UndefinedOp :
    AtemHIR_Op<"undefined", [Pure]>
{
    let summary = "Creates an undefined value of Atem HIR dialect type";
    let results = (outs AnyType:$res);
    let assemblyFormat = "attr-dict `:` type($res)";
}

def AtemHIR_ZeroInitOp :
    AtemHIR_Op<"zeroinit", [Pure]>
{
    let summary = "Creates an zero-init value of Atem HIR dialect type";
    let results = (outs AnyType:$res);
    let assemblyFormat = "attr-dict `:` type($res)";
}

//========================================================
// Atem HIR Allocate Variable Operation
//========================================================

class AllocationTypesMatchWith<
        string summary, string lhs_, string rhs_,
        string transform, string comparator = "::std::equal_to<>()"
    > :
    PredOpTrait<
        summary,
        CPred<
            comparator # "(" #
            !subst("$_self", "$" # lhs_ # ".getType()", transform) #
                   ", $" # rhs_ # ")"
        >
    >
{
    string lhs = lhs_;
    string rhs = rhs_;
    string transformer = transform;
}

def AtemHIR_AllocateVarOp :
    AtemHIR_Op<"allocate_var", [
        AllocationTypesMatchWith<
            "'allocaType' matches pointee type of 'addr'",
            "addr",
            "allocation_type",
            "::mlir::cast<::mlir::atemhir::PointerType>($_self).getPointeeType()"
        >,
        DeclareOpInterfaceMethods<PromotableAllocationOpInterface>
    ]>
{
    let summary = "Allocate a scope-local variable on stack";

    let arguments = (ins
        Optional<PrimitiveInt>:$dyn_allocation_size,
        TypeAttr:$allocation_type,
        StrAttr:$name,
        UnitAttr:$init,
        ConfinedAttr<
            OptionalAttr<I64Attr>, 
            [IntMinValue<0>]
        >:$alignment
    );

    let results = (outs
        Res<AtemHIR_PointerType, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$addr
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "::mlir::Type":$addr,
            "::mlir::Type":$allocation_type,
            "::llvm::StringRef":$name,
            "::mlir::IntegerAttr":$alignment
        )>,
        OpBuilder<(ins
            "::mlir::Type":$addr,
            "::mlir::Type":$allocation_type,
            "::llvm::StringRef":$name,
            "::mlir::IntegerAttr":$alignment,
            "::mlir::Value":$dyn_allocation_size
        ),
        [{
            if (dyn_allocation_size)
            {
                $_state.addOperands(dyn_allocation_size);
            }
            build($_builder, $_state, addr, allocation_type, name, alignment);
        }]
        >
    ];

    let extraClassDeclaration = [{
        auto isDynamic() -> bool {
            return static_cast<bool>(this->getDynAllocationSize());
        }
    }];

    let assemblyFormat = [{
        $allocation_type `,` qualified(type($addr)) `,`
        ($dyn_allocation_size^ `:` type($dyn_allocation_size) `,`)?
        `[` $name (`,` `init` $init^)? `]`
        attr-dict
    }];

    let hasVerifier = 0;
}

//========================================================
// Atem HIR Pointer Load Operation
//========================================================

def AtemHIR_LoadOp :
    AtemHIR_Op<
        "load", 
        [
            TypesMatchWith<
                "type of 'result' matches pointee type of 'addr'",
                "addr",
                "result",
                "::mlir::cast<::mlir::atemhir::PointerType>($_self).getPointeeType()"
            >,
            DeclareOpInterfaceMethods<PromotableMemOpInterface>
        ]
    >
{
    let summary = "Load value from pointer";

    let arguments = (ins
        Arg<AtemHIR_PointerType, "the address to load from", [MemRead]>:$addr,
        UnitAttr:$is_deref,
        UnitAttr:$is_volatile,
        OptionalAttr<I64Attr>:$alignment,
        OptionalAttr<MemoryOrder>:$memory_order
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        (`deref` $is_deref^)?
        (`volatile` $is_volatile^)?
        (`align` `(` $alignment^ `)`)?
        (`atomic` `(` $memory_order^ `)`)?
        $addr `:` qualified(type($addr)) `,` type($result) attr-dict
    }];
}

//========================================================
// Atem HIR Pointer Store Operation
//========================================================

def AtemHIR_StoreOp :
    AtemHIR_Op<
        "store",
        [
            TypesMatchWith<
                "type of 'value' matches pointee type of 'addr'",
                "addr",
                "value",
                "::mlir::cast<::mlir::atemhir::PointerType>($_self).getPointeeType()"
            >,
            DeclareOpInterfaceMethods<PromotableMemOpInterface>
        ]
    >
{
    let summary = "Store value to memory address";

    let arguments = (ins
        AnyType:$value,
        Arg<AtemHIR_PointerType, "the address to store the value", [MemWrite]>:$addr,
        UnitAttr:$is_volatile,
        OptionalAttr<I64Attr>:$alignment,
        OptionalAttr<MemoryOrder>:$memory_order
    );

    let assemblyFormat = [{
        (`volatile` $is_volatile^)?
        (`align` `(` $alignment^ `)`)?
        (`atomic` `(` $memory_order^ `)`)?
        $value `,` $addr attr-dict `:` type($value) `,` qualified(type($addr))
    }];
}

//========================================================
// Atem HIR Pointer Copy Operation
//========================================================

def AtemHIR_CopyOp :
    AtemHIR_Op<
        "copy", 
        [
            SameTypeOperands,
            DeclareOpInterfaceMethods<PromotableMemOpInterface>
        ]
    >
{
    let summary = "Copy contents from a pointer to another pointer";

    let arguments = (ins
        Arg<AtemHIR_PointerType, "source address", [MemRead]>:$source,
        Arg<AtemHIR_PointerType, "destination address", [MemWrite]>:$destination,
        UnitAttr:$is_volatile
    );

    let assemblyFormat = [{
        $source `to` $destination (`volatile` $is_volatile^)? attr-dict `:` qualified(type($destination))
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        auto getType() -> ::mlir::atemhir::PointerType { return this->getSource().getType(); }
        auto getLength() -> unsigned { return ::mlir::DataLayout::closest(*this).getTypeSize(this->getType().getPointeeType()); }
    }];
}

//========================================================
// Atem HIR Cast Operation
//========================================================

def CastKind_BitCast : I32EnumAttrCase<"bitcast", 0>;
def CastKind_IntegerPromotion : I32EnumAttrCase<"int_promotion", 1>;
def CastKind_FloatPromotion : I32EnumAttrCase<"float_promotion", 2>;
def CastKind_IntegerToFloat : I32EnumAttrCase<"int_to_float", 3>;
def CastKind_FloatToInteger : I32EnumAttrCase<"float_to_int", 4>;
def CastKind_BoolToInteger : I32EnumAttrCase<"bool_to_int", 5>;
def CastKind_IntegerToBool : I32EnumAttrCase<"int_to_bool", 6>;
def CastKind_IntegerNarrowing : I32EnumAttrCase<"int_narrowing", 7>;
def CastKind_FloatNarrowing : I32EnumAttrCase<"float_narrowing", 8>;

def CastKind : 
    I32EnumAttr<
        "CastKind",
        "cast operation kind",
        [
            CastKind_BitCast, CastKind_IntegerPromotion, CastKind_FloatPromotion, CastKind_IntegerToFloat, CastKind_FloatToInteger,
            CastKind_BoolToInteger, CastKind_IntegerToBool, CastKind_IntegerNarrowing, CastKind_FloatNarrowing
        ]
    >
{
    let cppNamespace = "::mlir::atemhir";
}

def AtemHIR_CastOp :
    AtemHIR_Op<
        "cast",
        [
            Pure,
            DeclareOpInterfaceMethods<PromotableOpInterface>
        ]
    >
{
    let summary = "Builtin conversion between values of different types(explicit or implicit)";

    let arguments = (ins
        CastKind:$kind,
        AnyType:$source
    );

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $kind `,` $source `:` type($source) `to` type($result) attr-dict
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
}

//========================================================
// Atem HIR Unary Operation
//========================================================

def UnaryOpKind_Inc : I32EnumAttrCase<"Inc", 0, "inc">;
def UnaryOpKind_Dec : I32EnumAttrCase<"Dec", 1, "dec">;
def UnaryOpKind_Pos : I32EnumAttrCase<"Pos", 2, "pos">;
def UnaryOpKind_Neg : I32EnumAttrCase<"Neg", 3, "neg">;
def UnaryOpKind_Not : I32EnumAttrCase<"Not", 4, "not">;
def UnaryOpKind_BitNot : I32EnumAttrCase<"BitNot", 5, "bnot">;

def UnaryOpKind : 
    I32EnumAttr<
        "UnaryOpKind",
        "unary operation kind",
        [
            UnaryOpKind_Inc,
            UnaryOpKind_Dec,
            UnaryOpKind_Pos,
            UnaryOpKind_Neg,
            UnaryOpKind_Not,
            UnaryOpKind_BitNot
        ]
    >
{
    let cppNamespace = "::mlir::atemhir";
}

def AtemHIR_UnaryOp :
    AtemHIR_Op<"unary", [Pure, SameOperandsAndResultType]>
{
    let summary = "Unary operations";

    let arguments = (ins
        Arg<UnaryOpKind, "unary operation kind">:$kind,
        AnyType:$input
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $kind `,` $input `:` type($input) attr-dict
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
}

//========================================================
// Atem HIR Binary Operation
//========================================================

def BinaryOpKind_Add : I32EnumAttrCase<"Add", 0, "add">;
def BinaryOpKind_Sub : I32EnumAttrCase<"Sub", 1, "sub">;
def BinaryOpKind_Mul : I32EnumAttrCase<"Mul", 2, "mul">;
def BinaryOpKind_Div : I32EnumAttrCase<"Div", 3, "div">;
def BinaryOpKind_Rem : I32EnumAttrCase<"Rem", 4, "rem">;
def BinaryOpKind_And : I32EnumAttrCase<"And", 5, "and">;
def BinaryOpKind_Or : I32EnumAttrCase<"Or", 6, "or">;
def BinaryOpKind_BitAnd : I32EnumAttrCase<"BitAnd", 7, "band">;
def BinaryOpKind_BitOr : I32EnumAttrCase<"BitOr", 8, "bor">;
def BinaryOpKind_BitXor : I32EnumAttrCase<"BitXor", 9, "bxor">;

def BinaryOpKind :
    I32EnumAttr<
        "BinaryOpKind",
        "binary operation kind",
        [
            BinaryOpKind_Add,
            BinaryOpKind_Sub,
            BinaryOpKind_Mul,
            BinaryOpKind_Div,
            BinaryOpKind_Rem,
            BinaryOpKind_And,
            BinaryOpKind_Or,
            BinaryOpKind_BitAnd,
            BinaryOpKind_BitOr,
            BinaryOpKind_BitXor
        ]
    >
{
    let cppNamespace = "::mlir::atemhir";
}

def AtemHIR_BinaryOp :
    AtemHIR_Op<"binary", [Pure, SameTypeOperands, SameOperandsAndResultType]>
{
    let summary = "Binary operations";

    let arguments = (ins
        Arg<BinaryOpKind, "binary op kind">:$kind,
        AnyType:$lhs,
        AnyType:$rhs,
        UnitAttr:$no_unsigned_wrap,
        UnitAttr:$no_signed_wrap
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $kind `,` $lhs `,` $rhs
        (`nsw` $no_signed_wrap^)?
        (`nuw` $no_unsigned_wrap^)?
        `:` type($lhs) attr-dict
    }];

    let hasVerifier = 1;
}

//========================================================
// Atem HIR Compare Operation
//========================================================

def CompareOpKind_LT : I32EnumAttrCase<"lt", 0>;
def CompareOpKind_LE : I32EnumAttrCase<"le", 1>;
def CompareOpKind_GT : I32EnumAttrCase<"gt", 2>;
def CompareOpKind_GE : I32EnumAttrCase<"ge", 3>;
def CompareOpKind_EQ : I32EnumAttrCase<"eq", 4>;
def CompareOpKind_NE : I32EnumAttrCase<"ne", 5>;

def CompareOpKind :
    I32EnumAttr<
        "CompareOpKind",
        "compare operation kind",
        [
            CompareOpKind_LT,
            CompareOpKind_LE,
            CompareOpKind_GT,
            CompareOpKind_GE,
            CompareOpKind_EQ,
            CompareOpKind_NE
        ]
    >
{
    let cppNamespace = "::mlir::atemhir";
}

def AtemHIR_CompareOp :
    AtemHIR_Op<"compare", [Pure, SameTypeOperands]>
{
    let summary = "Compare two values and produce a boolean result";

    let arguments = (ins 
        Arg<CompareOpKind, "compare operation kind">:$kind,
        AnyType:$lhs,
        AnyType:$rhs
    );
    let results = (outs AtemHIR_BoolType);

    let assemblyFormat = [{
        $kind `,` $lhs `,` $rhs `:` type($lhs) attr-dict
    }];

    let hasVerifier = 0;
}

//========================================================
// Atem HIR Bitshift Operation
//========================================================

def AtemHIR_ShiftOp :
    AtemHIR_Op<"shift", [Pure]>
{
    let summary = "Bitshifting operation";

    let arguments = (ins
        AtemHIR_IntType:$value,
        AtemHIR_IntType:$amount,
        UnitAttr:$is_shifting_left
    );
    let results = (outs AtemHIR_IntType:$result);

    let assemblyFormat = [{
        (`left` $is_shifting_left^) : (`right`)?
        $value `,` $amount `:` type($value) `,` type($amount)
        `->` type($result) attr-dict
    }];

    let hasVerifier = 0;
}

//========================================================
// Atem HIR Scope Operation
//========================================================

def AtemHIR_ScopeOp :
    AtemHIR_Op<
        "scope", 
        [
            DeclareOpInterfaceMethods<RegionBranchOpInterface>,
            RecursivelySpeculatable,
            AutomaticAllocationScope,
            NoRegionArguments
        ]
    >
{
    let summary = "Represents a scope in Atem";

    let results = (outs Optional<AnyType>:$result);
    let regions = (region AnyRegion:$scope_region);

    let assemblyFormat = [{
        custom<OmittedTerminatorRegion>($scope_region) (`:` type($result)^)? attr-dict
    }];

    let hasVerifier = 1;
    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins
            "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Type &, ::mlir::Location)>":$scope_builder
        )>,
        OpBuilder<(ins
            "::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$scope_builder
        )>
    ];
}

//========================================================
// Atem HIR Discard Operation
//========================================================

def AtemHIR_DiscardOp :
    AtemHIR_Op<"discard", [Pure]>
{
    let summary = "Value discarding operation";

    let arguments = (ins
        AnyType:$value
    );

    let assemblyFormat = [{
        $value `:` type($value) attr-dict
    }];

    let hasVerifier = 0;
}

#endif  //ATEM_HIR_OPS