parser grammar AtemParser;

options {
    tokenVocab = AtemLexer;
}

program: decls EOF;

decls: decl*;

decl:
variable_decl |
constant_decl |
function_decl |
struct_decl
;

stmts: stmt*;

stmt:
expr
;

block_or_then_expr:
scope_expr | KeywordThen expr
;

block_or_expr:
scope_expr | expr
;

variable_decl:
Identifier Colon KeywordVar type_expr Assign expr;

constant_decl:
Identifier Colon KeywordConst type_expr Assign expr;

function_decl:
Identifier Colon KeywordFunc function_type_expr Assign block_or_expr
;

struct_decl:
Identifier Colon KeywordType Assign struct_type_expr
;

expr:
  LeftParenthese expr RightParenthese       #paren_expression
| Remainder Assign expr                       #discard_expression
| literal_expr                              #literal_expression
| decl                                      #decl_expression
| scope_expr                                #scope_expression
| Identifier                                #identifier_expression
| type_expr                                 #type_expression
| return_expr                               #return_expression
| if_expr                                   #if_expression
| cfor_expr                                 #cfor_expression
| while_expr                                #while_expression
| do_while_expr                             #do_while_expression
| break_expr                                #break_expression
| continue_expr                             #continue_expression
| builtin_call_expr                         #builtin_call_expression
| expr KeywordAs type_expr                  #conversion_expression
| expr postfix_operator+                    #postfix_expression
| prefix_operator+ expr                     #prefix_expression
| expr mul_operator expr                    #multiplicative_expression
| expr add_operator expr                    #additive_expression
| expr bitshift_operator expr               #bitshift_expression
| expr bitwise_operator expr                #bitwise_expression
| expr comparison_operator expr             #comparison_expression
| expr binary_logical_operator expr         #binary_logical_expression
| expr assign_operator expr                 #assignment_expression
;

discard: Underscore;

builtin_call_expr:
Builtin Identifier function_call_operator
;

if_expr:
KeywordIf expr block_or_then_expr
(KeywordElse block_or_expr)?
;

while_expr:
KeywordWhile expr block_or_then_expr
(KeywordElse block_or_expr)?
;

do_while_expr:
KeywordDo block_or_expr KeywordWhile expr
(KeywordElse block_or_expr)?
;

cfor_expr:
KeywordFor expr KeywordWith block_or_expr KeywordStep block_or_expr block_or_then_expr
(KeywordElse block_or_expr)?
;

break_expr: KeywordBreak expr?;

continue_expr: KeywordContinue expr?;

scope_expr:
LeftCurly stmts RightCurly
;

return_expr: KeywordReturn expr?;

operator:
  postfix_operator
| prefix_operator
| mul_operator
| add_operator
| bitshift_operator
| bitwise_operator
| comparison_operator
| binary_logical_operator
| assign_operator
;

postfix_operator: 
PointerDeref |
ObjectAddress |
LeftSquare integer_literal RightSquare |
function_call_operator |
member_access_operator
;
prefix_operator: 
Minus |
KeywordNot |
BitNot
;
mul_operator: 
Mul |
Divide |
Remainder
;
add_operator: 
Add |
Minus
;
bitshift_operator: 
BitLeftShift |
BitRightShift
;
bitwise_operator: 
BitAnd |
BitOr |
BitXor
;
comparison_operator:
LessThan |
LessThanOrEqual |
GreaterThan |
GreaterThanOrEqual |
NotEqual |
Equal
;
binary_logical_operator: 
KeywordAnd |
KeywordOr
;
assign_operator: 
Assign |
AddAssign |
SubAssign |
MulAssign |
DivideAssign |
RemainderDivideAssign |
BitLeftShiftAssign |
BitRightShiftAssign |
BitAndAssign |
BitOrAssign |
BitXorAssign
;

function_call_operator: LeftParenthese (expr Comma)* expr? RightParenthese;

member_access_operator: Dot Identifier;

literal_expr:
integer_literal |
FloatingPointLiteral |
KeywordTrue |
KeywordFalse |
KeywordUndefined |
KeywordNull |
string_literal |
array_literal
;

integer_literal:
BinaryLiteral |
OctalLiteral |
DecimalLiteral |
HexadecimalLiteral |
DecimalDigits
;

string_literal:
extended_string_literal |
static_string_literal
;

extended_string_literal:
MultiLineExtendedStringOpen QuotedMultiLineExtendedText+ MultiLineExtendedStringClose |
SingleLineExtendedStringOpen QuotedSingleLineExtendedText+ SingleLineExtendedStringClose
;

static_string_literal:
MultiLineStringOpen QuotedMultiLineText* MultiLineStringClose |
SingleLineStringOpen QuotedSingleLineText* SingleLineStringClose
;

array_literal: 
LeftSquare integer_literal RightSquare type_expr 
LeftCurly (expr Comma)* expr? RightCurly
;

type_expr:
  simple_type_expr                                  #simple_type_expression
| type_expr PointerType                             #pointer_type_expression
| LeftSquare integer_literal RightSquare type_expr  #array_type_expression
| function_type_expr                                #function_type_expression
| struct_type_expr                                  #struct_type_expression
| Identifier                                        #identifier_type_expression
;

function_type_expr:
function_parameter_list? Arrow type_expr?
;

function_parameter_list:
LeftParenthese (function_parameter Comma)* function_parameter? RightParenthese
;

function_parameter: Identifier Colon type_expr;

struct_type_expr:
KeywordStruct LeftCurly decls RightCurly
;

simple_type_expr:
  KeywordBool       #bool_type_expression
| KeywordInt        #int_type_expression
| KeywordUInt       #uint_type_expression
| KeywordNoreturn   #noreturn_type_expression
| KeywordUnit       #unit_type_expression
| float_type_expr   #float_type_expression
| KeywordRune       #rune_type_expression
| KeywordString     #string_type_expression
;

float_type_expr:
  KeywordFloat16
| KeywordFloat32
| KeywordFloat64
| KeywordFloat80
| KeywordFloat128
;