
module;

#include <string>
#include <memory>
#include <optional>

#include "peglib.h"

#include "llvm/Support/raw_ostream.h"

export module Atem.Parser;

export namespace atem::parser
{
class AtemASTExtraInformation
{

};
using AtemAST = peg::AstBase<AtemASTExtraInformation>;
class AtemParserPackrat final
{
public:
    explicit AtemParserPackrat(std::string source) : atem_source_(std::move(source))
    {
        this->atem_parser_.load_grammar(atem_peg_grammar);
        this->atem_parser_.enable_packrat_parsing();
        this->atem_parser_.enable_ast<AtemAST>();
        this->atem_parser_.set_logger([](size_t line, size_t col, std::string const &msg, std::string const &rule) {
           llvm::errs() << line << ":" << col << ": " << msg << "\n";
        });
    }

    [[nodiscard]] auto parse() const -> std::optional<std::shared_ptr<AtemAST>>
    {
        std::shared_ptr<AtemAST> result;
        if (this->atem_parser_.parse(this->atem_source_, result) )
        {
            return this->atem_parser_.optimize_ast(result);
        }
        return std::nullopt;
    }

private:
    peg::parser atem_parser_;
    std::string atem_source_;

public:
    static constexpr inline std::string_view atem_peg_grammar = R"(
Root <- skip Statement* eof

# ** Top level **

Statement <-
Declaration /
Expr

# ** expressions **

Declaration <-
VariableDeclaration /
ConstantDeclaration /
FunctionDeclaration /
StructDeclaration

VariableDeclaration <-
IDENTIFIER COLON KEYWORD_var TypeExpr EQUAL Expr

ConstantDeclaration <-
IDENTIFIER COLON KEYWORD_const TypeExpr EQUAL Expr

FunctionDeclaration <-
IDENTIFIER COLON FunctionTypeExpr (Block / EQUAL Expr)

StructDeclaration <-
IDENTIFIER COLON KEYWORD_type StructTypeExpr

Expr <- AssignExpr

AssignExpr <- BinaryLogicalExpr (AssignOp BinaryLogicalExpr)?

BinaryLogicalExpr <- CompareExpr (BinaryLogicalOp CompareExpr)*

CompareExpr <- BitwiseExpr (CompareOp BitwiseExpr)*

BitwiseExpr <- BitShiftExpr (BitwiseOp BitShiftExpr)*

BitShiftExpr <- AdditionExpr (BitShiftOp AdditionExpr)*

AdditionExpr <- MultiplyExpr (AdditionOp MultiplyExpr)*

MultiplyExpr <- PrefixExpr (MultiplyOp PrefixExpr)*

PrefixExpr <- PrefixOp* SuffixExpr

SuffixExpr <- PrimaryExpr (SuffixOp / FunctionCallArguments)*

PrimaryExpr <-
IfExpr /
WhileExpr /
DoWhileExpr /
ForExpr /
KEYWORD_break Expr? /
KEYWORD_continue Expr? /
KEYWORD_return Expr? /
Block /
CurlySuffixExpr /
CHAR_LITERAL /
FLOAT /
INTEGER /
STRINGLITERAL /
KEYWORD_true /
KEYWORD_false /
KEYWORD_undefined

BlockOrThen <-
LBRACE Statement* RBRACE /
KEYWORD_then Statement

BlockOrSingleExpr <-
LBRACE Statement* RBRACE /
Statement

IfExpr <-
KEYWORD_if Expr BlockOrThen
(KEYWORD_else BlockOrSingleExpr)?

WhileExpr <-
KEYWORD_while Expr BlockOrThen
(KEYWORD_else BlockOrSingleExpr)?

DoWhileExpr <-
KEYWORD_do BlockOrThen KEYWORD_while Expr
(KEYWORD_else BlockOrSingleExpr)?

ForExpr <-
KEYWORD_for Expr KEYWORD_with BlockOrSingleExpr KEYWORD_step Expr BlockOrThen
(KEYWORD_else BlockOrSingleExpr)?

CurlySuffixExpr <- TypeExpr InitList?

InitList <-
LBRACE (FieldInit COMMA)* FieldInit? RBRACE /
LBRACE (Expr COMMA)* Expr? RBRACE

FieldInit <- DOT IDENTIFIER EQUAL Expr

Block <- LBRACE Statement* RBRACE

TypeExpr <-
IDENTIFIER /
PrimitiveTypeExpr /
FunctionTypeExpr /
StructTypeExpr /
StructDeclaration /
BUILTINIDENTIFIER FunctionCallArguments

PrimitiveTypeExpr <-
PrimitiveIntegerTypeExpr /
PrimitiveFloatTypeExpr /
KEYWORD_bool /
KEYWORD_noreturn /
KEYWORD_unit /
KEYWORD_rune /
KEYWORD_string /
KEYWORD_ptr TypeExpr

PrimitiveIntegerTypeExpr <-
KEYWORD_int INTEGER /
KEYWORD_uint INTEGER /
KEYWORD_isize /
KEYWORD_usize

PrimitiveFloatTypeExpr <-
KEYWORD_float16 /
KEYWORD_float32 /
KEYWORD_float64 /
KEYWORD_float80 /
KEYWORD_float128

FunctionTypeExpr <-
KEYWORD_func FunctionParameterList? MINUSRARROW TypeExpr?

FunctionParameterList <-
LPAREN (IDENTIFIER COLON TypeExpr)* RPAREN

StructTypeExpr <-
KEYWORD_struct LBRACE Declaration* RBRACE

# Operators
AssignOp <-
ASTERISKEQUAL /
SLASHEQUAL /
PERCENTEQUAL /
PLUSEQUAL /
MINUSEQUAL /
LARROW2EQUAL /
RARROW2EQUAL /
AMPERSANDEQUAL /
CARETEQUAL /
PIPEEQUAL /
EQUAL

CompareOp <-
EQUALEQUAL /
EXCLAMATIONMARKEQUAL /
LARROW /
RARROW /
LARROWEQUAL /
RARROWEQUAL

BitwiseOp <-
AMPERSAND /
CARET /
PIPE /
TILDE

BitShiftOp <-
LARROW2 /
RARROW2

AdditionOp <-
PLUS /
MINUS /
PLUS2 /
MINUS2

MultiplyOp <-
ASTERISK /
SLASH /
PERCENT /
ASTERISK2

BinaryLogicalOp <-
KEYWORD_and /
KEYWORD_or

PrefixOp <-
KEYWORD_not /
MINUS /
TILDE

SuffixOp <-
DOTASTERISK /
DOTAMPERSAND /
DOT IDENTIFIER /
LBRACKET Expr RBRACKET

FunctionCallArguments <- LPAREN ExprList RPAREN

# lists

ExprList <- (Expr COMMA)* Expr?

# *** Tokens ***
eof <- !.
bin <- [01]
bin_ <- '_'? bin
oct <- [0-7]
oct_ <- '_'? oct
hex <- [0-9a-fA-F]
hex_ <- '_'? hex
dec <- [0-9]
dec_ <- '_'? dec

bin_int <- bin bin_*
oct_int <- oct oct_*
dec_int <- dec dec_*
hex_int <- hex hex_*

ox80_oxBF <- [\200-\277]
oxF4 <- '\364'
ox80_ox8F <- [\200-\217]
oxF1_oxF3 <- [\361-\363]
oxF0 <- '\360'
ox90_0xBF <- [\220-\277]
oxEE_oxEF <- [\356-\357]
oxED <- '\355'
ox80_ox9F <- [\200-\237]
oxE1_oxEC <- [\341-\354]
oxE0 <- '\340'
oxA0_oxBF <- [\240-\277]
oxC2_oxDF <- [\302-\337]

mb_utf8_literal <-
       oxF4      ox80_ox8F ox80_oxBF ox80_oxBF
     / oxF1_oxF3 ox80_oxBF ox80_oxBF ox80_oxBF
     / oxF0      ox90_0xBF ox80_oxBF ox80_oxBF
     / oxEE_oxEF ox80_oxBF ox80_oxBF
     / oxED      ox80_ox9F ox80_oxBF
     / oxE1_oxEC ox80_oxBF ox80_oxBF
     / oxE0      oxA0_oxBF ox80_oxBF
     / oxC2_oxDF ox80_oxBF

ascii_char_not_nl_slash_squote <- [\000-\011\013-\046-\050-\133\135-\177]

char_escape
    <- "\\x" hex hex
     / "\\u{" hex+ "}"
     / "\\" [nr\\t'"]
char_char
    <- mb_utf8_literal
     / char_escape
     / ascii_char_not_nl_slash_squote

string_char
    <- char_escape
     / [^\\"\n]

# ** Comments **

line_comment <- '//' ![!/][^\n]* / '////' [^\n]*
line_string <- ("\\\\" [^\n]* [ \n]*)+
skip <- ([ \r\n\t] / line_comment)*


# ** literals and identifiers **

CHAR_LITERAL <- "'" char_char "'" skip
FLOAT
    <- "0x" hex_int "." hex_int ([pP] [-+]? dec_int)? skip
     /      dec_int "." dec_int ([eE] [-+]? dec_int)? skip
     / "0x" hex_int [pP] [-+]? dec_int skip
     /      dec_int [eE] [-+]? dec_int skip
INTEGER
    <- "0b" bin_int skip
     / "0o" oct_int skip
     / "0x" hex_int skip
     /      dec_int   skip
STRINGLITERALSINGLE <- "\"" string_char* "\"" skip
STRINGLITERAL
    <- STRINGLITERALSINGLE
     / (line_string                 skip)+
IDENTIFIER
    <- !keyword [A-Za-z_] [A-Za-z0-9_]* skip
     / "@\"" string_char* "\""                            skip
BUILTINIDENTIFIER <- "@"[A-Za-z_][A-Za-z0-9_]* skip

# ** sigils **

AMPERSAND            <- '&'      ![=]      skip
AMPERSANDEQUAL       <- '&='               skip
ASTERISK             <- '*'      ![*=]     skip
ASTERISK2            <- '**'               skip
ASTERISKEQUAL        <- '*='               skip
CARET                <- '^'      ![=]      skip
CARETEQUAL           <- '^='               skip
COLON                <- ':'                skip
COMMA                <- ','                skip
DOT                  <- '.'      ![*.?]    skip
DOTAMPERSAND         <- '.&'               skip
DOTASTERISK          <- '.*'               skip
EQUAL                <- '='      ![>=]     skip
EQUALEQUAL           <- '=='               skip
EXCLAMATIONMARKEQUAL <- '!='               skip
LARROW               <- '<'      ![<=]     skip
LARROW2              <- '<<'     ![=]      skip
LARROW2EQUAL         <- '<<='              skip
LARROWEQUAL          <- '<='               skip
LBRACE               <- '{'                skip
LBRACKET             <- '['                skip
LPAREN               <- '('                skip
MINUS                <- '-'      ![=>]     skip
MINUS2               <- '--'               skip
MINUSEQUAL           <- '-='               skip
MINUSRARROW          <- '->'               skip
PERCENT              <- '%'      ![=]      skip
PERCENTEQUAL         <- '%='               skip
PIPE                 <- '|'      ![|=]     skip
PIPEEQUAL            <- '|='               skip
PLUS                 <- '+'      ![+=]     skip
PLUS2                <- '++'               skip
PLUSEQUAL            <- '+='               skip
RARROW               <- '>'      ![>=]     skip
RARROW2              <- '>>'     ![=]      skip
RARROW2EQUAL         <- '>>='              skip
RARROWEQUAL          <- '>='               skip
RBRACE               <- '}'                skip
RBRACKET             <- ']'                skip
RPAREN               <- ')'                skip
SLASH                <- '/'      ![=]      skip
SLASHEQUAL           <- '/='               skip
TILDE                <- '~'                skip

# ** Keywords **
end_of_word <- ![a-zA-Z0-9_] skip
KEYWORD_and         <- 'and' end_of_word
KEYWORD_bool        <- 'Bool' end_of_word
KEYWORD_break       <- 'break' end_of_word
KEYWORD_const       <- 'const' end_of_word
KEYWORD_continue    <- 'continue' end_of_word
KEYWORD_do          <- 'do' end_of_word
KEYWORD_else        <- 'else' end_of_word
KEYWORD_false       <- 'false' end_of_word
KEYWORD_float16     <- 'float16' end_of_word
KEYWORD_float32     <- 'float32' end_of_word
KEYWORD_float64     <- 'float64' end_of_word
KEYWORD_float80     <- 'float80' end_of_word
KEYWORD_float128    <- 'float128' end_of_word
KEYWORD_for         <- 'for' end_of_word
KEYWORD_func        <- 'func' end_of_word
KEYWORD_if          <- 'if' end_of_word
KEYWORD_int         <- 'Int' end_of_word
KEYWORD_isize       <- 'ISize' end_of_word
KEYWORD_noreturn    <- 'Noreturn' end_of_word
KEYWORD_not         <- 'not' end_of_word
KEYWORD_or          <- 'or' end_of_word
KEYWORD_ptr         <- 'ptr' end_of_word
KEYWORD_return      <- 'return' end_of_word
KEYWORD_rune        <- 'Rune' end_of_word
KEYWORD_step        <- 'step' end_of_word
KEYWORD_string      <- 'String' end_of_word
KEYWORD_struct      <- 'struct' end_of_word
KEYWORD_then        <- 'then' end_of_word
KEYWORD_true        <- 'true' end_of_word
KEYWORD_type        <- 'type' end_of_word
KEYWORD_uint        <- 'UInt' end_of_word
KEYWORD_undefined   <- 'undefined' end_of_word
KEYWORD_unit        <- 'Unit' end_of_word
KEYWORD_usize       <- 'USize' end_of_word
KEYWORD_var         <- 'var' end_of_word
KEYWORD_while       <- 'while' end_of_word
KEYWORD_with        <- 'with' end_of_word

keyword <-
KEYWORD_and /
KEYWORD_bool /
KEYWORD_break /
KEYWORD_const /
KEYWORD_continue /
KEYWORD_do /
KEYWORD_else /
KEYWORD_false /
KEYWORD_float16 /
KEYWORD_float32 /
KEYWORD_float64 /
KEYWORD_float80 /
KEYWORD_float128 /
KEYWORD_for /
KEYWORD_func /
KEYWORD_if /
KEYWORD_int /
KEYWORD_isize /
KEYWORD_noreturn /
KEYWORD_not /
KEYWORD_or /
KEYWORD_return /
KEYWORD_rune /
KEYWORD_step /
KEYWORD_string /
KEYWORD_struct /
KEYWORD_then /
KEYWORD_true /
KEYWORD_type /
KEYWORD_uint /
KEYWORD_unit /
KEYWORD_usize /
KEYWORD_var /
KEYWORD_while /
KEYWORD_with
    )";
};
} // namespace atem::parser