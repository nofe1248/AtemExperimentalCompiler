lexer grammar AtemLexer;

@lexer::header {
	#include <stack>
}
@lexer::members {
	std::stack<int> curly = std::stack<int>{};

	void reset() override{
		curly = std::stack<int>{};
	}
}

//Keywords

KeywordAnd: 'and';
KeywordAs: 'as';
KeywordBool: 'Bool';
KeywordBreak: 'break';
KeywordConst: 'const';
KeywordContinue: 'continue';
KeywordDo: 'do';
KeywordElse: 'else';
KeywordExtern: 'extern';
KeywordFalse: 'false';
KeywordFloat16: 'Float16';
KeywordFloat32: 'Float32';
KeywordFloat64: 'Float64';
KeywordFloat80: 'Float80';
KeywordFloat128: 'Float128';
KeywordFor: 'for';
KeywordFunc: 'func';
KeywordIf: 'if';
KeywordInt: 'Int' DecimalLiteral;
KeywordISize: 'ISize';
KeywordNoreturn: 'Noreturn';
KeywordNot: 'not';
KeywordNull: 'null';
KeywordOr: 'or';
KeywordReturn: 'return';
KeywordRune: 'Rune';
KeywordStep: 'step';
KeywordString: 'String';
KeywordStruct: 'struct';
KeywordThen: 'then';
KeywordTrue: 'true';
KeywordType: 'type';
KeywordUInt: 'UInt' DecimalLiteral;
KeywordUndefined: 'undefined';
KeywordUnit: 'Unit';
KeywordUsize: 'USize';
KeywordVar: 'var';
KeywordWhile: 'while';
KeywordWith: 'with';

//Identifier

Identifier:	IdentifierHead IdentifierCharacters?;

fragment IdentifierHead:
	[a-zA-Z]
	| '_'
	| '\u00A8'
	| '\u00AA'
	| '\u00AD'
	| '\u00AF'
	| [\u00B2-\u00B5]
	| [\u00B7-\u00BA]
	| [\u00BC-\u00BE]
	| [\u00C0-\u00D6]
	| [\u00D8-\u00F6]
	| [\u00F8-\u00FF]
	| [\u0100-\u02FF]
	| [\u0370-\u167F]
	| [\u1681-\u180D]
	| [\u180F-\u1DBF]
	| [\u1E00-\u1FFF]
	| [\u200B-\u200D]
	| [\u202A-\u202E]
	| [\u203F-\u2040]
	| '\u2054'
	| [\u2060-\u206F]
	| [\u2070-\u20CF]
	| [\u2100-\u218F]
	| [\u2460-\u24FF]
	| [\u2776-\u2793]
	| [\u2C00-\u2DFF]
	| [\u2E80-\u2FFF]
	| [\u3004-\u3007]
	| [\u3021-\u302F]
	| [\u3031-\u303F]
	| [\u3040-\uD7FF]
	| [\uF900-\uFD3D]
	| [\uFD40-\uFDCF]
	| [\uFDF0-\uFE1F]
	| [\uFE30-\uFE44]
	| [\uFE47-\uFFFD]
	| [\u{10000}-\u{1FFFD}]
	| [\u{20000}-\u{2FFFD}]
	| [\u{30000}-\u{3FFFD}]
	| [\u{40000}-\u{4FFFD}]
	| [\u{50000}-\u{5FFFD}]
	| [\u{60000}-\u{6FFFD}]
	| [\u{70000}-\u{7FFFD}]
	| [\u{80000}-\u{8FFFD}]
	| [\u{90000}-\u{9FFFD}]
	| [\u{A0000}-\u{AFFFD}]
	| [\u{B0000}-\u{BFFFD}]
	| [\u{C0000}-\u{CFFFD}]
	| [\u{D0000}-\u{DFFFD}]
	| [\u{E0000}-\u{EFFFD}];

fragment IdentifierCharacter:
	[0-9]
	| [\u0300-\u036F]
	| [\u1DC0-\u1DFF]
	| [\u20D0-\u20FF]
	| [\uFE20-\uFE2F]
	| IdentifierHead;

fragment IdentifierCharacters: IdentifierCharacter+;

//Literals

BinaryLiteral: Sign? '0b' BinaryDigit BinaryLiteralCharacters?;
fragment BinaryDigit: [01];
fragment BinaryLiteralCharacter: BinaryDigit | '_';
fragment BinaryLiteralCharacters: BinaryLiteralCharacter+;

OctalLiteral: Sign? '0o' OctalDigit OctalLiteralCharacters?;
fragment OctalDigit: [0-7];
fragment OctalLiteralCharacter: OctalDigit | '_';
fragment OctalLiteralCharacters: OctalLiteralCharacter+;

DecimalDigits: DecimalDigit+;
DecimalLiteral: Sign? DecimalDigit DecimalLiteralCharacters?;
fragment DecimalDigit: [0-9];
fragment DecimalLiteralCharacter: DecimalDigit | '_';
fragment DecimalLiteralCharacters: DecimalLiteralCharacter+;

HexadecimalLiteral:
	Sign? '0x' HexadecimalDigit HexadecimalLiteralCharacters?;
fragment HexadecimalDigit: [0-9a-fA-F];
fragment HexadecimalLiteralCharacter: HexadecimalDigit | '_';
fragment HexadecimalLiteralCharacters:
	HexadecimalLiteralCharacter+;

FloatingPointLiteral:
	Sign? DecimalLiteral DecimalFraction? DecimalExponent?
	| Sign? HexadecimalLiteral HexadecimalFraction? HexadecimalExponent;
fragment DecimalFraction: '.' DecimalLiteral;
fragment DecimalExponent:
	FloatingPointE Sign? DecimalLiteral;
fragment HexadecimalFraction:
	'.' HexadecimalDigit HexadecimalLiteralCharacters?;
fragment HexadecimalExponent:
	FloatingPointP Sign? DecimalLiteral;
fragment FloatingPointE: [eE];
fragment FloatingPointP: [pP];
fragment Sign: [+-];

//Operators

LeftCurly: '{' {
					if (!curly.empty()) {
						int top = curly.top();
						curly.pop();
						curly.push(top + 1);
						curly.pop();
					}
				};
RightCurly: '}' { if(!curly.empty())
					{
						int top = curly.top();
						curly.pop();
						curly.push(top - 1);
						if(curly.top() == 0)
						{
							curly.pop();
							popMode();
						}
					}
				};
LeftParenthese: '(';
RightParenthese: ')';
LeftSquare: '[';
RightSquare: ']';

Dot: '.';
Colon: ':';
Semicolon: ';';
Comma: ',';
Question: '?';
Bang: '!';
Underscore: '_';

Add: '+';
Minus: '-';
Mul: '*';
Divide: '/';
Remainder: '%';

Assign: '=';
AddAssign: '+=';
SubAssign: '-=';
MulAssign: '*=';
DivideAssign: '/=';
RemainderDivideAssign: '%=';
BitLeftShiftAssign: '<<=';
BitRightShiftAssign: '>>=';
BitAndAssign: '&=';
BitOrAssign: '|=';

GreaterThan: '>';
LessThan: '<';
GreaterThanOrEqual: '>=';
LessThanOrEqual: '<=';
NotEqual: '!=';
Equal: '==';

BitNot: '~';
BitAnd: '&';
BitOr: '|';
BitXor: '^';
BitLeftShift: '<<';
BitRightShift: '>>';

PointerType: '.&';
PointerDeref: '.*';
ObjectAddress: '.@';

ClosedRange: '...';
RightOpenRange: '..<';
LeftOpenRange: '<..';
OpenedRange: '<.<';

Arrow: '->';

Builtin: '@';

OperatorHeadOther:
	[\u00A1-\u00A7]
	| [\u00A9\u00AB]
	| [\u00AC\u00AE]
	| [\u00B0-\u00B1\u00B6\u00BB\u00BF\u00D7\u00F7]
	| [\u2016-\u2017\u2020-\u2027]
	| [\u2030-\u203E]
	| [\u2041-\u2053]
	| [\u2055-\u205E]
	| [\u2190-\u23FF]
	| [\u2500-\u2775]
	| [\u2794-\u2BFF]
	| [\u2E00-\u2E7F]
	| [\u3001-\u3003]
	| [\u3008-\u3020\u3030];

OperatorFollowingCharacter:
	[\u0300-\u036F]
	| [\u1DC0-\u1DFF]
	| [\u20D0-\u20FF]
	| [\uFE00-\uFE0F]
	| [\uFE20-\uFE2F]
	| [\u{E0100}-\u{E01EF}];

//Whitespaces and comments
Whitespace: [ \n\r\t\u000B\u000C\u0000]+ -> channel(HIDDEN);
Newline: ('\r' '\n'? | '\n') -> channel(HIDDEN);
LineComment: ('//' (~[/!] | '//') ~[\r\n]* | '//') -> channel (HIDDEN);
InnerLineDocComment: '//!' ~[\n\r]* -> channel (HIDDEN);
OuterLineDocComment: '///!' (~[/] ~[\n\r]*)? -> channel (HIDDEN);

BlockComment
   :
   (
      '//{'
      (
         ~[*!]
         | '**'
         | BlockCommentOrDoc
      )
      (
         BlockCommentOrDoc
         | ~[*]
      )*? '}//'
      | '//{}//'
   ) -> channel (HIDDEN)
   ;

InnerBlockComment
   : '//!{'
   (
      BlockCommentOrDoc
      | ~[*]
   )*? '}//' -> channel (HIDDEN)
   ;

OuterBlockComment
   : '///!{'
   (
      ~[*]
      | BlockCommentOrDoc
   )
   (
      BlockCommentOrDoc
      | ~[*]
   )*? '}///' -> channel (HIDDEN)
   ;

BlockCommentOrDoc
   :
   (
      BlockComment
      | InnerBlockComment
      | OuterBlockComment
   ) -> channel (HIDDEN)
   ;

//String literals

MultiLineExtendedStringOpen:
	'#'+ '"""' -> pushMode(MultiLineExtended);

SingleLineExtendedStringOpen:
	'#'+ '"' -> pushMode(SingleLineExtended);

MultiLineStringOpen: '"""' -> pushMode(MultiLine);

SingleLineStringOpen: '"' -> pushMode(SingleLine);

mode SingleLine;

SingleLineStringClose: '"' -> popMode;

QuotedSingleLineText: QuotedText;

mode MultiLine;

MultiLineStringClose: '"""' -> popMode;

QuotedMultiLineText: MultilineQuotedText;

mode SingleLineExtended;

SingleLineExtendedStringClose: '"' '#'+ -> popMode;

QuotedSingleLineExtendedText: ~[\r\n"]+;

mode MultiLineExtended;

MultiLineExtendedStringClose: '"""' '#'+ -> popMode;

QuotedMultiLineExtendedText: ~["]+ | '"' '"'?;

fragment QuotedText: QuotedTextItem+;

fragment QuotedTextItem: EscapedCharacter | ~["\n\r$];

fragment MultilineQuotedText:
	EscapedCharacter
	| ~[\\"]+
	| '"' '"'?
	| EscapedNewline;

fragment EscapeSequence: '\\' '#'*;

fragment EscapedCharacter:
	EscapeSequence (
		[0\\tnr"'\u201c]
		| 'u' '{' UnicodeScalarDigits '}'
	);

//Between one and eight hexadecimal digits
fragment UnicodeScalarDigits:
	HexadecimalDigit HexadecimalDigit? HexadecimalDigit? HexadecimalDigit? HexadecimalDigit?
		HexadecimalDigit? HexadecimalDigit? HexadecimalDigit?;

fragment EscapedNewline:
	EscapeSequence InlineSpaces? LineBreak;

fragment InlineSpaces: [\u0009\u0020];

fragment LineBreak: [\u000A\u000D]| '\u000D' '\u000A';

mode DEFAULT_MODE;

CharLiteralOpen: '\'' -> pushMode(CharLiteral);

mode CharLiteral;

ValidChar
	: EscapedCharacter
	| ~ ['\\\r\n]
	;

CharLiteralClose: '\'' -> popMode;