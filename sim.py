from dataclasses import dataclass
from fractions import Fraction
from typing import Union, MutableMapping, List, TypeVar, Optional

class EndOfStream(Exception):
    pass

@dataclass
class Stream:
    source: str
    pos: int = 0

    def from_string(source):
        return Stream(source)

    def next_char(self):
        if self.pos >= len(self.source):
            raise EndOfStream()
        t = self.pos
        self.pos += 1
        return self.source[t]

    def unget_char(self):
        assert self.pos > 0
        self.pos -= 1

@dataclass
class NumToken:
    what: str

@dataclass
class BoolToken:
    what: str

@dataclass
class StrToken:
    what: str

@dataclass
class Keyword:
    what: str

@dataclass
class Operator:
    what: str

@dataclass
class Identifier:
    what: str

Token = (
    NumToken |
    BoolToken |
    StrToken |
    Keyword |
    Operator |
    Identifier
)

whitespace = " \t\n"
symbol_operators = "+ - × / ^ < > ≥ ≤ = ≠ ~ ← !".split()
keywords = "if then else end while do done begin let letmut in".split()
word_operators = "and or not quot rem".split()

class EndOfTokens(Exception):
    pass

def wordToken(word: str) -> Token:
    if word in keywords:
        return Keyword(word)
    if word in word_operators:
        return Operator(word)
    if word == "True" or word == "False":
        return BoolToken(word)
    return Identifier(word)

@dataclass
class Lexer:
    stream: Stream
    save: Optional[Token] = None

    def from_stream(stream):
        return Lexer(stream)

    def next_token(self) -> Token:
        try:
            match self.stream.get_char():
                case c if c in whitespace:
                    return self.next_token()
                case c if c in symbol_operators:
                    return Operator(c)
                case '"':
                    s = c
                    try:
                        while (c := self.stream.get_char()) != '"':
                            s += c
                        return StrToken(s)
                    except EndOfStream:
                        raise TokenError()
                case c if c.isdigit():
                    s = c
                    try:
                        while (c := self.stream.get_char()).isdigit():
                            s += c
                        self.stream.unget_char()
                        return NumToken(s)
                    except EndOfStream:
                        return NumToken(s)
                case c if c.isalpha():
                    s = c
                    try:
                        while (c := self.stream.get_char()).isalpha():
                            s += c
                        self.stream.unget_char()
                        return wordToken(s)
                    except EndOfStream:
                        return wordToken(s)
        except EndOfStream:
            raise EndOfTokens()

    def peek_token(self) -> Token:
        if self.save is not None:
            return self.save
        self.save = self.next_token()
        return self.save

    def advance(self):
        assert self.save is not None
        self.save = None

    def match_token(self, expected: Token):
        if self.save is None:
            self.save = self.next_token()
        if self.save == expected:
            return self.advance()
        raise TokenError()

@dataclass
class Parser:
    lexer: Lexer

    def from_lexer(lexer):
        self.lexer = lexer

    def parse_if(self):
        self.lexer.match_token(Keyword("if"))
        c = self.parse_expr()
        self.lexer.match_token(Keyword("then"))
        t = self.parse_expr()
        self.lexer.match_token(Keyword("else"))
        f = self.parse_expr()
        self.lexer.match_token(Keyword("end"))
        return IfElse(c, t, f)

    def parse_while(self):
        self.lexer.match_token(Keyword("while"))
        c = self.parse_expr()
        self.lexer.match_token(Keyword("do"))
        b = self.parse_expr()
        self.lexer.match_token("done")
        return While(c, b)

    def parse_let(self):
        self.lexer.match_token(Keyword("let"))
        v = self.parse_variable()
        self.lexer.match_token(Operator("="))
        e1 = self.parse_expr()
        self.lexer.match_token(Keyword("in"))
        e2 = self.parse_expr()
        self.lexer.match_token(Keyword("end"))
        return Let(v, e1, e2)

    def parse_letmut(self):
        self.lexer.match_token(Keyword("letmut"))
        v = self.parse_variable()
        self.lexer.match_token(Operator("="))
        e1 = self.parse_expr()
        self.lexer.match_token(Keyword("in"))
        e2 = self.parse_expr()
        self.lexer.match_token(Keyword("end"))
        return LetMut(v, e1, e2)

    def parse_seq(self):
        self.lexer.match_token(Keyword("begin"))
        seq = []
        while True:
            seq.append(self.parse_expr())
            if self.lexer.peek_token() != Operator(";"):
                break
        self.lexer.match_token(Keyword("end"))
        return Seq(seq)

    def parse_atom(self):
        match self.lexer.peek_token():
            case NumToken(what):
                return NumLiteral(what)
            case BoolToken(what):
                return BoolLiteral(what)
            case StrToken(what):
                return StrLiteral(what)
            case Punctuation("("):
                e = self.parse_expr()
                self.lexer.match_token(Punctuation(")"))
                return e

    def parse_prefix(self):
        prefix = "+-!"
        match self.lexer.peek_token():
            case Operator(op) if op in prefix:
                e = self.parse_prefix()
                return UnOp(op, e)
            case _:
                return self.parse_atom()

    def parse_exp(self):
        left = self.parse_prefix()
        match self.lexer.peek_token():
            case Operator("^"):
                right = self.parse_exp()
                return BinOp("^", left, right)
            case _:
                return left

    def parse_mul(self):
        mul = "× / quot rem".split()
        left = self.parse_exp()
        while True:
            match self.lexer.peek_token():
                case Operator(op) if op in mul:
                    right = self.parse_exp()
                    left = BinOp(op, left, right)
                case _:
                    return left

    def parse_add(self):
        add = "+-~"
        left = self.parse_mul()
        while True:
            match self.lexer.peek_token():
                case Operator(op) if op in add:
                    right = self.parse_mul()
                    left = BinOp(op, left, right)
                case _:
                    return left

    def parse_cmp(self):
        cmp = "<>≤≥"
        left = self.parse_add()
        match self.lexer.peek_token():
            case Operator(op) if op in cmp:
                right = self.parse_add()
                return BinOp(op, left, right)
            case _:
                return left

    def parse_logical_not(self):
        match self.lexer.peek_token():
            case Operator("not" as op):
                e = self.parse_logical_not()
                return UnOp(op, e)
            case _:
                return self.parse_cmp()

    def parse_logical_and(self):
        left = self.parse_logical_not()
        while True:
            match self.lexer.peek_token():
                case Operator("and" as op):
                    right = self.parse_logical_not()
                    left = BinOp(op, left, right)
                case _:
                    return left

    def parse_logical_or(self):
        left = self.parse_logical_and()
        while True:
            match self.lexer.peek_token():
                case Operator("or" as op):
                    right = self.parse_logical_and()
                    left = BinOp(op, left, right)
                case _:
                    return left

    def parse_eq(self):
        eq = "=≠"
        left = self.parse_logical_or()
        match self.lexer.peek_token():
            case Operator(op) if op in eq:
                right = self.parse_logical_or()
                return BinOp(op, left, right)
            case _:
                return left

    def parse_put(self):
        left = self.parse_eq()
        match self.lexer.peek_token():
            case Operator("←"):
                right = self.parse_eq()
                return Put(left, right)
            case _:
                return left

    def parse_simple():
        return self.parse_put()

    def parse_expr(self):
        match self.lexer.peek_token():
            case Keyword("if"):
                return self.parse_if()
            case Keyword("while"):
                return self.parse_while()
            case Keyword("let"):
                return self.parse_let()
            case Keyword("letmut"):
                return self.parse_letmut()
            case Keyword("begin"):
                return self.parse_seq()
            case _:
                return self.parse_simple()

@dataclass
class FractionType:
    pass

@dataclass
class BoolType:
    pass

@dataclass
class StrType:
    pass

@dataclass
class UnitType:
    pass

@dataclass
class RefType:
    base: 'SimType'

SimType = FractionType | BoolType | StrType | UnitType | RefType

@dataclass
class NumLiteral:
    value: Fraction
    type: SimType = FractionType()

@dataclass
class BoolLiteral:
    value: bool
    type: SimType = BoolType()

@dataclass
class StrLiteral:
    value: str
    type: SimType = StrType()

@dataclass
class Unit:
    type: SimType = UnitType()
    pass

@dataclass
class UnOp:
    operator: str
    operand: 'AST'
    type: SimType

@dataclass
class BinOp:
    operator: str
    left: 'AST'
    right: 'AST'
    type: Optional[SimType] = None

@dataclass
class Variable:
    name: str
    type: Optional[SimType] = None

@dataclass
class Let:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'
    type: Optional['AST'] = None

@dataclass
class IfElse:
    condition: 'AST'
    iftrue: 'AST'
    iffalse: 'AST'
    type: Optional[SimType] = None

@dataclass
class Seq:
    things: List['AST']
    type: Optional[SimType] = None

@dataclass
class LetMut:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'
    type: Optional[SimType] = None

@dataclass
class Get:
    var: 'AST'
    type: Optional[SimType] = None

@dataclass
class Put:
    var: 'AST'
    val: 'AST'
    type: Optional[SimType] = None

@dataclass
class While:
    condition: 'AST'
    body: 'AST'
    type: Optional[SimType] = None

AST = NumLiteral \
    | BoolLiteral \
    | StrLiteral \
    | UnOp \
    | BinOp \
    | Variable \
    | Let \
    | IfElse \
    | Seq \
    | LetMut \
    | Get \
    | Put \
    | While

Value = Fraction | bool | str | None

class InvalidProgram(Exception):
    pass

T = TypeVar('T')
Env = MutableMapping[str, T]
@dataclass
class EnvironmentType(MutableMapping[str, T]):
    envs: List[Env]

    def __init__(self):
        self.envs = [{}]

    def begin_scope(self):
        self.envs.append({})

    def end_scope(self):
        self.envs.pop()

    def __getitem__(self, k):
        for env in reversed(self.envs):
            try:
                x = env[k]
                return x
            except:
                pass

    def __setitem__(self, k, v):
        for env in reversed(self.envs):
            if k in env:
                env[k] = v
                return
        self.envs[-1][k] = v

    def __delitem__(self, k):
        for env in reversed(self.envs):
            if k in env:
                del env[k]

    def __iter__(self):
        from itertools import chain
        return chain.from_iterable(reversed(self.envs))

    def __len__(self):
        from itertools import accumulate
        return accumulate(map(len, self.envs))

arithmetic_ops = [ "+", "-", "*", "/" ]
cmp_ops        = [ "<", ">", "≤", "≥" ]
eq_ops         = [ "=", "≠" ]
sc_ops         = [ "&", "|" ]

def cmptype(t: SimType):
    return t in [FractionType(), StrType()]

def typecheck (
        program: AST,
        environment: EnvironmentType[SimType] = None
):
    if environment is None:
        environment = EnvironmentType[SimType]()

    def typecheck_(p: AST):
        return typecheck(p, environment)

    match program:
        case NumLiteral(value):
            return NumLiteral(value, FractionType())
        case BoolLiteral(value):
            return BoolLiteral(value, BoolType())
        case StrLiteral(value):
            return StrLiteral(value, StrType())
        case Unit():
            return Unit(UnitType())
        case Variable(name):
            if name not in environment:
                raise InvalidProgram()
            match environment[name]:
                case RefType(_):
                    raise InvalidProgram()
                case t:
                    return Variable(name, t)
        case UnOp(op, v):
            tv = typecheck_(v)
            match op:
                case "+" | "-":
                    if tv.type != FractionType():
                        raise InvalidProgram()
                    return UnOp(op, v, FractionType())
                case "~":
                    if tv.type != BoolType():
                        raise InvalidProgram()
                    return UnOp(op, v, BoolType())
        case BinOp("~", left, right):
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != StrType() or tright.type != StrType():
                raise InvalidProgram()
            return BinOp("~", tleft, tright, StrType())
        case BinOp(op, left, right) if op in arithmetic_ops:
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != FractionType() or tright.type != FractionType():
                raise InvalidProgram()
            return BinOp(op, tleft, tright, FractionType())
        case BinOp(op, left, right) if op in cmp_ops:
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != tright.type:
                raise InvalidProgram()
            if not cmptype(tleft.type):
                raise InvalidProgram()
            return BinOp(op, tleft, tright, BoolType())
        case BinOp(op, left, right) if op in eq_ops:
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != tright.type:
                raise InvalidProgram()
            return BinOp(op, tleft, tright, BoolType())
        case BinOp(op, left, right) if op in sc_ops:
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != BoolType() or tright.type != BoolType():
                raise InvalidProgram()
            return BinOp(op, tleft, tright, BoolType())
        case Let(Variable(name), e1, e2):
            te1 = typecheck_(e1)
            environment.begin_scope()
            environment[name] = te1.type
            te2 = typecheck_(e2)
            r = Let(Variable(name, te1.type), te1, te2, te2.type)
            environment.end_scope()
            return r
        case LetMut(Variable(name), e1, e2):
            te1 = typecheck_(e1)
            environment.begin_scope()
            environment[name] = RefType(te1.type)
            te2 = typecheck_(e2)
            r = LetMut(Variable(name, RefType(te1.type)), te1, te2, te2.type)
            environment.end_scope()
            return r
        case IfElse(condition, iftrue, iffalse):
            tc = typecheck_(condition)
            if tc.type != BoolType():
                raise InvalidProgram()
            tt = typecheck_(iftrue)
            tf = typecheck_(iffalse)
            if tt.type == tf.type:
                return IfElse(tc, tt, tf, tt.type)
            return IfElse(tc, tt, tf, UnitType())
        case Seq(things):
            tthings = list(map(lambda t: typecheck_(t), things))
            return Seq(tthings, tthings[-1].type)
        case Get(Variable(name)):
            if name not in environment:
                raise InvalidProgram()
            match environment[name]:
                case RefType(base) as type:
                    return Get(Variable(name, type), base)
                case _:
                    raise InvalidProgram()
        case Put(Variable(name), val):
            if name not in environment:
                raise InvalidProgram()
            tname = environment[name]
            tval = typecheck_(val)
            match tname:
                case RefType(base):
                    if base != tval.type:
                        raise InvalidProgram()
                    return Put(Variable(name, tname), tval, UnitType())
                case _:
                    raise InvalidProgram()
        case While(cond, body):
            tc = typecheck_(cond)
            if tc.type != BoolType():
                raise InvalidProgram()
            tb = typecheck_(body)
            return While(tc, tb, UnitType())
        case _:
            raise InvalidProgram()

def test_typecheck():
    e1 = NumLiteral(2)
    te1 = typecheck(e1)
    assert te1.type == FractionType()
    e2 = NumLiteral(3)
    e3 = BinOp("+", e1, e2)
    te3 = typecheck(e3)
    assert te3.type == FractionType()

def eval (
        program: AST,
        environment: EnvironmentType[Value] = None
) -> Value:
    if environment is None:
        environment = EnvironmentType[Value]()

    def eval_(p):
        return eval(p, environment)

    match program:
        case NumLiteral(value):
            return value
        case BoolLiteral(value):
            return value
        case StrLiteral(value):
            return value
        case Unit():
            return None
        case Variable(name) | Get(Variable(name)):
            if name not in environment:
                raise InvalidProgram()
            return environment[name]
        case UnOp("+", e):
            return eval_(e)
        case UnOp("-", e):
            return -eval_(e)
        case UnOp("~", e):
            return not eval_(e)
        case BinOp("~", left, right):
            return eval_(left) + eval_(right)
        case BinOp("&", left, right):
            v1 = eval_(left)
            if v1:
                v2 = eval_(right)
                return v2
        case BinOp("|", left, right):
            v1 = eval_(left)
            if not v1:
                v2 = eval_(right)
                return v2
        case BinOp(op, left, right):
            v1 = eval_(left)
            v2 = eval_(right)
            match op:
                case "+": return v1 + v2
                case "-": return v1 - v2
                case "*": return v1 * v2
                case "/":
                    if v2 == 0:
                        raise InvalidProgram()
                    return v1 / v2
                case "<": return v1 < v2
                case ">": return v1 > v2
                case "≤": return v1 <= v2
                case "≥": return v1 >= v2
                case "=": return v1 == v2
                case "≠": return v1 != v2
                case _: raise InvalidProgram()
        case IfElse(cond, iftrue, iffalse):
            vc = eval_(cond)
            if vc:
                return eval_(iftrue)
            else:
                return eval_(iffalse)
        case Let(Variable(name), e1, e2) | LetMut(Variable(name), e1, e2):
            ve1 = eval_(e1)
            environment.begin_scope()
            environment[name] = ve1
            ve2 = eval_(e2)
            environment.end_scope()
            return ve2
        case Put(Variable(name), e):
            environment[name] = eval_(e)
            return None
        case Seq(things):
            v = None
            for t in things:
                v = eval_(t)
            return v
        case While(cond, body):
            while eval_(cond):
                eval_(body)
            return None
        case _:
            raise InvalidProgram()
    raise InvalidProgram()

def test_fact():
    p = Variable("p")
    i = Variable("i")
    e = Put(p, BinOp("*", Get(p), Get(i)))
    f = Put(i, BinOp("+", Get(i), NumLiteral(1)))
    g = LetMut (p, NumLiteral(1),
        LetMut (i, NumLiteral(1),
            Seq([
                While (
                    BinOp("≤", Get(i), NumLiteral(10)),
                    Seq([e, f])
                ),
                Get(p)
            ])))
    assert eval(typecheck(g)) == 1*2*3*4*5*6*7*8*9*10

def test_mut_immut():
    import pytest
    # Trying to mutate an immutable variable.
    p = Variable("p")
    h = Let(p, NumLiteral(0), Put(p, BinOp("+", Get(p), 1)))
    with pytest.raises(InvalidProgram):
        typecheck(h)

def test_fib():
    f0 = Variable("f0")
    f1 = Variable("f1")
    i  = Variable("i")
    t  = Variable("t")
    step1 = Put(f1, BinOp("+", Get(f0), Get(f1)))
    step2 = Put(f0, t)
    step3 = Put(i, BinOp("+", Get(i), NumLiteral(1)))
    body = Let(t, Get(f1), Seq([step1, step2, step3]))
    check = BinOp("≤", Get(i), NumLiteral(10))
    loop  = While(check, body)
    e = LetMut(f0, NumLiteral(0),
        LetMut(f1, NumLiteral(1),
        LetMut(i, NumLiteral(1), Seq([loop, Get(f1)]))))
    assert eval(typecheck(e)) == 89
