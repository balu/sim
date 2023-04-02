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

    def from_file(filename):
        return Stream(open(filename).read())

    def get_char(self):
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
class UnitToken:
    pass

@dataclass
class Keyword:
    what: str

@dataclass
class Operator:
    what: str

@dataclass
class Identifier:
    what: str

@dataclass
class Punctuation:
    what: str

@dataclass
class EndToken:
    pass

Token = (
      NumToken
    | BoolToken
    | StrToken
    | UnitToken
    | Keyword
    | Operator
    | Identifier
    | Punctuation
    | EndToken
)

whitespace = " \t\n"
symbol_operators = "+ - * / ^ < > ≥ ≤ = ≠ ~ ← ! ; :".split()
punctuation = "( )".split()
keywords = "if then else end while do done begin let mut fun in num str bool unit".split()
word_operators = "and or not quot rem".split()


def wordToken(word: str) -> Token:
    if word in keywords:
        return Keyword(word)
    if word in word_operators:
        return Operator(word)
    if word == "True" or word == "False":
        return BoolToken(word)
    if word == "pass":
        return UnitToken()
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
                case c if c in punctuation:
                    return Punctuation(c)
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
                        c = self.stream.get_char()
                        while c.isalpha() or c.isdigit():
                            s += c
                            c = self.stream.get_char()
                        self.stream.unget_char()
                        return wordToken(s)
                    except EndOfStream:
                        return wordToken(s)
        except EndOfStream:
            return EndToken()

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

class ParseError(Exception):
    pass

@dataclass
class Parser:
    lexer: Lexer

    def from_lexer(lexer):
        return Parser(lexer)

    def parse_if(self):
        self.lexer.match_token(Keyword("if"))
        c = self.parse_expr()
        self.lexer.match_token(Keyword("then"))
        t = self.parse_seqexpr()
        match self.lexer.peek_token():
            case Keyword("else"):
                self.lexer.advance()
                f = self.parse_seqexpr()
                self.lexer.match_token(Keyword("end"))
                return IfElse(c, t, f)
            case Keyword("end"):
                self.lexer.advance()
                return IfElse(c, t, UnitLiteral())

    def parse_while(self):
        self.lexer.match_token(Keyword("while"))
        c = self.parse_expr()
        self.lexer.match_token(Keyword("do"))
        b = self.parse_seqexpr()
        self.lexer.match_token(Keyword("done"))
        return While(c, b)

    def parse_let(self):
        self.lexer.match_token(Keyword("let"))
        match self.lexer.peek_token():
            case Keyword("mut"): return self.parse_mut()
            case Keyword("fun"): return self.parse_fun()
            case _: return self.parse_plain_let()

    def parse_plain_let(self):
        v = self.parse_variable()
        self.lexer.match_token(Operator("="))
        e1 = self.parse_expr()
        self.lexer.match_token(Keyword("in"))
        e2 = self.parse_seqexpr()
        self.lexer.match_token(Keyword("end"))
        return Let(v, e1, e2)

    def parse_mut(self):
        self.lexer.match_token(Keyword("mut"))
        v = self.parse_variable()
        self.lexer.match_token(Operator("="))
        e1 = self.parse_expr()
        self.lexer.match_token(Keyword("in"))
        e2 = self.parse_seqexpr()
        self.lexer.match_token(Keyword("end"))
        return LetMut(v, e1, e2)

    def parse_param(self):
            v = self.parse_variable()
            self.lexer.match_token(Operator(":"))
            t = self.parse_type()
            return TypeAssertion(v, t)

    def parse_params(self):
        params = []
        self.lexer.match_token(Punctuation("("))
        match self.lexer.peek_token():
            case Punctuation(")"):
                self.lexer.advance()
                return params
            case _:
                while True:
                    params.append(self.parse_param())
                    match self.lexer.peek_token():
                        case Punctuation(","):
                            self.lexer.advance()
                        case Punctuation(")"):
                            self.lexer.advance()
                            return params

    def parse_type(self):
        match self.lexer.peek_token():
            case Keyword("num"):
                self.lexer.advance()
                return FractionType()
            case Keyword("bool"):
                self.lexer.advance()
                return BoolType()
            case Keyword("str"):
                self.lexer.advance()
                return StrType()
            case Keyword("unit"):
                self.lexer.advance()
                return UnitType()

    def parse_fun(self):
        self.lexer.match_token(Keyword("fun"))
        v = self.parse_variable()
        params = self.parse_params()
        self.lexer.match_token(Operator(":"))
        rettype = self.parse_type()
        self.lexer.match_token(Operator("="))
        body = self.parse_expr()
        self.lexer.match_token(Keyword("in"))
        expr = self.parse_expr()
        self.lexer.match_token(Keyword("end"))
        return LetFun(v, params, rettype, body, expr)

    def parse_seqexpr(self):
        seq = []
        while True:
            seq.append(self.parse_expr())
            if self.lexer.peek_token() != Operator(";"):
                break
            self.lexer.advance()
        return Seq(seq)

    def parse_seq(self):
        self.lexer.match_token(Keyword("begin"))
        seq = self.parse_seqexpr()
        self.lexer.match_token(Keyword("end"))
        return Seq(seq)

    def parse_variable(self):
        match self.lexer.peek_token():
            case Identifier(name):
                self.lexer.advance()
                return Variable(name)
            case _:
                raise ParseError()

    def parse_atom(self):
        match self.lexer.peek_token():
            case Identifier(name):
                return self.parse_variable()
            case NumToken(what):
                self.lexer.advance()
                return NumLiteral(int(what))
            case BoolToken(what):
                self.lexer.advance()
                match what:
                    case "True": return BoolLiteral(True)
                    case "False": return BoolLiteral(False)
            case StrToken(what):
                self.lexer.advance()
                return StrLiteral(what)
            case UnitToken():
                self.lexer.advance()
                return UnitLiteral()
            case Punctuation("("):
                self.lexer.advance()
                e = self.parse_expr()
                self.lexer.match_token(Punctuation(")"))
                return e
            case _:
                raise ParseError()

    def parse_call(self):
        fn = self.parse_atom()
        while True:
            match self.lexer.peek_token():
                case Punctuation("("):
                    self.lexer.advance()
                    args = []
                    while True:
                        match self.lexer.peek_token():
                            case Punctuation(")"):
                                self.lexer.advance()
                                break
                            case _:
                                while True:
                                    args.append(self.parse_expr())
                                    match self.lexer.peek_token():
                                        case Punctuation(","):
                                            self.lexer.advance()
                                        case _:
                                            break
                    return FunCall(fn, args)
                case _:
                    return fn

    def parse_prefix(self):
        prefix = "+-!"
        match self.lexer.peek_token():
            case Operator(op) if op in prefix:
                self.lexer.advance()
                e = self.parse_prefix()
                return UnOp(op, e)
            case _:
                return self.parse_call()

    def parse_exp(self):
        left = self.parse_prefix()
        match self.lexer.peek_token():
            case Operator("^"):
                self.lexer.advance()
                right = self.parse_exp()
                return BinOp("^", left, right)
            case _:
                return left

    def parse_mul(self):
        mul = "* / quot rem".split()
        left = self.parse_exp()
        while True:
            match self.lexer.peek_token():
                case Operator(op) if op in mul:
                    self.lexer.advance()
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
                    self.lexer.advance()
                    right = self.parse_mul()
                    left = BinOp(op, left, right)
                case _:
                    return left

    def parse_cmp(self):
        cmp = "<>≤≥"
        left = self.parse_add()
        match self.lexer.peek_token():
            case Operator(op) if op in cmp:
                self.lexer.advance()
                right = self.parse_add()
                return BinOp(op, left, right)
            case _:
                return left

    def parse_eq(self):
        eq = "=≠"
        left = self.parse_cmp()
        match self.lexer.peek_token():
            case Operator(op) if op in eq:
                self.lexer.advance()
                right = self.parse_cmp()
                return BinOp(op, left, right)
            case _:
                return left

    def parse_logical_not(self):
        match self.lexer.peek_token():
            case Operator("not" as op):
                self.lexer.advance()
                e = self.parse_logical_not()
                return UnOp(op, e)
            case _:
                return self.parse_eq()

    def parse_logical_and(self):
        left = self.parse_logical_not()
        while True:
            match self.lexer.peek_token():
                case Operator("and" as op):
                    self.lexer.advance()
                    right = self.parse_logical_not()
                    left = BinOp(op, left, right)
                case _:
                    return left

    def parse_logical_or(self):
        left = self.parse_logical_and()
        while True:
            match self.lexer.peek_token():
                case Operator("or" as op):
                    self.lexer.advance()
                    right = self.parse_logical_and()
                    left = BinOp(op, left, right)
                case _:
                    return left

    def parse_type_assert(self):
        expr = self.parse_logical_or()
        match self.lexer.peek_token():
            case Operator(":"):
                self.lexer.advance()
                type = self.parse_type()
                return TypeAssertion(expr, type)
            case _:
                return expr

    def parse_put(self):
        left = self.parse_type_assert()
        match self.lexer.peek_token():
            case Operator("←"):
                self.lexer.advance()
                right = self.parse_type_assert()
                return Put(left, right)
            case _:
                return left

    def parse_simple(self):
        return self.parse_put()

    def parse_expr(self):
        match self.lexer.peek_token():
            case Keyword("if"):
                return self.parse_if()
            case Keyword("while"):
                return self.parse_while()
            case Keyword("let"):
                return self.parse_let()
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

@dataclass
class FnType:
    params: List['SimType']
    ret: 'SIMType'

SimType = FractionType | BoolType | StrType | UnitType | RefType | FnType

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
class UnitLiteral:
    type: SimType = UnitType()
    pass

@dataclass
class UnOp:
    operator: str
    operand: 'AST'
    type: Optional[SimType] = None

@dataclass
class BinOp:
    operator: str
    left: 'AST'
    right: 'AST'
    type: Optional[SimType] = None

class Variable:
    __match_args__ = ("name",)

    name: str
    id: int
    fdepth: int
    localID: int
    type: Optional[SimType] = None

    def __init__(self, name, type = None):
        self.name = name
        self.type = type
        self.id = self.fdepth = self.localID = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f"{self.name}::{self.id}::{self.localID}"

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
class Put:
    var: 'AST'
    val: 'AST'
    type: Optional[SimType] = None

@dataclass
class While:
    condition: 'AST'
    body: 'AST'
    type: Optional[SimType] = None

@dataclass
class LetFun:
    name: 'AST'
    params: List['AST']
    rettype: 'AST'
    body: 'AST'
    expr: 'AST'
    type: Optional[SimType] = None

@dataclass
class FunCall:
    fn: 'AST'
    args: List['AST']
    type: Optional[SimType] = None

@dataclass
class TypeAssertion:
    expr: 'AST'
    type: SimType

@dataclass
class FunObject:
    params: List['AST']
    body: 'AST'

Value = Fraction | bool | str | None

AST = (
      NumLiteral
    | BoolLiteral
    | StrLiteral
    | UnitLiteral
    | UnOp
    | BinOp
    | Variable
    | Let
    | IfElse
    | Seq
    | LetMut
    | Put
    | While
    | LetFun
    | FunCall
    | TypeAssertion
)

class TokenError(Exception):
    pass

class ResolveError(Exception):
    pass

class TypeError(Exception):
    pass

class BUG(Exception):
    pass

class RunTimeError(Exception):
    pass

T = TypeVar('T')
U = TypeVar('U')
Env = MutableMapping[U, T]

@dataclass
class EnvironmentType(MutableMapping[U, T]):
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

arithmetic_ops = [ "+", "-", "*", "/", "quot", "rem" ]
cmp_ops        = [ "<", ">", "≤", "≥" ]
eq_ops         = [ "=", "≠" ]
sc_ops         = [ "and", "or" ]

def cmptype(t: SimType):
    return t in [FractionType(), StrType()]

class ResolveState:
    env: EnvironmentType[str, Variable]
    stk: List[List[int]]
    lastID: int

    def __init__(self):
        self.env = EnvironmentType()
        self.stk = [[0, -1]]
        self.lastID = -1

    def begin_fun(self):
        self.stk.append([0, -1])

    def end_fun(self):
        self.stk.pop()

    def handle_new(self, v):
        v.fdepth = len(self.stk) - 1
        v.id = self.lastID = self.lastID + 1
        v.localID = self.stk[-1][1] = self.stk[-1][1] + 1
        self.env[v.name] = v

    def begin_scope(self):
        self.env.begin_scope()

    def end_scope(self):
        self.env.end_scope()

def resolve (
        program: AST,
        rstate: ResolveState = None
) -> AST:
    if rstate is None:
        rstate = ResolveState()

    def resolve_(program):
        return resolve(program, rstate)

    match program:
        case NumLiteral() | BoolLiteral() | StrLiteral() | UnitLiteral():
            return program
        case UnOp(op, e):
            re = resolve_(e)
            return UnOp(op, re)
        case BinOp(op, left, right):
            rleft = resolve_(left)
            rright = resolve_(right)
            return BinOp(op, rleft, rright)
        case Variable(name):
            if name not in rstate.env:
                raise ResolveError()
            declared = rstate.env[name]
            return declared
        case Let(Variable(name) as v, e1, e2):
            re1 = resolve_(e1)
            rstate.begin_scope()
            rstate.handle_new(v)
            re2 = resolve_(e2)
            rstate.end_scope()
            return Let(v, re1, re2)
        case LetMut(Variable(name) as v, e1, e2):
            re1 = resolve_(e1)
            rstate.begin_scope()
            rstate.handle_new(v)
            re2 = resolve_(e2)
            rstate.end_scope()
            return LetMut(v, re1, re2)
        case IfElse(cond, true, false):
            rcond = resolve_(cond)
            rtrue = resolve_(true)
            rfalse = resolve_(false)
            return IfElse(rcond, rtrue, rfalse)
        case Seq(things):
            rthings = []
            for e in things:
                rthings.append(resolve_(e))
            return Seq(rthings)
        case Put(dest, src):
            rdest = resolve_(dest)
            rsrc = resolve_(src)
            return Put(rdest, rsrc)
        case While(cond, body):
            rcond = resolve_(cond)
            rbody = resolve_(body)
            return While(rcond, rbody)
        case LetFun(Variable(name) as f, params, rettype, body, expr):
            rstate.begin_scope() # Scope in which function exists.
            rstate.handle_new(f)
            rstate.begin_fun()
            rstate.begin_scope() # Function body scope.
            rparams = []
            for param in params:
                match param.expr:
                    case Variable(name) as v:
                        rstate.handle_new(v)
                        rparams.append(TypeAssertion(param.expr, param.type))
                    case _:
                        raise BUG()
            rbody = resolve_(body)
            rstate.end_scope() # Function body scope.
            rstate.end_fun()
            rexpr = resolve_(expr)
            rstate.end_scope() # Scope in which function exists.
            return LetFun(f, rparams, rettype, rbody, rexpr)
        case FunCall(Variable(_) as v, args):
            rv = resolve_(v)
            rargs = []
            for arg in args:
                rargs.append(resolve_(arg))
            return FunCall(rv, rargs)
        case TypeAssertion(expr, type):
            rexpr = resolve_(expr)
            return TypeAssertion(rexpr, type)
        case _:
            raise BUG()

def typecheck (
        program: AST,
        environment: EnvironmentType[Variable, SimType] = None
):
    if environment is None:
        environment = EnvironmentType()

    def typecheck_(p: AST):
        return typecheck(p, environment)

    match program:
        case NumLiteral(value):
            return NumLiteral(value, FractionType())
        case BoolLiteral(value):
            return BoolLiteral(value, BoolType())
        case StrLiteral(value):
            return StrLiteral(value, StrType())
        case UnitLiteral():
            return UnitLiteral(UnitType())
        case Variable() as v:
            match environment[v]:
                case RefType() | FnType():
                    raise TypeError()
                case t:
                    v.type = t
                    return v
        case UnOp("!", Variable() as v):
            if v not in environment:
                raise TypeError()
            match environment[v]:
                case RefType(base) as type:
                    return UnOp("!", v, base)
                case _:
                    raise TypeError()
        case UnOp("!", other):
            raise TypeError()
        case UnOp(op, v):
            tv = typecheck_(v)
            match op:
                case "+" | "-":
                    if tv.type != FractionType():
                        raise TypeError()
                    return UnOp(op, v, FractionType())
                case "not":
                    if tv.type != BoolType():
                        raise TypeError()
                    return UnOp(op, v, BoolType())
        case BinOp("~", left, right):
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != StrType() or tright.type != StrType():
                raise TypeError()
            return BinOp("~", tleft, tright, StrType())
        case BinOp(op, left, right) if op in arithmetic_ops:
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != FractionType() or tright.type != FractionType():
                raise TypeError()
            return BinOp(op, tleft, tright, FractionType())
        case BinOp(op, left, right) if op in cmp_ops:
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != tright.type:
                raise TypeError()
            if not cmptype(tleft.type):
                raise TypeError()
            return BinOp(op, tleft, tright, BoolType())
        case BinOp(op, left, right) if op in eq_ops:
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != tright.type:
                raise TypeError()
            return BinOp(op, tleft, tright, BoolType())
        case BinOp(op, left, right) if op in sc_ops:
            tleft = typecheck_(left)
            tright = typecheck_(right)
            if tleft.type != BoolType() or tright.type != BoolType():
                raise TypeError()
            return BinOp(op, tleft, tright, BoolType())
        case Let(Variable() as v, e1, e2):
            te1 = typecheck_(e1)
            v.type = te1.type
            environment.begin_scope()
            environment[v] = te1.type
            te2 = typecheck_(e2)
            r = Let(v, te1, te2, te2.type)
            environment.end_scope()
            return r
        case LetMut(Variable() as v, e1, e2):
            te1 = typecheck_(e1)
            v.type = RefType(te1.type)
            environment.begin_scope()
            environment[v] = v.type
            te2 = typecheck_(e2)
            r = LetMut(v, te1, te2, te2.type)
            environment.end_scope()
            return r
        case IfElse(condition, iftrue, iffalse):
            tc = typecheck_(condition)
            if tc.type != BoolType():
                raise TypeError()
            tt = typecheck_(iftrue)
            tf = typecheck_(iffalse)
            if tt.type == tf.type:
                return IfElse(tc, tt, tf, tt.type)
            return IfElse(tc, tt, tf, UnitType())
        case Seq(things):
            tthings = list(map(lambda t: typecheck_(t), things))
            return Seq(tthings, tthings[-1].type)
        case Put(Variable() as v, val):
            if v not in environment:
                raise TypeError()
            tv = environment[v]
            tval = typecheck_(val)
            match tv:
                case RefType(base):
                    if base != tval.type:
                        raise TypeError()
                    return Put(v, tval, UnitType())
                case _:
                    raise TypeError()
        case While(cond, body):
            tc = typecheck_(cond)
            if tc.type != BoolType():
                raise TypeError()
            tb = typecheck_(body)
            return While(tc, tb, UnitType())
        case LetFun(fnv, params, rettype, body, expr):
            tparams = []
            paramtypes = []
            paramvars = []
            for param in params:
                match param:
                    case TypeAssertion(Variable() as v, type):
                        v.type = type
                        paramvars.append(v)
                        paramtypes.append(type)
                        tparams.append(TypeAssertion(v, type))
                    case _:
                        raise BUG()
            environment.begin_scope()
            environment[fnv] = fnv.type = FnType(paramtypes, rettype)
            environment.begin_scope()
            for v, type in zip(paramvars, paramtypes):
                environment[v] = type
            tb = typecheck_(body)
            environment.end_scope()
            te = typecheck_(expr)
            environment.end_scope()
            return LetFun(fnv, tparams, rettype, tb, te, te.type)
        case FunCall(Variable(_) as v, args):
            v.type = ftype = environment[v]
            targs = []
            for ptype, arg in zip(ftype.params, args):
                targ = typecheck_(arg)
                if targ.type != ptype:
                    raise TypeError()
                targs.append(targ)
            return FunCall(v, targs, ftype.ret)
        case TypeAssertion(expr, type):
            texpr = typecheck_(expr)
            if expr.type != type:
                raise TypeError()
            return TypeAssertion(texpr, type, type)
        case _:
            raise TypeError()

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
        environment: EnvironmentType[Variable, Value] = None
) -> Value:
    if environment is None:
        environment = EnvironmentType()

    def eval_(p):
        return eval(p, environment)

    match program:
        case NumLiteral(value):
            return Fraction(int(value), 1)
        case BoolLiteral(value):
            return value
        case StrLiteral(value):
            return value
        case UnitLiteral():
            return None
        case Variable() as v:
            return environment[v]
        case UnOp("!", Variable() as v):
            return environment[v]
        case UnOp("+", e):
            return eval_(e)
        case UnOp("-", e):
            return -eval_(e)
        case UnOp("not", e):
            return not eval_(e)
        case BinOp("~", left, right):
            return eval_(left) + eval_(right)
        case BinOp("and", left, right):
            v1 = eval_(left)
            if v1:
                v2 = eval_(right)
                return v2
            return False
        case BinOp("or", left, right):
            v1 = eval_(left)
            if not v1:
                v2 = eval_(right)
                return v2
            return True
        case BinOp(op, left, right):
            v1 = eval_(left)
            v2 = eval_(right)
            match op:
                case "+": return v1 + v2
                case "-": return v1 - v2
                case "*": return v1 * v2
                case "/":
                    if v2 == 0:
                        raise RunTimeError()
                    return v1 / v2
                case "quot":
                    if v2 == 0:
                        raise RunTimeError()
                    iv1 = int(v1)
                    iv2 = int(v2)
                    return Fraction(iv1 // iv2)
                case "rem":
                    if v2 == 0:
                        raise RunTimeError()
                    iv1 = int(v1)
                    iv2 = int(v2)
                    return Fraction(iv1 % iv2)
                case "<": return v1 < v2
                case ">": return v1 > v2
                case "≤": return v1 <= v2
                case "≥": return v1 >= v2
                case "=": return v1 == v2
                case "≠": return v1 != v2
                case _: raise BUG()
        case IfElse(cond, iftrue, iffalse):
            vc = eval_(cond)
            if vc:
                return eval_(iftrue)
            else:
                return eval_(iffalse)
        case Let(Variable() as v, e1, e2) | LetMut(Variable() as v, e1, e2):
            ve1 = eval_(e1)
            environment.begin_scope()
            environment[v] = ve1
            ve2 = eval_(e2)
            environment.end_scope()
            return ve2
        case Put(Variable() as v, e):
            environment[v] = eval_(e)
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
        case LetFun(v, params, _, body, expr):
            environment.begin_scope()
            paramvars = list(map(lambda p: p.expr, params))
            environment[v] = FunObject(paramvars, body)
            ve = eval_(expr)
            environment.end_scope()
            return ve
        case FunCall(v, args):
            fn = environment[v]
            vargs = list(map(lambda arg: eval_(arg), args))
            environment.begin_scope()
            for param, varg in zip(fn.params, vargs):
                environment[param] = varg
            vresult = eval_(fn.body)
            environment.end_scope()
            return vresult
        case TypeAssertion(expr):
            return eval_(expr)
        case _:
            raise BUG()
    raise BUG()

@dataclass
class Label:
    target: int

class I:
    """The instructions for our stack VM."""
    @dataclass
    class PUSH:
        what: Value

    @dataclass
    class UMINUS:
        pass

    @dataclass
    class ADD:
        pass

    @dataclass
    class SUB:
        pass

    @dataclass
    class MUL:
        pass

    @dataclass
    class DIV:
        pass

    @dataclass
    class QUOT:
        pass

    @dataclass
    class REM:
        pass

    @dataclass
    class EXP:
        pass

    @dataclass
    class EQ:
        pass

    @dataclass
    class NEQ:
        pass

    @dataclass
    class LT:
        pass

    @dataclass
    class GT:
        pass

    @dataclass
    class LE:
        pass

    @dataclass
    class GE:
        pass

    @dataclass
    class JMP:
        label: Label

    @dataclass
    class JMP_IF_FALSE:
        label: Label

    @dataclass
    class JMP_IF_TRUE:
        label: Label

    @dataclass
    class NOT:
        pass

    @dataclass
    class DUP:
        pass

    @dataclass
    class POP:
        pass

    @dataclass
    class LOAD:
        localID: int

    @dataclass
    class STORE:
        localID: int

    @dataclass
    class HALT:
        pass

Instruction = (
      I.PUSH
    | I.ADD
    | I.SUB
    | I.MUL
    | I.DIV
    | I.QUOT
    | I.REM
    | I.NOT
    | I.UMINUS
    | I.JMP
    | I.JMP_IF_FALSE
    | I.JMP_IF_TRUE
    | I.DUP
    | I.POP
    | I.HALT
    | I.EQ
    | I.NEQ
    | I.LT
    | I.GT
    | I.LE
    | I.GE
    | I.LOAD
    | I.STORE
)

@dataclass
class ByteCode:
    insns: List[Instruction]

    def __init__(self):
        self.insns = []

    def label(self):
        return Label(-1)

    def emit(self, instruction):
        self.insns.append(instruction)

    def emit_label(self, label):
        label.target = len(self.insns)

def print_bytecode(code: ByteCode):
    for i, insn in enumerate(code.insns):
        match insn:
            case I.JMP(Label(offset)) | I.JMP_IF_TRUE(Label(offset)) | I.JMP_IF_FALSE(Label(offset)):
                print(f"{i:=4} {insn.__class__.__name__:<15} {offset}")
            case I.LOAD(localID) | I.STORE(localID):
                print(f"{i:=4} {insn.__class__.__name__:<15} {localID}")
            case I.PUSH(value):
                print(f"{i:=4} {'PUSH':<15} {value}")
            case _:
                print(f"{i:=4} {insn.__class__.__name__:<15}")

class Frame:
    locals: List[Value]

    def __init__(self):
        MAX_LOCALS = 32
        self.locals = [None] * MAX_LOCALS

class VM:
    bytecode: ByteCode
    ip: int
    data: List[Value]
    currentFrame: Frame

    def load(self, bytecode):
        self.bytecode = bytecode
        self.restart()

    def restart(self):
        self.ip = 0
        self.data = []
        self.currentFrame = Frame()

    def execute(self) -> Value:
        while True:
            assert self.ip < len(self.bytecode.insns)
            match self.bytecode.insns[self.ip]:
                case I.PUSH(val):
                    self.data.append(val)
                    self.ip += 1
                case I.UMINUS():
                    op = self.data.pop()
                    self.data.append(-op)
                    self.ip += 1
                case I.ADD():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left+right)
                    self.ip += 1
                case I.SUB():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left-right)
                    self.ip += 1
                case I.MUL():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left*right)
                    self.ip += 1
                case I.DIV():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left/right)
                    self.ip += 1
                case I.EXP():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left**right)
                    self.ip += 1
                case I.QUOT():
                    right = self.data.pop()
                    left = self.data.pop()
                    if left.denominator != 1 or right.denominator != 1:
                        raise RunTimeError()
                    left, right = int(left), int(right)
                    self.data.append(Fraction(left // right, 1))
                    self.ip += 1
                case I.REM():
                    right = self.data.pop()
                    left = self.data.pop()
                    if left.denominator != 1 or right.denominator != 1:
                        raise RunTimeError()
                    left, right = int(left), int(right)
                    self.data.append(Fraction(left % right, 1))
                    self.ip += 1
                case I.EQ():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left==right)
                    self.ip += 1
                case I.NEQ():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left!=right)
                    self.ip += 1
                case I.LT():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left<right)
                    self.ip += 1
                case I.GT():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left>right)
                    self.ip += 1
                case I.LE():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left<=right)
                    self.ip += 1
                case I.GE():
                    right = self.data.pop()
                    left = self.data.pop()
                    self.data.append(left>=right)
                    self.ip += 1
                case I.JMP(label):
                    self.ip = label.target
                case I.JMP_IF_FALSE(label):
                    op = self.data.pop()
                    if not op:
                        self.ip = label.target
                    else:
                        self.ip += 1
                case I.JMP_IF_TRUE(label):
                    op = self.data.pop()
                    if op:
                        self.ip = label.target
                    else:
                        self.ip += 1
                case I.NOT():
                    op = self.data.pop()
                    self.data.append(not op)
                    self.ip += 1
                case I.DUP():
                    op = self.data.pop()
                    self.data.append(op)
                    self.data.append(op)
                    self.ip += 1
                case I.POP():
                    self.data.pop()
                    self.ip += 1
                case I.LOAD(localID):
                    self.data.append(self.currentFrame.locals[localID])
                    self.ip += 1
                case I.STORE(localID):
                    v = self.data.pop()
                    self.currentFrame.locals[localID] = v
                    self.ip += 1
                case I.HALT():
                    return self.data.pop()

def codegen(program: AST) -> ByteCode:
    code = ByteCode()
    do_codegen(program, code)
    code.emit(I.HALT())
    return code

def do_codegen (
        program: AST,
        code: ByteCode
) -> None:
    def codegen_(program):
        do_codegen(program, code)

    simple_ops = {
        "+": I.ADD(),
        "-": I.SUB(),
        "*": I.MUL(),
        "/": I.DIV(),
        "quot": I.QUOT(),
        "rem": I.REM(),
        "<": I.LT(),
        ">": I.GT(),
        "≤": I.LE(),
        "≥": I.GE(),
        "=": I.EQ(),
        "≠": I.NEQ(),
        "not": I.NOT()
    }

    match program:
        case NumLiteral(what) | BoolLiteral(what) | StrLiteral(what):
            code.emit(I.PUSH(what))
        case UnitLiteral():
            code.emit(I.PUSH(None))
        case BinOp(op, left, right) if op in simple_ops:
            codegen_(left)
            codegen_(right)
            code.emit(simple_ops[op])
        case BinOp("and", left, right):
            E = code.label()
            codegen_(left)
            code.emit(I.DUP())
            code.emit(I.JMP_IF_FALSE(E))
            code.emit(I.POP())
            codegen_(right)
            code.emit_label(E)
        case BinOp("or", left, right):
            E = code.label()
            codegen_(left)
            code.emit(I.DUP())
            code.emit(I.JMP_IF_TRUE(E))
            code.emit(I.POP())
            codegen_(right)
            code.emit_label(E)
        case UnOp("-", operand):
            codegen_(operand)
            code.emit(I.UMINUS())
        case Seq(things):
            for thing in things:
                codegen_(thing)
        case IfElse(cond, iftrue, iffalse):
            E = code.label()
            F = code.label()
            codegen_(cond)
            code.emit(I.JMP_IF_FALSE(F))
            codegen_(iftrue)
            code.emit(I.JMP(E))
            code.emit_label(F)
            codegen_(iffalse)
            code.emit_label(E)
        case While(cond, body):
            B = code.label()
            E = code.label()
            code.emit_label(B)
            codegen_(cond)
            code.emit(I.JMP_IF_FALSE(E))
            codegen_(body)
            code.emit(I.JMP(B))
            code.emit_label(E)
        case (Variable() as v) | UnOp("!", Variable() as v):
            code.emit(I.LOAD(v.localID))
        case Put(Variable() as v, e):
            codegen_(e)
            code.emit(I.STORE(v.localID))
        case Let(Variable() as v, e1, e2) | LetMut(Variable() as v, e1, e2):
            codegen_(e1)
            code.emit(I.STORE(v.localID))
            codegen_(e2)
        case TypeAssertion(expr, _):
            codegen_(expr)

def parse_string(s):
    return Parser.from_lexer(Lexer.from_stream(Stream.from_string(s))).parse_expr()

def parse_file(f):
    return Parser.from_lexer(Lexer.from_stream(Stream.from_file(f))).parse_expr()

def verify(program):
    return typecheck(resolve(program))

def compile(program):
    return codegen(typecheck(resolve(program)))

def run(program):
    return eval(typecheck(resolve(program)))

def run_file(f):
    return run(parse_file(f))

def test_resolve():
    print(resolve(parse_string("let fun f(x : num) : num = let a = 1 in x + a end in f(0) end")))

def test_codegen():
    programs = {
        "(2+3)*5+6/2": 28,
        "if 2 > 3 then 5 else 6 end": 6,
        "if 3 > 2 then 5 else 6 end": 5,
        "5 > 7 and 6 > 5": False,
        "5 > 7 and 6 < 5": False,
        "5 < 7 or 6 > 5": True,
        "5 > 7 or 6 > 5": True,
        "5 > 7 or 5 > 5": False,
        "let x = 5 in x + x end": 10,
        "let x = 5 in let y = 6 in x*x + y*y + 2*x*y end end": 121
    }
    v = VM()
    for p, e in programs.items():
        v.load(compile(parse_string(p)))
        assert e == v.execute()

def print_codegen():
    print_bytecode(compile(parse_file("samples/euler2.sim")))

print_codegen()

def test_fact():
    p = Variable("p")
    i = Variable("i")
    e = Put(p, BinOp("*", UnOp("!", p), UnOp("!", i)))
    f = Put(i, BinOp("+", UnOp("!", i), NumLiteral(1)))
    g = LetMut (p, NumLiteral(1),
        LetMut (i, NumLiteral(1),
            Seq([
                While (
                    BinOp("≤", UnOp("!", i), NumLiteral(10)),
                    Seq([e, f])
                ),
                UnOp("!", p)
            ])))
    assert run(g) == 1*2*3*4*5*6*7*8*9*10

def test_mut_immut():
    import pytest
    # Trying to mutate an immutable variable.
    p = Variable("p")
    h = Let(p, NumLiteral(0), Put(p, BinOp("+", UnOp("!", p), NumLiteral(1))))
    with pytest.raises(TypeError):
        verify(h)

def test_fib():
    f0 = Variable("f0")
    f1 = Variable("f1")
    i  = Variable("i")
    t  = Variable("t")
    step1 = Put(f1, BinOp("+", UnOp("!", f0), UnOp("!", f1)))
    step2 = Put(f0, t)
    step3 = Put(i, BinOp("+", UnOp("!", i), NumLiteral(1)))
    body = Let(t, UnOp("!", f1), Seq([step1, step2, step3]))
    check = BinOp("≤", UnOp("!", i), NumLiteral(10))
    loop  = While(check, body)
    e = LetMut(f0, NumLiteral(0),
        LetMut(f1, NumLiteral(1),
        LetMut(i, NumLiteral(1), Seq([loop, UnOp("!", f1)]))))
    assert run(e) == 89

def test_fib_parse():
    assert run_file("samples/fib10.sim") == 89

def test_simple_resolve():
    assert 10 == run(parse_string("let mut a = 5 in !a + !a end"))

def test_euler1():
    def euler1():
        sum = 0
        for i in range(1, 1000):
            if i % 3 == 0 or i % 5 == 0:
                sum += i
        return sum

    assert euler1() == run_file("samples/euler1.sim")

def test_const_fn():
    assert 0 == run(parse_string("let fun f(): num = 0 in f() end"))

test_const_fn()

def test_id_fn():
    assert 3 == run(parse_string("let fun id(n: num): num = n in id(1) + id(2) end"))

def test_inc_fn():
    assert 3 == run(parse_string("let fun inc(n: num): num = n+1 in inc(0) + inc(1) end"))

def test_multi_param_fn():
    assert 10 == run(parse_string("let fun lin(a: num, x: num, b: num): num = a * x + b in lin(2, 3, 4) end"))

def test_recursive_fn():
    assert 3628800 == run(parse_string("let fun fact(n: num): num = if n = 0 then 1 else n * fact(n-1) end in fact(10) end"))

def test_euler2():
    def euler2():
        sum, f0, f1 = 0, 1, 2
        while f1 < 4000000:
            if f1 % 2 == 0: sum += f1
            f0, f1 = f1, f0+f1
        return sum

    assert euler2() == run_file("samples/euler2.sim")

def test_euler3():
    assert 7 == run_file("samples/euler3.sim")
