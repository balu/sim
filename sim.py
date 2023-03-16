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

Token = (
      NumToken
    | BoolToken
    | StrToken
    | UnitToken
    | Keyword
    | Operator
    | Identifier
    | Punctuation
)

whitespace = " \t\n"
symbol_operators = "+ - * / ^ < > ≥ ≤ = ≠ ~ ← ! ; :".split()
punctuation = "( )".split()
keywords = "if then else end while do done begin let letmut letfun in".split()
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
        self.lexer.match_token(Keyword("else"))
        f = self.parse_seqexpr()
        self.lexer.match_token(Keyword("end"))
        return IfElse(c, t, f)

    def parse_while(self):
        self.lexer.match_token(Keyword("while"))
        c = self.parse_expr()
        self.lexer.match_token(Keyword("do"))
        b = self.parse_seqexpr()
        self.lexer.match_token(Keyword("done"))
        return While(c, b)

    def parse_let(self):
        self.lexer.match_token(Keyword("let"))
        v = self.parse_variable()
        self.lexer.match_token(Operator("="))
        e1 = self.parse_expr()
        self.lexer.match_token(Keyword("in"))
        e2 = self.parse_seqexpr()
        self.lexer.match_token(Keyword("end"))
        return Let(v, e1, e2)

    def parse_letmut(self):
        self.lexer.match_token(Keyword("letmut"))
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
        return self.parse_variable()

    def parse_letfun(self):
        self.lexer.match_token(Keyword("letfun"))
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
                return Unit()
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
                type = self.parse_logical_or()
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
            case Keyword("letmut"):
                return self.parse_letmut()
            case Keyword("letfun"):
                return self.parse_letfun()
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
class Unit:
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

@dataclass
class Variable:
    name: str
    id: int
    type: Optional[SimType] = None

    def __init__(self, name, id = None, type = None):
        self.name = name
        if id is None:
            self.id = fresh()
        else:
            self.id = id
        self.type = type

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

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

unique_id = -1

def fresh():
    global unique_id
    return (unique_id := unique_id + 1)

@dataclass
class FunObject:
    params: List['AST']
    body: 'AST'

Value = Fraction | bool | str | None

AST = (
      NumLiteral
    | BoolLiteral
    | StrLiteral
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

builtin_variables = {
    #   name:  variable, type object, value object.
    "num": (Variable("num"), FractionType(), None),
    "bool": (Variable("bool"), BoolType(), None),
    "str": (Variable("str"), StrType(), None),
    "unit": (Variable("unit"), UnitType(), None),
}

def resolve (
        program: AST,
        environment: EnvironmentType[str, Variable] = None
) -> AST:
    if environment is None:
        environment = EnvironmentType()
        for k, (v, _, _) in builtin_variables.items():
            environment[k] = v

    def resolve_(program):
        return resolve(program, environment)

    match program:
        case NumLiteral() | BoolLiteral() | StrLiteral() | Unit():
            return program
        case UnOp(op, e):
            re = resolve_(e)
            return UnOp(op, re)
        case BinOp(op, left, right):
            rleft = resolve_(left)
            rright = resolve_(right)
            return BinOp(op, rleft, rright)
        case Variable(name):
            if name not in environment:
                raise ResolveError()
            declared = environment[name]
            return Variable(name, declared.id)
        case Let(Variable(name) as v, e1, e2):
            re1 = resolve_(e1)
            environment.begin_scope()
            environment[name] = v
            re2 = resolve_(e2)
            environment.end_scope()
            return Let(v, re1, re2)
        case LetMut(Variable(name) as v, e1, e2):
            re1 = resolve_(e1)
            environment.begin_scope()
            environment[name] = v
            re2 = resolve_(e2)
            environment.end_scope()
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
        case LetFun(Variable(name) as v, params, rettype, body, expr):
            environment.begin_scope()
            environment[name] = v
            environment.begin_scope()
            rparams = []
            for param in params:
                match param.expr:
                    case Variable(name):
                        environment[name] = param.expr
                    case _:
                        raise BUG()
                rparams.append(TypeAssertion(param.expr, resolve_(param.type)))
            rrettype = resolve_(rettype)
            rbody = resolve_(body)
            environment.end_scope()
            rexpr = resolve_(expr)
            return LetFun(v, rparams, rrettype, rbody, rexpr)
        case FunCall(Variable(_) as v, args):
            rv = resolve_(v)
            rargs = []
            for arg in args:
                rargs.append(resolve_(arg))
            return FunCall(rv, rargs)
        case TypeAssertion(expr, type):
            rexpr = resolve_(expr)
            rtype = resolve_(type)
            return TypeAssertion(rexpr, rtype)
        case _:
            raise BUG()

def typecheck (
        program: AST,
        environment: EnvironmentType[Variable, SimType] = None
):
    if environment is None:
        environment = EnvironmentType()
        for _, (v, t, _) in builtin_variables.items():
            environment[v] = t

    def typecheck_(p: AST):
        return typecheck(p, environment)

    def typecheck_type(type: AST):
        match type:
            case Variable() as v:
                if v not in environment:
                    raise TypeError
                return environment[v]
            case _:
                raise BUG()

    match program:
        case NumLiteral(value):
            return NumLiteral(value, FractionType())
        case BoolLiteral(value):
            return BoolLiteral(value, BoolType())
        case StrLiteral(value):
            return StrLiteral(value, StrType())
        case Unit():
            return Unit(UnitType())
        case Variable(name, id) as v:
            match environment[v]:
                case RefType() | FnType():
                    raise TypeError()
                case t:
                    return Variable(name, id, t)
        case UnOp("!", Variable(name, id) as v):
            if name not in environment:
                raise TypeError()
            match environment[v]:
                case RefType(base) as type:
                    return UnOp("!", Variable(name, id, type), base)
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
        case Let(Variable(name, id) as v, e1, e2):
            te1 = typecheck_(e1)
            environment.begin_scope()
            environment[v] = te1.type
            te2 = typecheck_(e2)
            r = Let(Variable(name, id, te1.type), te1, te2, te2.type)
            environment.end_scope()
            return r
        case LetMut(Variable(name, id) as v, e1, e2):
            te1 = typecheck_(e1)
            environment.begin_scope()
            environment[v] = RefType(te1.type)
            te2 = typecheck_(e2)
            r = LetMut(Variable(name, id, RefType(te1.type)), te1, te2, te2.type)
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
        case Put(Variable(name, id) as v, val):
            if v not in environment:
                raise TypeError()
            tv = environment[v]
            tval = typecheck_(val)
            match tv:
                case RefType(base):
                    if base != tval.type:
                        raise TypeError()
                    return Put(Variable(name, id, tv), tval, UnitType())
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
                        ttype = typecheck_type(type)
                        v.type = ttype
                        paramvars.append(v)
                        paramtypes.append(ttype)
                        tparams.append(TypeAssertion(v, ttype))
                    case _:
                        raise BUG()
            environment.begin_scope()
            trettype = typecheck_type(rettype)
            environment[fnv] = fnv.type = FnType(paramtypes, trettype)
            environment.begin_scope()
            for v, type in zip(paramvars, paramtypes):
                environment[v] = type
            tb = typecheck_(body)
            environment.end_scope()
            te = typecheck_(expr)
            environment.end_scope()
            return LetFun(fnv, tparams, trettype, tb, te, te.type)
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
        case Unit():
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
    assert eval(typecheck(g)) == 1*2*3*4*5*6*7*8*9*10

def test_mut_immut():
    import pytest
    # Trying to mutate an immutable variable.
    p = Variable("p")
    h = Let(p, NumLiteral(0), Put(p, BinOp("+", UnOp("!", p), 1)))
    with pytest.raises(TypeError):
        typecheck(h)

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
    assert eval(typecheck(e)) == 89

def test_fib_parse():
    e = Parser.from_lexer (
        Lexer.from_stream (
            Stream.from_file (
                "samples/fib10.sim"
            )
        )
    ).parse_expr()
    assert eval(typecheck(resolve(e))) == 89


def parse_string(s):
    return Parser.from_lexer(Lexer.from_stream(Stream.from_string(s))).parse_expr()


def run(e):
    return eval(typecheck(resolve(parse_string(e))))

def run_file(f):
    return run(open(f).read())

def test_simple_resolve():
    assert 10 == run("letmut a = 5 in !a + !a end")

def test_euler1():
    def euler1():
        sum = 0
        for i in range(1, 1000):
            if i % 3 == 0 or i % 5 == 0:
                sum += i
        return sum

    assert euler1() == run_file("samples/euler1.sim")

def test_const_fn():
    assert 0 == run("letfun f(): num = 0 in f() end")

def test_id_fn():
    assert 3 == run("letfun id(n: num): num = n in id(1) + id(2) end")

def test_inc_fn():
    assert 3 == run("letfun inc(n: num): num = n+1 in inc(0) + inc(1) end")

def test_multi_param_fn():
    assert 10 == run("letfun lin(a: num, x: num, b: num): num = a * x + b in lin(2, 3, 4) end")

def test_recursive_fn():
    assert 3628800 == run("letfun fact(n: num): num = if n = 0 then 1 else n * fact(n-1) end in fact(10) end")
