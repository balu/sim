from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Mapping, List

currentID = 0

def fresh():
    global currentID
    currentID = currentID + 1
    return currentID

@dataclass
class NumLiteral:
    value: Fraction
    def __init__(self, *args):
        self.value = Fraction(*args)

@dataclass
class BinOp:
    operator: str
    left: 'AST'
    right: 'AST'

@dataclass(frozen=True)
class Variable:
    name: str
    id: int

    def make(name):
        return Variable(name, fresh())

@dataclass
class Let:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'

@dataclass
class LetMut:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'

@dataclass
class Put:
    var: 'AST'
    e1: 'AST'

@dataclass
class Get:
    var: 'AST'

@dataclass
class Seq:
    things: List['AST']

@dataclass
class LetFun:
    name: 'AST'
    params: List['AST']
    body: 'AST'
    expr: 'AST'

@dataclass
class FunCall:
    fn: 'AST'
    args: List['AST']

AST = NumLiteral | BinOp | Variable | Let | LetMut | Put | Get | Seq | LetFun | FunCall

@dataclass
class FnObject:
    params: List['AST']
    body: 'AST'

Value = Fraction | FnObject

class InvalidProgram(Exception):
    pass

class Environment:
    envs: List

    def __init__(self):
        self.envs = [{}]

    def enter_scope(self):
        self.envs.append({})

    def exit_scope(self):
        assert self.envs
        self.envs.pop()

    def add(self, name, value):
        assert name not in self.envs[-1]
        self.envs[-1][name] = value

    def get(self, name):
        for env in reversed(self.envs):
            if name in env:
                return env[name]
        raise KeyError()

    def update(self, name, value):
        for env in reversed(self.envs):
            if name in env:
                env[name] = value
                return
        raise KeyError()

def resolve(program: AST, environment: Environment = None) -> AST:
    if environment is None:
        environment = Environment()

    def resolve_(program: AST) -> AST:
        return resolve(program, environment)

    match program:
        case NumLiteral(_) as N:
            return N
        case Variable(name):
            return environment.get(name)
        case Let(Variable(name) as v, e1, e2):
            re1 = resolve_(e1)
            environment.enter_scope()
            environment.add(name, v)
            re2 = resolve_(e2)
            environment.exit_scope()
            return Let(v, re1, re2)
        case LetFun(Variable(name) as v, params, body, expr):
            environment.enter_scope()
            environment.add(name, v)
            environment.enter_scope()
            for param in params:
                environment.add(param.name, param)
            rbody = resolve_(body)
            environment.exit_scope()
            rexpr = resolve_(expr)
            environment.exit_scope()
            return LetFun(v, params, rbody, rexpr)
        case FunCall(fn, args):
            rfn = resolve_(fn)
            rargs = []
            for arg in args:
                rargs.append(resolve_(arg))
            return FunCall(rfn, rargs)

def eval(program: AST, environment: Environment = None) -> Value:
    if environment is None:
        environment = Environment()

    def eval_(program):
        return eval(program, environment)

    match program:
        case NumLiteral(value):
            return value
        case Variable(_) as v:
            return environment.get(v)
        case Let(Variable(_) as v, e1, e2) | LetMut(Variable(_) as v, e1, e2):
            v1 = eval_(e1)
            environment.enter_scope()
            environment.add(v, v1)
            v2 = eval_(e2)
            environment.exit_scope()
            return v2
        case BinOp("+", left, right):
            return eval_(left) + eval_(right)
        case BinOp("-", left, right):
            return eval_(left) - eval_(right)
        case BinOp("*", left, right):
            return eval_(left) * eval_(right)
        case BinOp("/", left, right):
            return eval_(left) / eval_(right)
        case Put(Variable(_) as v, e):
            environment.update(v, eval_(e))
            return environment.get(v)
        case Get(Variable(_) as v):
            return environment.get(v)
        case Seq(things):
            v = None
            for thing in things:
                v = eval_(thing)
            return v
        case LetFun(Variable(_) as v, params, body, expr):
            environment.enter_scope()
            environment.add(v, FnObject(params, body))
            v = eval_(expr)
            environment.exit_scope()
            return v
        case FunCall(Variable(_) as v, args):
            fn = environment.get(v)
            argv = []
            for arg in args:
                argv.append(eval_(arg))
            environment.enter_scope()
            for param, arg in zip(fn.params, argv):
                environment.add(param, arg)
            v = eval_(fn.body)
            environment.exit_scope()
            return v
    raise InvalidProgram()

def test_eval():
    e1 = NumLiteral(2)
    e2 = NumLiteral(7)
    e3 = NumLiteral(9)
    e4 = NumLiteral(5)
    e5 = BinOp("+", e2, e3)
    e6 = BinOp("/", e5, e4)
    e7 = BinOp("*", e1, e6)
    assert eval(e7) == Fraction(32, 5)

def test_let_eval():
    a  = Variable("a")
    e1 = NumLiteral(5)
    e2 = BinOp("+", a, a)
    e  = Let(a, e1, e2)
    assert eval(e) == 10
    e  = Let(a, e1, Let(a, e2, e2))
    assert eval(e) == 20
    e  = Let(a, e1, BinOp("+", a, Let(a, e2, e2)))
    assert eval(e) == 25
    e  = Let(a, e1, BinOp("+", Let(a, e2, e2), a))
    assert eval(e) == 25
    e3 = NumLiteral(6)
    e  = BinOp("+", Let(a, e1, e2), Let(a, e3, e2))
    assert eval(e) == 22

def test_letmut():
    a = Variable("a")
    b = Variable("b")
    e1 = LetMut(b, NumLiteral(2), Put(a, BinOp("+", Get(a), Get(b))))
    e2 = LetMut(a, NumLiteral(1), Seq([e1, Get(a)]))
    assert eval(e2) == 3

def test_letfun():
    a = Variable("a")
    b = Variable("b")
    f = Variable("f")
    g = BinOp (
        "*",
        FunCall(f, [NumLiteral(15), NumLiteral(2)]),
        FunCall(f, [NumLiteral(12), NumLiteral(3)])
    )
    e = LetFun(
        f, [a, b], BinOp("+", a, b),
        g
    )
    assert eval(e) == (15+2)*(12+3)


def test_resolve():
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    e = Let(Variable.make("a"), NumLiteral(0), Variable.make("a"))
    # pp.pprint(e)
    re = resolve(e)
    # pp.pprint(re)

    e = LetFun(Variable.make("foo"), [Variable.make("a")], FunCall(Variable.make("foo"), [Variable.make("a")]),
               Let(Variable.make("g"), Variable.make("foo"),
                   LetFun(Variable.make("foo"), [Variable.make("a")], NumLiteral(0),
                          FunCall(Variable.make("g"), [NumLiteral(0)]))))
    pp.pprint(e)
    pp.pprint(r := resolve(e))
    print(eval(r))

test_resolve()
