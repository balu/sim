from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Mapping, List

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

@dataclass
class Variable:
    name: str

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

AST = NumLiteral | BinOp | Variable | Let | LetMut | Put | Get | Seq

Value = Fraction

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

def eval(program: AST, environment: Environment = None) -> Value:
    if environment is None:
        environment = Environment()

    def eval_(program):
        return eval(program, environment)

    match program:
        case NumLiteral(value):
            return value
        case Variable(name):
            return environment.get(name)
        case Let(Variable(name), e1, e2) | LetMut(Variable(name), e1, e2):
            v1 = eval_(e1)
            environment.enter_scope()
            environment.add(name, v1)
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
        case Put(Variable(name), e):
            environment.update(name, eval_(e))
            return environment.get(name)
        case Get(Variable(name)):
            return environment.get(name)
        case Seq(things):
            v = None
            for thing in things:
                v = eval_(thing)
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
