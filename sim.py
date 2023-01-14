from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Mapping, cast

@dataclass
class NumLiteral:
    value: Fraction
    def __init__(self, *args):
        self.value = Fraction(*args)

@dataclass
class BoolLiteral:
    value: bool

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
class IfElse:
    condition: 'AST'
    iftrue: 'AST'
    iffalse: 'AST'

AST = NumLiteral | BoolLiteral | BinOp | Variable | Let | IfElse

Value = Fraction | bool

class InvalidProgram(Exception):
    pass

def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
    if environment is None:
        environment = {}
    match program:
        case NumLiteral(value):
            return value
        case BoolLiteral(value):
            return value
        case Variable(name):
            if name in environment:
                return environment[name]
            raise InvalidProgram()
        case Let(Variable(name), e1, e2):
            v1 = eval(e1, environment)
            return eval(e2, environment | { name: v1 })
        case IfElse(condition, iftrue, iffalse):
            c = eval(condition, environment)
            match c:
                case bool(True):
                    return eval(iftrue, environment)
                case bool(False):
                    return eval(iffalse, environment)
            raise InvalidProgram()
        case BinOp("+", left, right):
            match eval(left, environment), eval(right, environment):
                case [Fraction() as left, Fraction() as right]:
                    return left + right
            raise InvalidProgram()
        case BinOp("-", left, right):
            match eval(left, environment), eval(right, environment):
                case [Fraction() as left, Fraction() as right]:
                    return left - right
            raise InvalidProgram()
        case BinOp("*", left, right):
            match eval(left, environment), eval(right, environment):
                case [Fraction() as left, Fraction() as right]:
                    return left * right
            raise InvalidProgram()
        case BinOp("/", left, right):
            match eval(left, environment), eval(right, environment):
                case [Fraction() as left, Fraction() as right]:
                    if right == 0:
                        raise InvalidProgram()
                    return left / right
            raise InvalidProgram()
        case BinOp("=", left, right):
            return eval(left, environment) == eval(right, environment)
        case BinOp("≠", left, right):
            return eval(left, environment) != eval(right, environment)
        case BinOp(">", left, right):
            match eval(left, environment), eval(right, environment):
                case [Fraction() as left, Fraction() as right]:
                    return left > right
            raise InvalidProgram()
        case BinOp("<", left, right):
            match eval(left, environment), eval(right, environment):
                case [Fraction() as left, Fraction() as right]:
                    return left < right
            raise InvalidProgram()
        case BinOp("≥", left, right):
            match eval(left, environment), eval(right, environment):
                case [Fraction() as left, Fraction() as right]:
                    return left >= right
            raise InvalidProgram()
        case BinOp("≤", left, right):
            match eval(left, environment), eval(right, environment):
                case [Fraction() as left, Fraction() as right]:
                    return left <= right
            raise InvalidProgram()
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
    pi = Variable("π")
    piv = NumLiteral(22, 7)
    e = Let(pi, piv, BinOp("*", pi, BinOp("*", NumLiteral(7), NumLiteral(7))))
    assert eval(e) == 22*7

def test_ifelse_eval():
    c = BinOp(">", NumLiteral(5), NumLiteral(3))
    t = BinOp("-", NumLiteral(5), NumLiteral(3))
    f = BinOp("+", NumLiteral(5), NumLiteral(3))
    e = IfElse(c, t, f)
    assert eval(e) == 2
