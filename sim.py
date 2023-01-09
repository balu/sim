from dataclasses import dataclass
from fractions import Fraction
from typing import Union

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

AST = NumLiteral | BinOp

Value = Fraction

class InvalidProgram(Exception):
    pass

def eval(program: AST) -> Value:
    match program:
        case NumLiteral(value):
            return value
        case BinOp("+", left, right):
            return eval(left) + eval(right)
        case BinOp("-", left, right):
            return eval(left) - eval(right)
        case BinOp("*", left, right):
            return eval(left) * eval(right)
        case BinOp("/", left, right):
            return eval(left) / eval(right)
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
