# print(foo(2)) # foo is not available here.

def foo(a):
    return a + 1

print(foo(2))

def fact(n):
    if n <= 1: return 1
    return n * fact(n-1)

print(fact(10))

# futuredef odd(n)

def even(n):
    match n:
        case 0: return True
        case 1: return False
        case _: return odd(n-1)

# print(even(15)) # odd is not defined.

def odd(n):
    match n:
        case 0: return False
        case 1: return True
        case _: return even(n-1)

print(even(15))
