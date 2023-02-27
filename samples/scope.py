x = 1
y = x
x = 2
print(y) # 1

def foo(n):
    return 0

bar = foo

def foo(n):
    return 1

print(bar(3))

def fact(n):
    return 1 if n <= 1 else n*fact(n-1)

baz = fact

def fact(n):
    return 1

print(baz(3)) # 3

def abc(n):
    return 1 if n <= 1 else n*abc(n-1)

deg = abc

def hij():
    def abc(n):
        return 1
    print(deg(3))

hij()
