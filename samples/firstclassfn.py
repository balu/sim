def foo(x):
    return 2*x*x + 3*x + 5

def derivative(f):
    dx = 0.0001
    def df(x):
        return (f(x+dx)-f(x))/dx
    return df

dfoo = derivative(foo)

print(dfoo(3))

def make_counter(init=0):
    def inc():
        nonlocal init
        init += 1
    def dec():
        nonlocal init
        init -= 1
    def get():
        nonlocal init
        return init
    return (inc, dec, get)

(i1, d1, g1) = make_counter()
(i2, d2, g2) = make_counter(5)

i1()
i1()
i2()
d1()
i2()
print(g1(), g2())
