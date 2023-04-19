def isprime(n):
    def go(d):
        return d * d > n or n % d != 0 and go(d+1)
    return go(2)

innum = 2*3*5

def go(cur, max):
    if cur > innum:
        return max
    else:
        if innum % cur != 0:
            return go(cur+1, max)
        else:
            return go(cur+1, cur if isprime(cur) else max)

print(go(2, 1))
