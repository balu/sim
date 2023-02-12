#include <stdio.h>

int fact(int n)
{
    return 1;
}

int odd(int n);

int even(int n)
{
    if (n == 0) return 1;
    if (n == 1) return 0;
    return odd(n-1);
}

int odd(int n)
{
    if (n == 0) return 0;
    if (n == 1) return 1;
    return even(n-1);
}

int main()
{
    printf("%d\n", even(1));
}
