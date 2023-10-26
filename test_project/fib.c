#include <stdio.h>
#include <stdlib.h>
#include "md5.h"

int fib(int n) {
    if (n <= 1) {
        return n;
    }
    return fib(n-1) + fib(n-2);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s n\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    printf("fib(%d) = %d\n", n, fib(n));

    char *input = "hello world";
    char result[32];
    md5(input, result);
    printf("md5(\"%s\") = %s\n", input, result);

    return 0;
}
