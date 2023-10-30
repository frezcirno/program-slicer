#include <stdio.h>
#include <stdlib.h>
#include "md5.h"

int fib(int n) {
    if (n <= 1) {
        return n;
    }
    return fib(n-1) + fib(n-2);
}

int
main(
    int argc,
    char *argv[]
) {
    if (argc != 2) {
        printf("Usage: %s n\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    printf("fib(%d) = %d\n", n, fib(n));

    switch (argc) {
        case 1:
        case 2:
            printf("12");
            break;
        case 3:
            printf("3");
            break;
        default:
            printf("4");
    }

    if (1) n = 2;
    else n = 3;

    while (0) n = 3;

    while (n > 0) {
        printf("%d\n", n);
        n--;
    }

    do {
        printf("%d\n", n);
        n++;
    } while (n < 10);

    int bval = (argc == 1) ? 1 : 0;

    char *input = "hello world";
    char result[32];
    md5(input, result);
    printf("md5(\"%s\") = %s\n", input, result);

    return 0;
}
