
int foo() {
    int a = 0;
    int b;
    int c;

    if (a == 1) {
        b = 2;
        c = 3;
    } else {
        b = 4;
        c = 5;
        if (b == 6) {
            c = 7;
        } else {
            c = 8;
        }
    }

    b = bar(c);

    c = 9;

    return 0;
}

int bar(int x) {
    if (x == 10) {
        return
            foobar();
    } else {
        return 0;
    }
}

int foobar() {
    return 11;
}

