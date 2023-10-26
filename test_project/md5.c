#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

void md5(const char* input, char* result) {
    uint32_t h0, h1, h2, h3;
    uint32_t a, b, c, d, f, g, temp;
    uint32_t w[16];
    int i, j, len;

    // Initialize variables
    h0 = 0x67452301;
    h1 = 0xEFCDAB89;
    h2 = 0x98BADCFE;
    h3 = 0x10325476;

    // Pre-processing
    len = strlen(input);
    for (i = 0; i < len - 63; i += 64) {
        a = h0;
        b = h1;
        c = h2;
        d = h3;

        // Copy input chunk into w
        for (j = 0; j < 16; j++) {
            w[j] = ((uint32_t)input[i + 4*j]) |
                   ((uint32_t)input[i + 4*j + 1] << 8) |
                   ((uint32_t)input[i + 4*j + 2] << 16) |
                   ((uint32_t)input[i + 4*j + 3] << 24);
        }

        // Main loop
        for (j = 0; j < 64; j++) {
            if (j < 16) {
                f = (b & c) | ((~b) & d);
                g = j;
            } else if (j < 32) {
                f = (d & b) | ((~d) & c);
                g = (5*j + 1) % 16;
            } else if (j < 48) {
                f = b ^ c ^ d;
                g = (3*j + 5) % 16;
            } else {
                f = c ^ (b | (~d));
                g = (7*j) % 16;
            }

            temp = d;
            d = c;
            c = b;
            b = b + LEFTROTATE((a + f + w[g] + 0x5A827999), 5);
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
    }

    // Output hash
    sprintf(result, "%08x%08x%08x%08x", h0, h1, h2, h3);
}
