#include <common/error.h>

#include <stdlib.h>

/*
 * error function, from: Kernighan Ritchie page 174
 * print an error message and die
 */
void exit_with_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, args);
    va_end(args);
    exit(EXIT_FAILURE);
}
