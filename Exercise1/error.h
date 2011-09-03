#ifndef _ERROR_FUNCTION_
#define _ERROR_FUNCTION_

#include <stdio.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

void exit_with_error(const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* _ERROR_FUNCTION_ */
