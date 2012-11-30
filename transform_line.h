#include <stdio.h>
#include <stdlib.h>
#include <cstring>

/**
This function is called during the file read when the problem
is first loaded.

If param->S is set to enforce a (anti)symmetric weight set, these
are used to modify data matrix on the fly.
*/

#ifdef __cplusplus
extern "C" {
#endif
void transform_line(double * read_line, int y, int max_idx, char* line, char structured_w, double ** tmx, int tmx_r, int tmx_c);
void split(char* _str, double * read_line, int * y);
#ifdef __cplusplus
}
#endif 

