#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif
void transform_line(double * read_line, int y, int max_idx, char* line, char structured_w, double ** tmx, int tmx_r, int tmx_c);
void split(char* _str, double * read_line, int * y);
#ifdef __cplusplus
}
#endif 

