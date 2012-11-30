#include "linear.h"
#include <stdlib.h>
#include <cstdio>
#include <cstring>
#include <ctype.h>

/**
This function is called in "__QMMCHAT__" macro to perform prediction
on the file_name with the trained model_.

Used to print training and test accuracies at each iteration.
*/
double do_predict_iter(char* file_name, const struct model* model_);