#include "do_predict_iter.h"

/*! \mainpage Documentation for QMM Library mod for LIBLINEAR
 *
 * \section intro_sec Introduction
 *
 * QMM library is a general solver for binary and multi-class regularized empirical risk classification.
 *
 * \section highlight_sec Highlights
 * 
 * Below are some useful code snippets at a glance:
 * 
 * -# solve_mmcd() : Function that performs coordinate descent with QMM
 * -# qmmchat.h : Macro to print information at each iteration
 * -# \ref parameter : Struct that control training parameters
 * -# \ref problem : Loaded training file
 * \section install_sec Installation
 *
 * QMM library is %100 compatible with LIBLINEAR. Just "make" to obtain command line binaries, make.m in MATLAB to obtain MEX binaries.
 *  
 * \section doc_sec Scope of Documentation
 * 
 * Documentation is available for added functions for QMM Library, as well as available documentation in LIBLINEAR functions.
 * 
 * Also documentation is added for some existing LIBLINEAR code (e.g. TRON) to ease the work of future development.
 *
 */

/**
A macro definition for printing information at each iteration,
if called, following values should be set:

-# iter: Current iteration number
-# cpu_time_elapsed: Time passed since beinning
-# objective_value: Current value of objective function

If chat_level > 3, a test_file should be passed so test accuracy can be calculated
*/
#define __QMMCHAT__

#ifndef DOXYGEN_SHOULD_READ_THIS_IFDEFINED_SKIP_IFNDEFINED 

#define __QMMCHAT__																																			\
if (param->chat_level > 0) {																																			\
	printf("Iter\t%d\tCPU_time\t%.10f\t", iter, cpu_time_elapsed);																									\
	if (param->chat_level > 1) {																																		\
        /* printf("|w|_inf\t%.4f\t|diff_w|_inf\t%.4f\t",wmax_new,Dmax_new); */                                                                                         	\
		if (param->chat_level > 2) {																																	\
			printf("Objective_value\t%lf\t", objective_value);																										\
			if (param->chat_level > 3) {																																\
				if(param->test_file==0)																																\
				{																																						\
					fprintf(stderr,"\nERROR: Please provide a test file name if chat_level>=4\n");																\
					exit(1);																																			\
				}																																						\
				printf("Train_acc\t%.2f\t",do_predict_iter(param->train_file,model_));																				\
				printf("Test_acc\t%.2f\t",do_predict_iter(param->test_file,model_));																					\
			}																																							\
		}																																								\
	}																																									\
	printf("\n");																																						\
}																																										\

#endif 