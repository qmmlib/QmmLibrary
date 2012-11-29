#include "do_predict_iter.h"

#define __QMMCHAT__																																					\
if (param->chat_level > 0) {																																			\
	printf("Iter\t%d\tCPU_time\t%.10f\t", iter, cpu_time_elapsed);																									\
	if (param->chat_level > 1) {																																		\
        /* printf("|w|_inf\t%.4f\t|diff_w|_inf\t%.4f\t",wmax_new,Dmax_new); */                                                                                         	\ 
		if (param->chat_level > 2) {																																	\
			/* printf("Loss type\t%d\tcurvature type\t%d\tReg_param\t%.1e\tloss_param:\t%.1e\tcurv_param\t%.1e\t",loss_type, curv_type, reg_param, loss_param, curv_param);*/	\
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

