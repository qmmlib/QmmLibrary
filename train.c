#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#include <vector>
#include "linear.h"
#include "transform_line.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

double ** read_transform_matrix(char* filename,int & _nr, int & _nc);

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- multi-class support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"	11 -- L2-regularized L2-loss epsilon support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss epsilon support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss epsilon support vector regression (dual)\n"
	"	50 -- MMCD - please specify loss, curv\n"
	"	51 -- MMCD_SM - soft-max method\n"
	"	52 -- MMCD_SG - sub-gradient\n"
	"	53 -- MMCD_SIMPLE - please specify loss, curv\n"
	"	54 -- MMGCD - please specify loss, curv\n"
	"	55 -- MMCG - please specify loss, curv\n"
	"-l loss_type : L1, L2, LOG, HU1, HU2, LS\n"
	"   0 -- L1\n"
	"   1 -- L2\n"
	"   2 -- LOG\n"
	"   3 -- HU1\n"
	"   4 -- HU2\n"
	"   5 -- LS\n"
	"-u curv_type : MC, OC, NC\n"
	"   0  -- MC\n"
	"   1  -- OC\n"
	"   2  -- NC\n"
	"	for -s 50, 54, and 55 :\n"
	"   3 -- Start with MC and continue with NC after 1st iteration\n"
	"   4 -- Start with OC and continue with NC after 1st iteration\n"
	"-r alpha : regularization parameter, between 0 and 1\n"
	"   0 -- L1 regularization\n"
	"   1 -- L2 regularization\n"
	"   0 < r <1 -- elastic net\n"
	"-t tau   : loss parameter for HU1, HU2 and L1 losses, default 0.5\n"
	"-n epsi  : curv parameter, minimum curvature for NC, typical 0.001\n"
	"-i initmodel : initial model file to start iterations\n"
	"-g cg_tol : conjugate gradient tolerance for MMCG, default 0.001\n"
	"-d cd_tol : coordinate decent tolerance for MMCD, default 0.01\n"
	"-f n : only for MMCD, reset active coordinates every nth iteration, default 10\n"
	"-h chat_level : how much should I talk?\n"
	"   0 -- minimal\n"
	"   1 -- calc and print obj\n"
	"   for MMCD_SIMPLE, MMGCD and MMCG:\n"
	"       1 - running time\n" 
	"       2 - |f'(w)|_2 and |f'(w)|_inf\n"
	"       3 - objective value\n"
	"       4 - training and testing accuracy; mention test file with -x\n"
	"-x filename : test file to be used if chat_level>=4\n"
	"-X filename: save model at each iteration to 'filename[ITER]' (default don't save) \n" 
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n" 
	"	-s 1, 3, 4, and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"	-s 50, 54, and 55\n"
	"		|w-w^prev|_inf <= eps*|w|_inf,\n"
	"-S structure : trains structured weights;\n"
	"		structure is s for symmetric, a for antisymmetric\n"
	"		or a filename of the free transform matrix (default none)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

struct parameter param;

static char* readline(FILE *input, int max_idx) 
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}

	if(max_idx>0 && (param.structured_w=='s' || param.structured_w=='a' || param.structured_w=='f')) 
	{
		int y;
		double * read_line = (double*) malloc(sizeof(double)*(max_idx+1));
		memset(read_line, 0, sizeof(double)*(max_idx+1));
		split(line, read_line, &y);
		transform_line(read_line,y,max_idx,line, param.structured_w,param.transform_matrix,param.tmx_r,param.tmx_c);
		free(read_line);
	}
	
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

struct feature_node *x_space;
struct problem prob;
struct model* model_;
int flag_cross_validation;
int nr_fold;
double bias;

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	param.train_file = Malloc(char,1024);
	strcpy(param.train_file, input_file_name);
	error_msg = check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	if(flag_cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		clock_t start_cpu, end_cpu;
		double cpu_time_used;
     	start_cpu = clock();
		model_=train(&prob, &param);
		end_cpu = clock();
     	cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
		if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"can't save model to file %s\n",model_file_name);
			exit(1);
		}
		free_and_destroy_model(&model_);
	}
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);

	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR || 
	   param.solver_type == L2R_L1LOSS_SVR_DUAL || 
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(i=0;i<prob.l;i++)
                {
                        double y = prob.y[i];
                        double v = target[i];
                        total_error += (v-y)*(v-y);
                        sumv += v;
                        sumy += y;
                        sumvv += v*v;
                        sumyy += y*y;
                        sumvy += v*y;
                }
                printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
                printf("Cross Validation Squared correlation coefficient = %g\n",
                        ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
                        ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
                        );
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}

	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.solver_type = MMCD;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	flag_cross_validation = 0;
	bias = -1;
	param.reg_param   = 1; // L2 regularizer
	param.loss_param  = 0.5; // tau for HU1, HU2
	param.curv_param  = 1e-3; // epsilon for minimum curvature value for NC
	param.loss_type   = HU2; // Huberized-hinge loss
	param.curv_type   = OC; // optimal curv
	param.turn_to_nc  = 0;
	param.save_each_iter = 0;
	param.init_model_file = NULL;
	param.chat_level = 0; // minimal talk
	param.cg_tol = 0.001;
	param.cd_tol = 0.01;
	param.cd_reset = 10;
	param.structured_w = 0;
	param.transform_matrix = 0;
	param.tmx_c = 0;
	param.tmx_r = 0;
	
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;

			case 'l':
				param.loss_type = (loss_var) atoi(argv[i]);
				break;

			case 'u':
				if(atoi(argv[i]) < 3)
				{
					param.curv_type = (curv_var) atoi(argv[i]);
					param.turn_to_nc = 0;
				}
				else if(atoi(argv[i]) == 3)
				{
					param.curv_type = (curv_var) 0;
					param.turn_to_nc = 1;
				}
				else if(atoi(argv[i]) == 4)
				{
					param.curv_type = (curv_var) 1;
					param.turn_to_nc = 1;
				}		
				break;
				
			case 'r':
				param.reg_param = atof(argv[i]);
				break;

			case 'd':
				param.cd_tol = atof(argv[i]);
				break;

			case 'f':
				param.cd_reset = atoi(argv[i]);
				break;

			case 'g':
				param.cg_tol = atof(argv[i]);
				break;

			case 'x':
				param.test_file = Malloc(char,1024);
				strcpy(param.test_file, argv[i]);
				break;

			case 't':
				param.loss_param = atof(argv[i]);
				break;

			case 'n':
				param.curv_param = atof(argv[i]);
				break;

			case 'i':
				param.init_model_file = Malloc(char,1024);
				strcpy(param.init_model_file, argv[i]);
				break;

			case 'h':
				param.chat_level = atoi(argv[i]);
				break;

			case 'S': 
				if(strcmp(argv[i],"s")==0 || strcmp(argv[i],"a")==0)
				{
					param.structured_w = argv[i][0];
					printf("\nUsing symmetric or asymmetric matrix as %c\n",param.structured_w);
				}
				else
				{
					param.transform_matrix = read_transform_matrix(argv[i],param.tmx_r,param.tmx_c);
					param.structured_w = 'f';
					printf("\nUsing free transform matrix %s\n",argv[i]);
				}
				break;

			case 'X':
				param.save_each_iter = Malloc(char,1024);
				strcpy(param.save_each_iter, argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR: 
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL: 
			case L2R_L1LOSS_SVC_DUAL: 
			case MCSVM_CS: 
			case L2R_LR_DUAL: 
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC: 
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
		
		if(param.solver_type == MMCD ||param.solver_type == MMCD_SM || param.solver_type == MMCD_SG || param.solver_type == MMCD_SIMPLE || param.solver_type == MMGCD || param.solver_type == MMCG )
			param.eps = 0.01;
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i, max_index_=0;
	long int elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp, 0)!=NULL) 
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			idx = strtok(NULL,":"); 
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			if((int) strtol(idx,&endptr,10)>max_index_) 
				max_index_ = (int) strtol(idx,&endptr,10); 
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	param.real_dim = max_index_; 
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp, param.real_dim); //sym
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n; 
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);
}

// below are for free transform matrix
int endofline(FILE *ifp)
{
	int c = fgetc(ifp);
	int d = c;
    int eol = (c == '\r' || c == '\n');
    if (c == '\r')
    {
        c = getc(ifp);
        if (c != '\n' && c != EOF)
            ungetc(c, ifp);
    }
    return(eol);
}

double ** read_transform_matrix(char* filename,int & _nr, int & _nc)
{
    float read_float;    
    FILE * fp = fopen(filename,"r");
    int nlines = 0;
    std::vector<float> v_read_floats;
    int check_ncols = -1;
    while (fscanf(fp,"%f",&read_float) == 1)
	{
		v_read_floats.push_back(read_float);
		if(endofline(fp))
		{
			nlines++;
			if(check_ncols==-1)
			{
				check_ncols = v_read_floats.size();
				if(nlines!=1)
				{
					printf("\nERROR: This is illogical (%d)\n",nlines);
					exit(1);
				}
					
			}
		}
	}
	int ncols = v_read_floats.size()/nlines;
	if(check_ncols!=ncols)
	{
		printf("\nERROR (1): File does not seem like a matrix! (%d %d)\n",ncols,check_ncols);
		exit(1);
	}

	double ** transform_matrix = new double*[nlines];
	for(int i = 0 ; i<nlines; i++)
		transform_matrix[i] = new double[ncols];
	
	int row = 0;
	int col = 0;
	for(int i = 0 ; i<v_read_floats.size(); i++)
	{
		transform_matrix[row][col] = (double)v_read_floats[i];
		col++;
		if(col >= ncols)
		{
			col = 0;
			row++;
		}
	}
	
	if(row!=nlines || col!=0)
	{
		printf("\nERROR (2): File does not seem like a matrix! (%d %d)\n",row,col);
		exit(1);
	}
		
	_nr = nlines;
	_nc = ncols;
	
	return transform_matrix;
}

