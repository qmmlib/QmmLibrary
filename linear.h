#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
Holds features of an instance
*/
struct feature_node
{
	int index;
	double value;
};

/**
When training files are read, they are loaded into an object of type problem.
*/
struct problem
{
	int l, n;
	double *y; ///< Labels of the training instance
	struct feature_node **x; ///< Sparse instance data in a linked list
	double bias;            /* < 0 if no bias term */  
};

/**
Solver type, ones >=50 are added during QMM development
*/
enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL, MMCD = 50, MMCD_SM, MMCD_SG, MMCD_SIMPLE, MMGCD, MMCG  }; /* solver_type */
/**
Loss type for QMM related methods
*/
typedef enum { L1, L2, LOG, HU1, HU2, LS } loss_var; /* loss type for MMCD */
/**
Curvature type; Maximal, Optimal or Newton's Curvature
*/
typedef enum {MC,OC,NC} curv_var; /* curv type for MMCD */

/**
Training parameters that control the training and shared among almost all solvers.
*/
struct parameter
{
	int solver_type;

	/* these are for mm* */
	loss_var loss_type;
	double loss_param;
	curv_var curv_type;
	double curv_param;
	double reg_param; /* alpha in paper */
	char *init_model_file;
	char *test_file;
	char *train_file;
	int chat_level;
	int turn_to_nc;
	char *save_each_iter;

	/* these are for mmcd */
	double cd_tol;
	int cd_reset;

	/* this is for mmcg */
	double cg_tol;

	/* these are for structured weights */
	int real_dim;
	char structured_w;
	double ** transform_matrix;
	int tmx_r, tmx_c;
		
	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	double p;
};

/**
Trained model
*/
struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
	double bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);
void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, double *target);

double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double predict(const struct model *model_, const struct feature_node *x);
double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
int check_probability_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));


#ifdef DOXYGEN_SHOULD_READ_THIS_IFDEFINED_SKIP_IFNDEFINED 

/**
 A coordinate descent algorithm for 
 the dual of L2-regularized logistic regression problems

  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
    s.t.      0 <= \alpha_i <= upper_bound_i,
 
  where Qij = yi yj xi^T xj and 
  upper_bound_i = Cp if y_i = 1
  upper_bound_i = Cn if y_i = -1

 Given: 
 x, y, Cp, Cn
 eps is the stopping tolerance

 solution will be put in w

 See Algorithm 5 of Yu et al., MLJ 2010
*/
void solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn);

/**
 A coordinate descent algorithm for 
 L1-regularized L2-loss support vector classification

  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,

 Given: 
 x, y, Cp, Cn
 eps is the stopping tolerance

 solution will be put in w

 See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)
*/
static void solve_l1r_l2_svc(
	problem *prob_col, double *w, double eps, 
	double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);

/**
 A coordinate descent algorithm for 
 L1-regularized logistic regression problems

  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),

 Given: 
 x, y, Cp, Cn
 eps is the stopping tolerance

 solution will be put in w

 See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)
*/
static void solve_l1r_lr(
	const problem *prob_col, double *w, double eps, 
	double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);


/**
 A coordinate descent algorithm for 
 L1-loss and L2-loss epsilon-SVR dual problem

   min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i  ,
    s.t.        -upper_bound_i <= \beta_i <= upper_bound_i  ,
 
  where   Qij = xi^T xj   and
  D is a diagonal matrix 

 In L1-SVM case:
 		  upper_bound_i = C  ,
 		  lambda_i = 0  
 In L2-SVM case:
 		  upper_bound_i = INF  ,
 		  lambda_i = 1/(2*C)  

 Given: 
 x, y, p, C
 eps is the stopping tolerance

 solution will be put in w

 See Algorithm 4 of Ho and Lin, 2012   
*/
static void solve_l2r_l1l2_svr(
	const problem *prob, double *w, const parameter *param,
	int solver_type);

/**
 A coordinate descent algorithm for 
 L1-loss and L2-loss SVM dual problems

   min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha ,
    s.t.        0 <= \alpha_i <= upper_bound_i  ,
 
  where   Qij = yi yj xi^T xj   and
  D is a diagonal matrix 

 In L1-SVM case:
 		  upper_bound_i = Cp if y_i = 1  ,
 		  upper_bound_i = Cn if y_i = -1  ,
 		  D_ii = 0  
 In L2-SVM case:
 		  upper_bound_i = INF  ,
 		  D_ii = 1/(2*Cp)	if y_i = 1  ,
 		  D_ii = 1/(2*Cn)	if y_i = -1  

 Given: 
 x, y, Cp, Cn
 eps is the stopping tolerance

 solution will be put in w
 
 See Algorithm 3 of Hsieh et al., ICML 2008
*/
static void solve_l2r_l1l2_svc(const problem *prob, double *w, double eps, 	double Cp, double Cn, int solver_type);

/**
 A coordinate descent algorithm for 
 multi-class support vector machines by Crammer and Singer

   min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i 
    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i 
 
  where  e^m_i = 0 if y_i  = m ,
         e^m_i = 1 if y_i != m ,
   C^m_i = C if m  = y_i , 
   C^m_i = 0 if m != y_i, 
  and  w_m(\alpha) = \sum_i \alpha^m_i x_i 

 Given: 
 x, y, C
 eps is the stopping tolerance

 solution will be put in w

 See Appendix of LIBLINEAR paper, Fan et al. (2008)
*/
class Solver_MCSVM_CS;

/**
A multi-class soft-max algorithm based on QMM
using any loss (param->loss_type)
and L1, L2 or elastic net regularizer (param->alpha).

\f$ min_w \sum {(1-alpha)|wj| + alpha wj^2 }  + C \sum max(0, 1-yi w^T xi)^2 \f$

It does not do any speed improvements like update schedule, random ordering, etc.

Given: 
x, y, C, param (training parameters),
eps (the stopping tolerance), start, count, nr_class;
solution will be put in w.
*/
static void solve_mmcd_sm(problem *prob_col, const parameter *param , double *w, double C, int *start, int *count, int nr_class);

/**
A coordinate descent algorithm based on QMM
using any loss (param->loss_type)
and L1, L2 or elastic net regularizer (param->alpha).

\f$ min_w \sum {(1-alpha)|wj| + alpha wj^2 }  + C \sum max(0, 1-yi w^T xi)^2 \f$

It does not do any speed improvements like update schedule, random ordering, etc.

Given: 
x, y, Cp, Cn, param (training parameters),
eps (the stopping tolerance);
solution will be put in w.

model_ is the model file of w and cpu_begin is the training start time;
called while information is printed at each iteration.
*/
static void solve_mmcd_simple(problem *prob_col, double *w, double eps, double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);

/**
A group coordinate descent algorithm based on QMM
using any loss (param->loss_type)
and L1, L2 or elastic net regularizer (param->alpha).

\f$ min_w \sum {(1-alpha)|wj| + alpha wj^2 }  + C \sum max(0, 1-yi w^T xi)^2 \f$

Given: 
x, y, Cp, Cn, param (training parameters),
eps (the stopping tolerance);
solution will be put in w.

model_ is the model file of w and cpu_begin is the training start time;
called while information is printed at each iteration.
*/
static void solve_mmgcd(problem *prob_col, double *w, double eps, double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);

/**
A conjugate gradient algorithm based on QMM
using any loss (param->loss_type)
and L1, L2 or elastic net regularizer (param->alpha).

\f$ min_w \sum {(1-alpha)|wj| + alpha wj^2 }  + C \sum max(0, 1-yi w^T xi)^2 \f$

Given: 
x, y, Cp, Cn, param (training parameters),
eps (the stopping tolerance);
solution will be put in w.
 
model_ is the model file of w and cpu_begin is the training start time;
called while information is printed at each iteration.

*/
static void solve_mmcg(const problem *prob_row, double *w, double eps, double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);

/**
A coordinate descent algorithm based on QMM 
using any loss (param->loss_type)
and L1, L2 or elastic net regularizer (param->alpha).

\f$ min_w \sum {(1-alpha)|wj| + alpha wj^2 }  + C \sum max(0, 1-yi w^T xi)^2 \f$

Given: 
x, y, Cp, Cn, param (training parameters),
eps (the stopping tolerance);
solution will be put in w.

model_ is the model file of w and cpu_begin is the training start time;
called while information is printed at each iteration.
*/
static void solve_mmcd(problem *prob_col, double *w, double eps, double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);


/**
A class for L2 regularized L2 loss that implements function interface;
Initialized with the loaded problem and C parameter.

Called by TRON methods during training
*/
class l2r_l2_svc_fun;


/**
A class for L2 regularized logistic loss that implements function interface;
Initialized with the loaded problem and C parameter.

Called by TRON methods during training
*/
class l2r_lr_fun;

#endif

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

