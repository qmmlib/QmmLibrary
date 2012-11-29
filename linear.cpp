#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"
#include <vector>
#include "qmmchat.h"
#include <time.h>

static void solve_mmcd_sm(problem *prob_col, const parameter *param , double *w, double C, int *start, int *count, int nr_class);
static void solve_mmcd_simple(problem *prob_col, double *w, double eps, double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);
static void solve_mmgcd(problem *prob_col, double *w, double eps, double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);
static void solve_mmcg(const problem *prob_row, double *w, double eps, double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);
static void solve_mmcd(problem *prob_col, double *w, double eps, double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin);

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{   
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

class l2r_lr_fun: public function
{
public:
	l2r_lr_fun(const problem *prob, double *C);
	~l2r_lr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const problem *prob;
};

l2r_lr_fun::l2r_lr_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	this->C = C;
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
}


double l2r_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + g[i];
}

int l2r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2r_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

class l2r_l2_svc_fun: public function
{
public:
	l2r_l2_svc_fun(const problem *prob, double *C);
	~l2r_l2_svc_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

protected:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	I = new int[l];
	this->C = C;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_svc_fun::subXv(double *v, double *Xv)
{
	int i;
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

class l2r_l2_svr_fun: public l2r_l2_svc_fun
{
public:
	l2r_l2_svr_fun(const problem *prob, double *C, double p);

	double fun(double *w);
	void grad(double *w, double *g);

private:
	double p;
};

l2r_l2_svr_fun::l2r_l2_svr_fun(const problem *prob, double *C, double p):
	l2r_l2_svc_fun(prob, C)
{
	this->p = p;
}

double l2r_l2_svr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;
	
	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2;	
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];
		if(d < -p)
			f += C[i]*(d+p)*(d+p);
		else if(d > p)
			f += C[i]*(d-p)*(d-p);
	}

	return(f);
}

void l2r_l2_svr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	sizeI = 0;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];
		
		// generate index set I
		if(d < -p)
		{
			z[sizeI] = C[i]*(d+p);
			I[sizeI] = i;
			sizeI++;
		}
		else if(d > p)
		{
			z[sizeI] = C[i]*(d-p);
			I[sizeI] = i;
			sizeI++;
		}
	
	}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

// A coordinate descent algorithm for 
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
// 
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i, 
//  C^m_i = 0 if m != y_i, 
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i 
//
// Given: 
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Appendix of LIBLINEAR paper, Fan et al. (2008)

#define GETI(i) ((int) prob->y[i])
// To support weights for instances, use GETI(i) (i)

class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
		~Solver_MCSVM_CS();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
		double *B, *C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_MCSVM_CS::Solver_MCSVM_CS(const problem *prob, int nr_class, double *weighted_C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->B = new double[nr_class];
	this->G = new double[nr_class];
	this->C = weighted_C;
}

Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
	delete[] B;
	delete[] G;
}

int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}

void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
	int r;
	double *D;

	clone(D, B, active_i);
	if(yi < active_i)
		D[yi] += A_i*C_yi;
	qsort(D, active_i, sizeof(double), compare_double);

	double beta = D[0] - A_i*C_yi;
	for(r=1;r<active_i && beta<r*D[r];r++)
		beta += D[r];
	beta /= r;

	for(r=0;r<active_i;r++)
	{
		if(r == yi)
			alpha_new[r] = min(C_yi, (beta-B[r])/A_i);
		else
			alpha_new[r] = min((double)0, (beta - B[r])/A_i);
	}
	delete[] D;
}

bool Solver_MCSVM_CS::be_shrunk(int i, int m, int yi, double alpha_i, double minG)
{
	double bound = 0;
	if(m == yi)
		bound = C[GETI(i)];
	if(alpha_i == bound && G[m] < minG)
		return true;
	return false;
}

void Solver_MCSVM_CS::Solve(double *w)
{
	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *alpha_new = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	int *d_ind = new int[nr_class];
	double *d_val = new double[nr_class];
	int *alpha_index = new int[nr_class*l];
	int *y_index = new int[l];
	int active_size = l;
	int *active_size_i = new int[l];
	double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;

	// Initial alpha can be set here. Note that 
	// sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
	// alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
	// alpha[i*nr_class+m] <= 0 if prob->y[i] != m
	// If initial alpha isn't zero, uncomment the for loop below to initialize w
	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0; 
	for(i=0;i<l;i++)
	{
		for(m=0;m<nr_class;m++)
			alpha_index[i*nr_class+m] = m;
		feature_node *xi = prob->x[i];
		QD[i] = 0;
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
	
			// Uncomment the for loop if initial alpha isn't zero
			// for(m=0; m<nr_class; m++)
			//	w[(xi->index-1)*nr_class+m] += alpha[i*nr_class+m]*val;
			xi++;
		}
		active_size_i[i] = nr_class;
		y_index[i] = (int)prob->y[i];
		index[i] = i;
	}

	while(iter < max_iter) 
	{
		double stopping = -INF;
		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for(s=0;s<active_size;s++)
		{
			i = index[s];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*nr_class];

			if(Ai > 0)
			{
				for(m=0;m<active_size_i[i];m++)
					G[m] = 1;
				if(y_index[i] < active_size_i[i])
					G[y_index[i]] = 0;

				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
						G[m] += w_i[alpha_index_i[m]]*(xi->value);
					xi++;
				}

				double minG = INF;
				double maxG = -INF;
				for(m=0;m<active_size_i[i];m++)
				{
					if(alpha_i[alpha_index_i[m]] < 0 && G[m] < minG)
						minG = G[m];
					if(G[m] > maxG)
						maxG = G[m];
				}
				if(y_index[i] < active_size_i[i])
					if(alpha_i[(int) prob->y[i]] < C[GETI(i)] && G[y_index[i]] < minG)
						minG = G[y_index[i]];

				for(m=0;m<active_size_i[i];m++)
				{
					if(be_shrunk(i, m, y_index[i], alpha_i[alpha_index_i[m]], minG))
					{
						active_size_i[i]--;
						while(active_size_i[i]>m)
						{
							if(!be_shrunk(i, active_size_i[i], y_index[i], 
											alpha_i[alpha_index_i[active_size_i[i]]], minG))
							{
								swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
								swap(G[m], G[active_size_i[i]]);
								if(y_index[i] == active_size_i[i])
									y_index[i] = m;
								else if(y_index[i] == m) 
									y_index[i] = active_size_i[i];
								break;
							}
							active_size_i[i]--;
						}
					}
				}

				if(active_size_i[i] <= 1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;	
					continue;
				}

				if(maxG-minG <= 1e-12)
					continue;
				else
					stopping = max(maxG - minG, stopping);

				for(m=0;m<active_size_i[i];m++)
					B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]] ;

				solve_sub_problem(Ai, y_index[i], C[GETI(i)], active_size_i[i], alpha_new);
				int nz_d = 0;
				for(m=0;m<active_size_i[i];m++)
				{
					double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
					alpha_i[alpha_index_i[m]] = alpha_new[m];
					if(fabs(d) >= 1e-12)
					{
						d_ind[nz_d] = alpha_index_i[m];
						d_val[nz_d] = d;
						nz_d++;
					}
				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m]*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
		{
			info(".");
		}

		if(stopping < eps_shrink)
		{
			if(stopping < eps && start_from_all == true)
				break;
			else
			{
				active_size = l;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class;
				info("*");
				eps_shrink = max(eps_shrink/2, eps);
				start_from_all = true;
			}
		}
		else
			start_from_all = false;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l*nr_class;i++)
	{
		v += alpha[i];
		if(fabs(alpha[i]) > 0)
			nSV++;
	}
	for(i=0;i<l;i++)
		v -= alpha[i*nr_class+(int)prob->y[i]];
	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] d_ind;
	delete [] d_val;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;
}

// A coordinate descent algorithm for 
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix 
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
// 
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svc(
	const problem *prob, double *w, double eps, 
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1; 
		}
		else
		{
			y[i] = -1;
		}
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for(i=0; i<l; i++)
		alpha[i] = 0;
	
	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			w[xi->index-1] += y[i]*alpha[i]*val;
			xi++;
		}
		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			G = 0;
			schar yi = y[i];

			feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{
				G += w[xi->index-1]*(xi->value);
				xi++;
			}
			G = G*yi-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				xi = prob->x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}


// A coordinate descent algorithm for 
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
// 
//  where Qij = xi^T xj and
//  D is a diagonal matrix 
//
// In L1-SVM case:
// 		upper_bound_i = C
// 		lambda_i = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		lambda_i = 1/(2*C)
//
// Given: 
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012   

#undef GETI
#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svr(
	const problem *prob, double *w, const parameter *param,
	int solver_type)
{
	int l = prob->l;
	double C = param->C;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init;
	double *beta = new double[l];
	double *QD = new double[l];
	double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	double lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = INF;

	if(solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = 0;
		feature_node *xi = prob->x[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			w[xi->index-1] += beta[i]*val;
			xi++;
		}

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			feature_node *xi = prob->x[i];
			while(xi->index != -1)
			{
				int ind = xi->index-1;
				double val = xi->value;
				G += val*w[ind];
				xi++;
			}

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
			{
				xi = prob->x[i];
				while(xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	info("Objective value = %lf\n", v);
	info("nSV = %d\n",nSV);

	delete [] beta;
	delete [] QD;
	delete [] index;
}


// A coordinate descent algorithm for 
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
//    s.t.      0 <= \alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and 
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 5 of Yu et al., MLJ 2010

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double *xTx = new double[l];
	int max_iter = 1000;
	int *index = new int[l];		
	double *alpha = new double[2*l]; // store alpha and C - alpha
	schar *y = new schar[l];	
	int max_inner_iter = 100; // for inner Newton
	double innereps = 1e-2; 
	double innereps_min = min(1e-8, eps);
	double upper_bound[3] = {Cn, 0, Cp};

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1; 
		}
		else
		{
			y[i] = -1;
		}
	}
		
	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		xTx[i] = 0;
		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			xTx[i] += val*val;
			w[xi->index-1] += y[i]*alpha[2*i]*val;
			xi++;
		}
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}
		int newton_iter = 0;
		double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			schar yi = y[i];
			double C = upper_bound[GETI(i)];
			double ywTx = 0, xisq = xTx[i];
			feature_node *xi = prob->x[i];
			while (xi->index != -1)
			{
				ywTx += w[xi->index-1]*xi->value;
				xi++;
			}
			ywTx *= y[i];
			double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0) 
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			double alpha_old = alpha[ind1];
			double z = alpha_old;
			if(C - z < 0.5 * C) 
				z = 0.1*z;
			double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const double eta = 0.1; // xi in the paper
			int inner_iter = 0;
			while (inner_iter <= max_inner_iter) 
			{
				if(fabs(gp) < innereps)
					break;
				double gpp = a + C/(C-z)/z;
				double tmpz = z - gp/gpp;
				if(tmpz <= 0) 
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				xi = prob->x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += sign*(z-alpha_old)*yi*xi->value;
					xi++;
				}  
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gmax < eps) 
			break;

		if(newton_iter <= l/10) 
			innereps = max(innereps_min, 0.1*innereps);

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

	// calculate objective value
	
	double v = 0;
	for(i=0; i<w_size; i++)
		v += w[i] * w[i];
	v *= 0.5;
	for(i=0; i<l; i++)
		v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1]) 
			- upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
	info("Objective value = %lf\n", v);

	delete [] xTx;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// A coordinate descent algorithm for 
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_l2_svc(
	problem *prob_col, double *w, double eps, 
	double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int max_iter = 1000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init;
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];
	feature_node *x;

	clock_t cpu_now;
	double cpu_time_elapsed;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}
	for(j=0; j<w_size; j++)
	{
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x->value *= y[ind]; // x->value stores yi*xij
			double val = x->value;
			b[ind] -= w[j]*val;
			xj_sq[j] += C[GETI(ind)]*val*val;
			x++;
		}
	}

	if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,0); save_model(filenametosave,model_);}
	if (param->chat_level > 0)
	{
		cpu_now = clock();
		cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
		double objective_value = 0;
		for(j=0; j<w_size; j++)
			if(w[j] != 0)
				objective_value += fabs(w[j]);
		for(j=0; j<l; j++)
			if(b[j] > 0)
				objective_value += C[GETI(j)]*b[j]*b[j];
		__QMMCHAT__
		cpu_begin += (clock() - cpu_now);
	}

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[GETI(ind)]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*w[j])
				d = -Gp/H;
			else if(Gn > H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					while(x->index != -1)
					{
						b[x->index-1] += d_diff*x->value;
						x++;
					}
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						if(b[ind] > 0)
							loss_old += C[GETI(ind)]*b[ind]*b[ind];
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					while(x->index != -1)
					{
						b[x->index-1] -= w[i]*x->value;
						x++;
					}
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;

		iter++;
		
		if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,iter); save_model(filenametosave,model_);}
		if (param->chat_level > 0)
		{
			cpu_now = clock();
			cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
			double objective_value = 0;
			for(j=0; j<w_size; j++)
				if(w[j] != 0)
					objective_value += fabs(w[j]);
			for(j=0; j<l; j++)
				if(b[j] > 0)
					objective_value += C[GETI(j)]*b[j]*b[j];
			__QMMCHAT__
			cpu_begin += (clock() - cpu_now);
		}

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[GETI(j)]*b[j]*b[j];

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

// A coordinate descent algorithm for 
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr(
	const problem *prob_col, double *w, double eps, 
	double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, newton_iter=0, iter=0;
	int max_newton_iter = 100;
	int max_iter = 1000;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double w_norm, w_norm_new;
	double z, G, H;
	double Gnorm1_init;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = INF;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *Hdiag = new double[w_size];
	double *Grad = new double[w_size];
	double *wpd = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xTd = new double[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *tau = new double[l];
	double *D = new double[l];
	feature_node *x;

	clock_t cpu_now;
	double cpu_time_elapsed;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;

		exp_wTx[j] = 0;
	}

	w_norm = 0;
	for(j=0; j<w_size; j++)
	{
		w_norm += fabs(w[j]);
		wpd[j] = w[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			double val = x->value;
			exp_wTx[ind] += w[j]*val;
			if(y[ind] == -1)
				xjneg_sum[j] += C[GETI(ind)]*val;
			x++;
		}
	}
	for(j=0; j<l; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C[GETI(j)]*tau_tmp;
		D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
	}
	
	if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,0); save_model(filenametosave,model_);}
	if (param->chat_level > 0)
	{
		cpu_now = clock();
		cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
		int _iter = iter;
		iter = newton_iter;
		double objective_value = 0;
		for(j=0; j<w_size; j++)
			if(w[j] != 0)
				objective_value += fabs(w[j]);
		for(j=0; j<l; j++)
			if(y[j] == 1)
				objective_value += C[GETI(j)]*log(1+1/exp_wTx[j]);
			else
				objective_value += C[GETI(j)]*log(1+exp_wTx[j]);
		__QMMCHAT__
		iter = _iter;
		cpu_begin += (clock() - cpu_now);
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			Hdiag[j] = nu;
			Grad[j] = 0;

			double tmp = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				Hdiag[j] += x->value*x->value*D[ind];
				tmp += x->value*tau[ind];
				x++;
			}
			Grad[j] = -tmp + xjneg_sum[j];

			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(newton_iter > 0 && param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,newton_iter); save_model(filenametosave,model_);}
		if(newton_iter > 0 && param->chat_level > 0)
		{
			cpu_now = clock();
			cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
			int _iter = iter;
			iter = newton_iter;
			double objective_value = 0;
			for(j=0; j<w_size; j++)
				if(w[j] != 0)
					objective_value += fabs(w[j]);
			for(j=0; j<l; j++)
				if(y[j] == 1)
					objective_value += C[GETI(j)]*log(1+1/exp_wTx[j]);
				else
					objective_value += C[GETI(j)]*log(1+exp_wTx[j]);
			__QMMCHAT__
			iter = _iter;
			cpu_begin += (clock() - cpu_now);
		}

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(j=0; j<QP_active_size; j++)
			{
				int i = j+rand()%(QP_active_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<QP_active_size; s++)
			{
				j = index[s];
				H = Hdiag[j];

				x = prob_col->x[j];
				G = Grad[j] + (wpd[j]-w[j])*nu;
				while(x->index != -1)
				{
					int ind = x->index-1;
					G += x->value*D[ind]*xTd[ind];
					x++;
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(wpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[j])
					z = -Gp/H;
				else if(Gn > H*wpd[j])
					z = -Gn/H;
				else
					z = -wpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[j] += z;

				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index-1;
					xTd[ind] += x->value*z;
					x++;
				}
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		if(iter >= max_iter)
			info("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		w_norm_new = 0;
		for(j=0; j<w_size; j++)
		{
			delta += Grad[j]*(wpd[j]-w[j]);
			if(wpd[j] != 0)
				w_norm_new += fabs(wpd[j]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(int i=0; i<l; i++)
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<l; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(j=0; j<w_size; j++)
					w[j] = wpd[j];
				for(int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D[i] = C[GETI(i)]*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(j=0; j<w_size; j++)
				{
					wpd[j] = (w[j]+wpd[j])*0.5;
					if(wpd[j] != 0)
						w_norm_new += fabs(wpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				while(x->index != -1)
				{
					exp_wTx[x->index-1] += w[i]*x->value;
					x++;
				}
			}

			for(int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;

		info("iter %3d  #CD cycles %d\n", newton_iter, iter);
	}

	info("=========================\n");
	info("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		info("WARNING: reaching max number of iterations\n");

	// calculate objective value
	
	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	for(j=0; j<l; j++)
		if(y[j] == 1)
			v += C[GETI(j)]*log(1+1/exp_wTx[j]);
		else
			v += C[GETI(j)]*log(1+exp_wTx[j]);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] Hdiag;
	delete [] Grad;
	delete [] wpd;
	delete [] xjneg_sum;
	delete [] xTd;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] tau;
	delete [] D;
}

// transpose matrix X from row format to column format
static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	int nnz = 0;
	int *col_ptr = new int[n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn, const struct model * model_)
{
	double eps=param->eps;
	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i] > 0)
			pos++;
	neg = prob->l - pos;
	
	double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;

	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2R_LR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol);
			tron_obj.set_print_string(liblinear_print_string);
			clock_t cpu_begin = clock();
			tron_obj.tron(w, param, model_, cpu_begin);
			delete fun_obj;
			delete C;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol);
			tron_obj.set_print_string(liblinear_print_string);
			clock_t cpu_begin = clock();
			tron_obj.tron(w, param, model_, cpu_begin);
			delete fun_obj;
			delete C;
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case L1R_L2LOSS_SVC:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			clock_t cpu_begin = clock();
			solve_l1r_l2_svc(&prob_col, w, primal_solver_tol, Cp, Cn, param, model_, cpu_begin);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L1R_LR:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			clock_t cpu_begin = clock();
			solve_l1r_lr(&prob_col, w, primal_solver_tol, Cp, Cn, param, model_, cpu_begin);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L2R_LR_DUAL:
			solve_l2r_lr_dual(prob, w, eps, Cp, Cn);
			break;
		case L2R_L2LOSS_SVR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
				C[i] = param->C;
			
			fun_obj=new l2r_l2_svr_fun(prob, C, param->p);
			TRON tron_obj(fun_obj, param->eps);
			tron_obj.set_print_string(liblinear_print_string);
			clock_t cpu_begin = clock();
			tron_obj.tron(w, param, model_, cpu_begin);
			delete fun_obj;
			delete C;
			break;

		}
		case L2R_L1LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L1LOSS_SVR_DUAL);
			break;
		case L2R_L2LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L2LOSS_SVR_DUAL);
			break;
		case MMCD:
			{
				problem prob_col;
				feature_node *x_space = NULL;
				transpose(prob, &x_space ,&prob_col);
				clock_t cpu_begin = clock();
				solve_mmcd(&prob_col, w, eps, Cp, Cn, param, model_, cpu_begin);
				delete [] prob_col.y;
				delete [] prob_col.x;
				delete [] x_space;
				break;
			}
		case MMCD_SIMPLE:
			{
				problem prob_col;
				feature_node *x_space = NULL;
				transpose(prob, &x_space ,&prob_col);
				clock_t cpu_begin = clock();
				solve_mmcd_simple(&prob_col, w, eps, Cp, Cn, param, model_, cpu_begin);
				delete [] prob_col.y;
				delete [] prob_col.x;
				delete [] x_space;
				break;
			}	
		case MMGCD:
			{
				problem prob_col;
				feature_node *x_space = NULL;
				transpose(prob, &x_space ,&prob_col);
				clock_t cpu_begin = clock();
				solve_mmgcd(&prob_col, w, eps, Cp, Cn, param, model_, cpu_begin);
				delete [] prob_col.y;
				delete [] prob_col.x;
				delete [] x_space;
				break;
			}	
		case MMCG:
			{
				clock_t cpu_begin = clock();
				solve_mmcg(prob, w, eps, Cp, Cn, param, model_, cpu_begin);
				break;
			}	
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
	}
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	if(param->solver_type == L2R_L2LOSS_SVR ||
	   param->solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param->solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		model_->w = Malloc(double, w_size);
		model_->nr_class = 2;
		model_->label = NULL;
		train_one(prob, param, &model_->w[0], 0, 0, model_);
	}
	else
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];

		// multi-class svm by Crammer and Singer
		if(param->solver_type == MCSVM_CS)
		{
			model_->w=Malloc(double, n*nr_class);
			for(i=0;i<nr_class;i++)
				for(j=start[i];j<start[i]+count[i];j++)
					sub_prob.y[j] = i;
			Solver_MCSVM_CS Solver(&sub_prob, nr_class, weighted_C, param->eps);
			Solver.Solve(model_->w);
		}
		else if(param->solver_type == MMCD_SM)
		{
			model_->w=Malloc(double, n*nr_class);
			for(i=0;i<nr_class;i++)
				for(j=start[i];j<start[i]+count[i];j++)
					sub_prob.y[j] = i;

			problem prob_col;
			feature_node *x_space = NULL;
			transpose(&sub_prob, &x_space ,&prob_col);
		
			solve_mmcd_sm(&prob_col, param,model_->w, weighted_C[0],start,count,nr_class);
			
			//delete [] prob_col.y;
			//delete [] prob_col.x;
			//delete [] x_space;
		}
		else
		{
			if(nr_class == 2)
			{
				model_->w=Malloc(double, w_size);

				int e0 = start[0]+count[0];
				k=0;
				for(; k<e0; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1],model_);
			}
			else
			{
				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class;i++)
				{
					int si = start[i];
					int ei = si+count[i];

					k=0;
					for(; k<si; k++)
						sub_prob.y[k] = -1;
					for(; k<ei; k++)
						sub_prob.y[k] = +1;
					for(; k<sub_prob.l; k++)
						sub_prob.y[k] = -1;

					train_one(&sub_prob, param, w, weighted_C[i], param->C,model_);

					for(int j=0;j<w_size;j++)
						model_->w[j*nr_class+i] = w[j];
				}
				free(w);
			}

		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
		free(weighted_C);
	}
	return model_;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);

	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		if(model_->param.solver_type == L2R_L2LOSS_SVR ||
		   model_->param.solver_type == L2R_L1LOSS_SVR_DUAL ||
		   model_->param.solver_type == L2R_L2LOSS_SVR_DUAL)
			return dec_values[0];
		else
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		double label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;		
	}
	else
		return 0;
}

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"", "", "",
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", 
	"", "", "","", "", "","",
	"", "", "","", "", "","","", "", "",
	"", "", "","", "", "","","", "", "",
	"", "", "","", "", "","","", "",
	"MMCD", "MMCD_SM", "MMCD_SG", "MMCD_SIMPLE", "MMGCD", "MMCG",
	NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
	
	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", param.real_dim);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	if(param.structured_w=='f')
	{
		double ** outputw = new double*[param.tmx_c];
		for(i=0; i<param.tmx_c; i++)
			outputw[i] = new double[nr_w];
		
		for(i=0; i<nr_w; i++) //classes
		{
			for(int j=0; j<param.tmx_c; j++)
			{
				double wtoput = 0;
				for(int k=0; k<w_size;k++)
					wtoput += param.transform_matrix[k][j]*model_->w[k*nr_w+i];
				outputw[j][i] = wtoput;
			}
		}
		
		for(int k=0; k<param.tmx_c;k++)
		{
			for(i=0; i<nr_w; i++) 
				fprintf(fp, "%.16g ", outputw[k][i]);
			fprintf(fp, "\n");
		}
		
		for(i=0; i<param.tmx_c; i++)
			delete[] outputw[i];
		delete[] outputw;
	}
	else
	{
		for(i=0; i<w_size; i++)
		{
			int j;
			for(j=0; j<nr_w; j++)
				fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
			fprintf(fp, "\n");
			if(i==nr_feature-1 && (param.structured_w=='s' || param.structured_w=='a'))
			{
				int k;
				for(k=nr_feature-(param.real_dim%2==0?1:2); k>=0; k--)
				{
					int j;
					for(j=0; j<nr_w; j++) 
						fprintf(fp, "%.16g ", (param.structured_w=='a'?-1:1) * model_->w[k*nr_w+j]);
					fprintf(fp, "\n");
				}
			}
		}
	}
	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;
	
	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				
				setlocale(LC_ALL, old_locale);
				free(model_->label);
				free(model_);
				free(old_locale);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			setlocale(LC_ALL, old_locale);
			free(model_->label);
			free(model_);
			free(old_locale);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}
	
	setlocale(LC_ALL, old_locale);
	free(old_locale);
	
	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";
	
	if(param->p < 0)
		return "p < 0";

	if(param->solver_type != L2R_LR
		&& param->solver_type != L2R_L2LOSS_SVC_DUAL
		&& param->solver_type != L2R_L2LOSS_SVC
		&& param->solver_type != L2R_L1LOSS_SVC_DUAL
		&& param->solver_type != MCSVM_CS
		&& param->solver_type != L1R_L2LOSS_SVC
		&& param->solver_type != L1R_LR
		&& param->solver_type != L2R_LR_DUAL
		&& param->solver_type != L2R_L2LOSS_SVR
		&& param->solver_type != L2R_L2LOSS_SVR_DUAL
		&& param->solver_type != L2R_L1LOSS_SVR_DUAL
		&& param->solver_type != MMCD
		&& param->solver_type != MMCD_SM
		&& param->solver_type != MMCD_SG
		&& param->solver_type != MMGCD
		&& param->solver_type != MMCD_SIMPLE
		&& param->solver_type != MMCG)
		return "unknown solver type";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_LR ||
			model_->param.solver_type==L2R_LR_DUAL ||
			model_->param.solver_type==L1R_LR);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL) 
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

// A coordinate descent algorithm based on MMCD 
// any loss (param loss_type)
// and L1, L2 or elastic net regularizer (param alpha)
//
//  min_w \sum {(1-alpha)|wj| + alpha wj^2 }  + C \sum max(0, 1-yi w^T xi)^2,
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

static double mysign(double w) {
	 return w > 0 ? 1.0 : (w < 0 ? -1.0 : 0.0);
}

static double shrink(double w, double gamma) {
	if (w < gamma && w > -gamma)
		return 0;
	else if (w < 0)
		return w+gamma;
	else if (w > 0)
		return w-gamma;
	return 0;
}

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static double mmcd_sm_obj(problem *prob_col,const parameter *param,double *w, double *z,  double C, int *nnzp, int num_class, double &reg_value) {

	loss_var loss_type = param->loss_type;
	double loss_param = param->loss_param;
	double reg_param = param->reg_param;
	int l = prob_col->l;  // N in paper
	int w_size = prob_col->n*num_class;  // M in paper

	double v = 0;
	reg_value = 0;
	int nnz = 0;
	int j;

	for(j=0; j<w_size; j++)
	{
		if(w[j] != 0)
		{
			reg_value += (1-reg_param)*fabs(w[j]) + reg_param * 0.5 *w[j]*w[j];
			nnz++;
		}
	}
	if (loss_type == L2) {
		for(j=0; j<l; j++)
			if(z[j] < 1)
				v += C*(1-z[j])*(1-z[j]);
	} else if (loss_type == L1) {
		for(j=0; j<l; j++)
			if(z[j] < 1)
				v += C*(1-z[j]);
	} else if (loss_type == HU1) {
		for(j=0; j<l; j++)
			if(z[j] < 1-loss_param)
				v += C*(1-z[j]);
			else if (z[j] < 1 + loss_param)
				v += C*1/(4*loss_param)*(z[j]-1-loss_param)*(z[j]-1-loss_param);
	} else if (loss_type == HU2) {
		for(j=0; j<l; j++)
			if(z[j] < 1-loss_param)
				v += C*(1-z[j]-loss_param/2);
			else if (z[j] < 1)
				v += C*1/(2*loss_param)*(z[j]-1)*(z[j]-1);
	} else if (loss_type == LOG) {
		for(j=0; j<l; j++)
			v += C*log(1+exp(-z[j]));
	} else if (loss_type == LS) {
		for(j=0; j<l; j++)
			v += C*(1-z[j])*(1-z[j]);
	}
	*nnzp = nnz;
	return v;
}

bool custom_isnan(double var)
{
    volatile double d = var;
    return d != d;
}

static void solve_mmcd_sm(
					   problem *prob_col, const parameter *param ,
					   double *w, double C, int *start, int *count, int nr_class)
{

	int l = prob_col->l;  // N in paper
	int w_size = prob_col->n*nr_class;  // M x nr_class
	int w_size2 = prob_col->n;// M in paper
	int i,j,k, s, iter = 0;
	int max_iter = 1000;
	int active_size = w_size;
	double eps = param->eps;
	//double eps = 0.0001;
	double d;
	double Gmax_old = 0;
	double Gmax_new;
	double wmax_old = INF;
	double wmax_new;
	double Dmax_old = INF;
	double Dmax_new;
	double G2_old = INF;
	double G2_new;
	double Gmax_init;


	int *index = new int[w_size];
	double *z = new double[l];		// z in paper
	double *hdot = new double[l];	// hdot
	double *curv = new double[l];	// curv
	double *xj_sq = new double[w_size];
	double *Djpp = new double[w_size];
	double *Diff = new double[w_size];
	double **ewx = Malloc(double *,l);
	double *wixi = Malloc(double,l);
	double *lse = Malloc(double,l);
	double delta_w ;
	feature_node *x;
	bool PREC = false, DENUPDATE = true;
	loss_var loss_type = param->loss_type;
	curv_var curv_type = param->curv_type;
	double loss_param = param->loss_param;
	double curv_param = param->curv_param;
	double reg_param = param->reg_param;
	struct model* model_init = NULL;
	double v;
	clock_t cpu_now, cpu_begin;
	double cpu_time_elapsed = 0;
	double *y = prob_col->y;
	double temp_double;
	int nnz;
	int Maxruns = 1;

	cpu_begin = clock();
	if (param->init_model_file) {
		if((model_init=load_model(param->init_model_file))==0)
		{
			fprintf(stderr,"can't open model file %s\n",param->init_model_file);
			exit(1);
		}
	}

	// initialize w and z either from a model file or from w=0
	if (model_init) {
		if (model_init->nr_feature != w_size2) { info("initial model file weight dimension does not match the data."); exit(0);}

		for(j=0; j<w_size; j++)
		{
			w[j] = model_init->w[j]; // we initialize with model w
		}
		for(i=0;i<l;i++)
		{
			wixi[i] = 0;
		}

		for(i=0; i<l; i++)
		{
			double *temp_array = Malloc(double,w_size2);
			for (k=0;k<w_size2;k++)
			{
				temp_array[k] = 1;
			}
			ewx[i] = temp_array;
		}

		for(j=0;j<nr_class;j++)
		{
			int sj = start[j];
			int ej = sj+count[j];

			for(k=0;k<w_size2;k++)
			{
				s = j*nr_class+k;
				x = prob_col->x[k];
				i=0;
				for(;i<sj;i++)
				{
					double val=x->value;
					ewx[i][j]*=exp(w[s]*val);
					x++;
				}
				for(; i<ej; i++)
				{
					double val=x->value;
					wixi[i]+=val*w[s];
					ewx[i][j]*=exp(w[s]*val);
					x++;
				}
				for(;i<l;i++)
				{
					double val=x->value;
					ewx[i][j]*=exp(w[s]*val);
					x++;
				}
			}
		}

		for(i=0;i<l;i++)
		{
			lse[i] = 0;
			for(j=0;j<nr_class;j++)
			{
				lse[i]+=ewx[i][j];
			}
			lse[i]-=exp(wixi[i]);
			z[i] = wixi[i] - log(lse[i]);
		}

		
	} else {
		temp_double = -log(double(nr_class-1));
		for(j=0; j<l; j++)
		{
			z[j] = temp_double;  // we  initialize with w=0
			wixi[j] = 0;
			lse[j] = double(nr_class-1);
			double *temp_array = Malloc(double,nr_class);
			for (k=0;k<nr_class;k++)
			{
				temp_array[k] = 1;
			}
			ewx[j] = temp_array;
		}

		for(j=0; j<w_size2; j++)
		{
			index[j] = j;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				double val = x->value;
				x++;
			}
		}
		for (i=0; i<w_size;i++)
		{
			w[i] = 0;
		}
	}

	// calculate objective value
	if (param->chat_level > 2) {
		double reg_value;
		//printf("Loss type:%d, curvature type:%d\nReg_param:%.1e, loss_param:%.1e, curv_param:%.1e\n",loss_type, curv_type, reg_param, loss_param, curv_param);
		v = mmcd_sm_obj(prob_col,param,w,z,C,&nnz,nr_class,reg_value);
		printf("loss: %lf, reg: %lf, Obj: %lf\n", v,reg_value,v+reg_value);
	}

	if (curv_type == MC) {
		PREC = true;
		if (loss_type == L2) {
			//for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
			for(j=0; j<l; j++) curv[j] = 2.0;
		}else if (loss_type == L1){
			//for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/loss_param;
			for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == HU1){
			//for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/(2.0*loss_param);
			for(j=0; j<l; j++) curv[j] = 1.0/(2.0*loss_param);
		}else if (loss_type == HU2){
			//for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1/loss_param;
			for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == LOG){
			//for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/4.0;
			for(j=0; j<l; j++) curv[j] = 1.0/4.0;
		}
	}

	if (loss_type == LS) {
		PREC = true;
		//for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	} else if (loss_type == L2 && curv_type == OC) {
		PREC = true;
		//for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	}

	while(iter < max_iter)
	{
		//printf("Iter: %d\n",iter);
 		Gmax_new = 0;
		wmax_new = 0;
		Dmax_new = 0;
		G2_new = 0;

		if (loss_type == L2) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -2.0*(1.0-z[j]) : 0; }
		} else if (loss_type == L1) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -1.0 : 0; }
		} else if (loss_type == HU1) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1 + t) ? 1.0/(2.0*t)*(z[j]-1.0-t) : 0.0; }
		} else if (loss_type == HU2) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1) ? 1.0/(t)*(z[j]-1) : 0; }
		} else if (loss_type == LOG) {
			for(j=0; j<l; j++) { hdot[j] = -1.0/(1+exp(z[j])); }
		} else if (loss_type == LS) {
			for(j=0; j<l; j++) { hdot[j] = -2.0*(1-z[j]); }
		}
		

		// compute curv's if necessary
		if (PREC==false) {
			if (loss_type == L2 && curv_type == NC) {
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (z[j] < 1) ? 2.0 : epsi; }
			} else if (loss_type == L1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == L1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == HU1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU2 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : 1 / (2*fabs(1-t/2-z[j]));}
			} else if (loss_type == HU2 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : epsi;}
			} else if (loss_type == LOG && curv_type == OC) {
				for(j=0; j<l; j++) { 
					double ez = exp(z[j]);
					double emz = exp(-z[j]);
					double eme = ez - emz;
					curv[j] = (fabs(eme)<1e-3) ? 1.0/4.0 : fabs(eme/z[j])/(2*(2+ez+emz));
				}
			} else if (loss_type == LOG && curv_type == NC) {
				for(j=0; j<l; j++) { 
					double ee = exp(z[j])+exp(-z[j]);
					curv[j] = 1.0/(2+ee);
				}
			} else {
				info("invalid loss and curv type combination\n");
				exit(0); //revisit
			}
		}


		for(int run=0; run < Maxruns; run++) {
			for(s=0; s<w_size; s++)
			{
				k = s/nr_class;
				j = s%nr_class;

				/*printf("\ns:%i, lse values:\n\n",s);
				for (i=0;i<l;i++)
				{
					printf("%f ",lse[i]);
				}
				printf("\n");*/

				int sj = start[j];
				int ej = sj+count[j];

				/*printf("\ns:%i, wixi values:\n\n",s);
				for (i=0;i<l;i++)
				{
					printf("%f ",wixi[i]);
				}
				printf("\n");*/

				/*printf("\ns:%i, wixi values:\n\n",s);
				for (i=0;i<l;i++)
				{
					for(int i2 = 0; i2<w_size2;i2++)
						printf("%f ",ewx[i][i2]);
				}
				printf("\n");*/

				double Djp = 0;
				double Djppnow = 0;

				x = prob_col->x[k];
				/*printf("wixi values for s=%i:\n",s);
				for (i=0;i<l;i++)
					printf("%f ",wixi[i]);*/

				//printf("\n\ns: %i\n\n\n",s);
				i=0;
				for(;i<sj;i++)
				{
					double val=x->value;
					Djp -= (ewx[i][j]*val/lse[i])*(hdot[i]+ curv[i] *(wixi[i]-log(lse[i])-z[i]));
					Djppnow += curv[i] *val*val*ewx[i][j]*ewx[i][j] / (lse[i]*lse[i]); //assume that second derivative of z is zero
					//Djppnow += val*val*ewx[i][j]*( curv[i]*ewx[i][j]   + (hdot[i] + curv[i]*(wixi[i] - log(lse[i]) - z[i]))*(lse[i] - ewx[i][j])   )   / (lse[i]*lse[i]); //no approximation
					//printf("i: %i, j: %i, k: %i, Djp: %f, Djpp: %f\n",i,j,k,Djp,Djppnow);
					x++;
				}
				for(; i<ej; i++)
				{
					double val=x->value;
					Djp += val*(hdot[i]+curv[i]*(wixi[i]-log(lse[i])-z[i]));
					Djppnow += curv[i]*val*val;
					//printf("i: %i, j: %i, k: %i, Djp: %f, Djpp: %f\n",i,j,k,Djp,Djppnow);
					x++;
				}
				for(;i<l;i++)
				{
					double val=x->value;
					Djp -= (ewx[i][j]*val/lse[i])*(hdot[i]+ curv[i] *(wixi[i]-log(lse[i])-z[i]));
					Djppnow += curv[i] *val*val*ewx[i][j]*ewx[i][j] / (lse[i]*lse[i]); //assume that second derivative of z is zero
					//Djppnow += val*val*ewx[i][j]*( curv[i]*ewx[i][j]   + (hdot[i] + curv[i]*(wixi[i] - log(lse[i]) - z[i]))*(lse[i] - ewx[i][j])   )   / (lse[i]*lse[i]); //no approximation
					//printf("i: %i, j: %i, k: %i, Djp: %f, Djpp: %f\n",i,j,k,Djp,Djppnow);
					x++;
				}
				Djp*=C;
				Djppnow*=C;
				/*for(i=0;i<l;i++)
					printf("curv: %f ",curv[i]);

				printf("\n");*/
				//Djp = Djp/l;
				//Djppnow = Djppnow/l;
				//printf("Djp: %f, Djpp: %f, current w:%f\n",Djp,Djppnow,w[s]); 

				if (reg_param > 0) {
					Djp += reg_param * w[s];
					Djppnow += reg_param;
				}

				// newton step

				d = -Djp/Djppnow;

				//printf("s: %i, d: %f\n",s,d);
				if(fabs(d) < 1.0e-12)
					continue;

				//printf("FINAL Djp: %f, Djpp: %f d: %f\n",Djp,Djppnow,d);
				double wjhat = w[s] + d;

				if (reg_param < 1) {
					wjhat = shrink(wjhat, (1-reg_param)/(Djppnow));
				}
				delta_w = wjhat - w[s];
				w[s]=wjhat;

				double adifwj = fabs(delta_w);

				//Diff[j] = adifwj;

				//printf("delta_w: %f\n",delta_w);

				Dmax_new = max(Dmax_new, adifwj);
				wmax_new = max(wmax_new, fabs(wjhat));

				x = prob_col->x[k];
				i=0;
				for(;i<sj;i++)
				{
					double val=x->value;
					lse[i]-=ewx[i][j];
					ewx[i][j]*=exp(delta_w*val);
					lse[i]+=ewx[i][j];
					x++;
				}
				for(; i<ej; i++)
				{
					double val=x->value;
					wixi[i]+=delta_w*val;
					ewx[i][j] = exp(wixi[i]);
					x++;
				}
				for(;i<l;i++)
				{
					double val=x->value;
					lse[i]-=ewx[i][j];
					ewx[i][j]*=exp(delta_w*val);
					lse[i]+=ewx[i][j];
					x++;
				}
				/*printf("\ns:%i, lse values:\n\n",s);
				for (i=0;i<l;i++)
				{
					printf("%f ",lse[i]);
				}
				printf("\n");*/

				
			}
		}

		// // update z[i]
		
		for(int i=0; i<l; i++)
			z[i] = wixi[i]-log(lse[i]);

		if(Dmax_new <= eps*wmax_new)
			break;
		Dmax_old = Dmax_new;
		Gmax_old = Gmax_new;

		if(iter % 10 == 0) {
			if (param->chat_level == 0) // display a . every 10th when no other info displayed
				info(".");
		}

		// calculate objective value
		if (param->chat_level > 0) {
			cpu_now = clock();
			cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / CLOCKS_PER_SEC;
			//printf("Iter %d CPU_time %.3e", iter, cpu_time_elapsed);
			if (param->chat_level > 1) {
				//G2_new = sqrt(G2_new);
				//printf(" ||Grad||_inf %.3e act %d", Gmax_new, active_size);
				//printf(", ||Grad||_inf = %.3e, ||Grad||_2 = %.3e", Gmax_new, G2_new);
				if (param->chat_level > 2) {
					double reg_value;
					v = mmcd_sm_obj(prob_col,param,w,z,C,&nnz,nr_class,reg_value);
					printf("loss: %lf, reg: %lf, Obj: %lf\n",v,reg_value,v+reg_value);
					//printf(" Objective_value %lf", v);
				}
			}
			printf("\n");
		}
	}


	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double reg_value;
	v = mmcd_sm_obj(prob_col,param,w,z,C,&nnz,nr_class,reg_value);

	info("loss: %lf, reg: %lf, Obj: %lf\n", v,reg_value,reg_value+v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);


	

	delete [] index;
	delete [] y;
	delete [] z;
	delete [] hdot;
	delete [] curv;
	delete [] xj_sq;
	delete [] Djpp;
	delete [] ewx;
	delete [] wixi;
	delete [] lse;
}



#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static double mmcd_obj(const problem *prob_col,const parameter *param,double *w, double *z, schar *y, double *C, int *nnzp) {

	loss_var loss_type = param->loss_type;
	double loss_param = param->loss_param;
	double reg_param = param->reg_param;
	int l = prob_col->l;  // N in paper
	int w_size = prob_col->n;  // M in paper

	double v = 0;
	int nnz = 0;
	int j;

	for(j=0; j<w_size; j++)
	{
		//x = prob_col->x[j];
		//while(x->index != -1)
		//{
		//	x->value *= prob_col->y[x->index-1]; // restore x->value
		//	x++;
		//}
		if(w[j] != 0)
		{
			v += (1-reg_param)*fabs(w[j]) + reg_param * 0.5 *w[j]*w[j];
			nnz++;
		}
	}
	if (loss_type == L2) {
		for(j=0; j<l; j++)
			if(z[j] < 1)
				v += C[GETI(j)]*(1-z[j])*(1-z[j]);
	} else if (loss_type == L1) {
		for(j=0; j<l; j++)
			if(z[j] < 1)
				v += C[GETI(j)]*(1-z[j]);
	} else if (loss_type == HU1) {
		for(j=0; j<l; j++)
			if(z[j] < 1-loss_param)
				v += C[GETI(j)]*(1-z[j]);
			else if (z[j] < 1 + loss_param)
				v += C[GETI(j)]*1/(4*loss_param)*(z[j]-1-loss_param)*(z[j]-1-loss_param);
	} else if (loss_type == HU2) {
		for(j=0; j<l; j++)
			if(z[j] < 1-loss_param)
				v += C[GETI(j)]*(1-z[j]-loss_param/2);
			else if (z[j] < 1)
				v += C[GETI(j)]*1/(2*loss_param)*(z[j]-1)*(z[j]-1);
	} else if (loss_type == LOG) {
		for(j=0; j<l; j++)
			v += C[GETI(j)]*log(1+exp(-z[j]));
	} else if (loss_type == LS) {
		for(j=0; j<l; j++)
			v += C[GETI(j)]*(1-z[j])*(1-z[j]);
	}
	*nnzp = nnz;
	return v;
}


// mmcd without any tricks (to make it similar to the matlab code)
// and without random orders etc.
// and without stopping criterion (run up to max iter)
static void solve_mmcd_simple(
					   problem *prob_col, double *w, double eps, 
					   double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin)
{
	printf("\n......solving with mmcd_simple.......\n");
	int l = prob_col->l;  // N in paper
	int w_size = prob_col->n;  // M in paper
	int j, s, iter = 0;
	int max_iter = 1000;
	
	//double sigma = 0.01;
	double d;
	//double G_loss, G, H;
	double Gmax_old = 0;
	double Gmax_new;
	double wmax_old = INF;
	double wmax_new;
	double Dmax_old = INF;
	double Dmax_new;
	double G2_old = INF;
	double G2_new;
	double Gmax_init;
	//double d_old, d_diff;
	//double loss_old, loss_new;
	//double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *z = new double[l];		// z in paper
	double *hdot = new double[l];	// hdot
	double *qdot = new double[l];	// qdot
	double *curv = new double[l];	// curv
	double *xj_sq = new double[w_size];
	double *Djpp = new double[w_size];
	double *Diff = new double[w_size];
	feature_node *x;
	bool PREC = false, DENUPDATE = true;
	loss_var loss_type = param->loss_type;
	curv_var curv_type = param->curv_type;
	double loss_param = param->loss_param;
	double curv_param = param->curv_param;
	double reg_param = param->reg_param;
	struct model* model_init = NULL;
	double v;
	double & objective_value = v;
	clock_t cpu_now;
	double cpu_time_elapsed = 0;

	double C[3] = {Cn,0,Cp};
	int nnz;

	if (param->init_model_file) {
		if((model_init=load_model(param->init_model_file))==0)
		{
			fprintf(stderr,"can't open model file %s\n",param->init_model_file);
			exit(1);
		}
	}

	// initialize w and z either from a model file or from w=0
	if (model_init) {
		if (model_init->nr_feature != w_size) { info("initial model file weight dimension does not match the data."); exit(0);}

		for(j=0; j<l; j++)
		{
			z[j] = 0;  // we  initialize z vector with z=0
			if(prob_col->y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;
		}

		for(j=0; j<w_size; j++)
		{
			w[j] = model_init->w[j]; // we initialize with model w
			index[j] = j;
			xj_sq[j] = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				double val = x->value;
				x->value *= y[ind]; // x->value now stores yi*xij, xtilde in paper
				xj_sq[j] += C[GETI(ind)]*val*val;
				x++;
			}
		}
		// if w is nonzero, need to find initial z
		for(int i=0; i<w_size; i++)
		{	
			if(w[i]==0) continue;
			x = prob_col->x[i];
			while(x->index != -1)
			{
				z[x->index-1] += w[i]*x->value;
				x++;
			}
		}
	} else {
		for(j=0; j<l; j++)
		{
			z[j] = 0;  // we  initialize with w=0
			if(prob_col->y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;
		}

		for(j=0; j<w_size; j++)
		{
			w[j] = 0; // we  initialize with w=0
			index[j] = j;
			xj_sq[j] = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				double val = x->value;
				x->value *= y[ind]; // x->value now stores yi*xij, xtilde in paper
				xj_sq[j] += C[GETI(ind)]*val*val;
				x++;
			}
		}
	}

	if (curv_type == MC) {
		PREC = true;
		if (loss_type == L2) {
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
			for(j=0; j<l; j++) curv[j] = 2.0;
		}else if (loss_type == L1){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/loss_param;
			for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == HU1){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/(2.0*loss_param);
			for(j=0; j<l; j++) curv[j] = 1.0/(2.0*loss_param);
		}else if (loss_type == HU2){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1/loss_param;
			for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == LOG){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/4.0;
			for(j=0; j<l; j++) curv[j] = 1.0/4.0;
		}
	}

	if (loss_type == LS) {
		PREC = true;
		for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	} else if (loss_type == L2 && curv_type == OC) {
		PREC = true;
		for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	}

	if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,iter); save_model(filenametosave,model_);}
	if (param->chat_level > 0)
	{
		cpu_now = clock();
		cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
		v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);																										
		__QMMCHAT__
		cpu_begin += (clock() - cpu_now);
	}

	while(iter < max_iter)
	{
		if (loss_type == L2) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -2.0*(1.0-z[j]) : 0; }
		} else if (loss_type == L1) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -1.0 : 0; }
		} else if (loss_type == HU1) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1 + t) ? 1.0/(2.0*t)*(z[j]-1.0-t) : 0.0; }
		} else if (loss_type == HU2) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1) ? 1.0/(t)*(z[j]-1) : 0; }
		} else if (loss_type == LOG) {
			for(j=0; j<l; j++) { hdot[j] = -1.0/(1+exp(z[j])); }
		} else if (loss_type == LS) {
			for(j=0; j<l; j++) { hdot[j] = -2.0*(1-z[j]); }
		}
		// initialize qdot
		for(j=0; j<l; j++) { qdot[j] = hdot[j]; }

		// compute curv's if necessary
		if (PREC==false) {
			if (loss_type == L2 && curv_type == NC) {
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (z[j] < 1) ? 2.0 : epsi; }
			} else if (loss_type == L1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == L1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == HU1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU2 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : 1 / (2*fabs(1-t/2-z[j]));}
			} else if (loss_type == HU2 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : epsi;}
			} else if (loss_type == LOG && curv_type == OC) {
				for(j=0; j<l; j++) { 
					double ez = exp(z[j]);
					double emz = exp(-z[j]);
					double eme = ez - emz;
					curv[j] = (fabs(eme)<1e-3) ? 1.0/4.0 : fabs(eme/z[j])/(2*(2+ez+emz));
				}
			} else if (loss_type == LOG && curv_type == NC) {
				for(j=0; j<l; j++) { 
					double ee = exp(z[j])+exp(-z[j]);
					curv[j] = 1.0/(2+ee);
				}
			} else {
				info("invalid loss and curv type combination\n");
				exit(0); //revisit
			}
		}

		for(j=0; j<w_size; j++)
			{

				double Djp = 0;
				double Djppnow = 0;

				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index-1;
					if(qdot[ind] != 0)
					{
						double val = x->value;
						Djp += C[GETI(ind)]*val*qdot[ind];
					}
					x++;
				}

				if (PREC == true || DENUPDATE == false) {
					Djppnow = Djpp[j];
				} else {
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						double val = x->value;
						Djppnow += C[GETI(ind)]*val*val*curv[ind];
						Djpp[j] = Djppnow;
						x++;
					}
				}

				if (reg_param > 0) {
					Djp += reg_param * w[j];
					Djppnow += reg_param;
				}
				// newton step
				d = -Djp/Djppnow;

				double wjhat = w[j] + d;

				if (reg_param < 1) {
					wjhat = shrink(wjhat, (1-reg_param)/(Djppnow));
				}

				double difwj  = wjhat-w[j]; // difference
				w[j]=wjhat;

				// update qdot
				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index-1;
					double val = x->value;
					qdot[ind] += val*curv[ind]*difwj;
					x++;
				}
			}

		// // update z[i]
		if (curv_param > 1e-12) {
			for(j=0;j<l;j++)
				z[j] += (qdot[j]-hdot[j])/curv[j];
		} else { // if there is zero curvature value (possible for some loss if param->curv_param=0)
			info("#");
			for(int i=0; i<l; i++)
				z[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				while(x->index != -1)
				{
					z[x->index-1] += w[i]*x->value;
					x++;
				}
			}
		}


		iter++;
		
		if(iter % 10 == 0) {
			if (param->chat_level == 0) // display a . every 10th when no other info displayed
				info(".");
		} 
		
		if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,iter); save_model(filenametosave,model_);}
		if (param->chat_level > 0)
		{
			cpu_now = clock();
			cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
			v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);																										
			__QMMCHAT__
			cpu_begin += (clock() - cpu_now);
		}
	}


	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);


	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
	}

	delete [] index;
	delete [] y;
	delete [] z;
	delete [] hdot;
	delete [] qdot;
	delete [] curv;
	delete [] xj_sq;
	delete [] Djpp;
}


// solve mm with gcd; modified from solve_mmcd_simple
static void solve_mmgcd(
					   problem *prob_col, double *w, double eps, 
					   double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin)
{
	printf("\n......solving with mmgcd.......\n");
	int l = prob_col->l;  // N in paper
	int w_size = prob_col->n;  // M in paper
	int j, s, iter = 0;
	int max_iter = 1000;
	
	//double sigma = 0.01;
	double d;
	//double G_loss, G, H;
	double Gmax_old = 0;
	double Gmax_new;
	double wmax_old = INF;
	double wmax_new;
	double Dmax_old = INF;
	double Dmax_new;
	double G2_old = INF;
	double G2_new;
	double Gmax_init;
	//double d_old, d_diff;
	//double loss_old, loss_new;
	//double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *z = new double[l];		// z in paper
	double *hdot = new double[l];	// hdot
	double *qdot = new double[l];	// qdot
	double *curv = new double[l];	// curv
	double *xj_sq = new double[w_size];
	double *Djpp = new double[w_size];
	double *Diff = new double[w_size];
	feature_node *x;
	bool PREC = false, DENUPDATE = true;
	loss_var loss_type = param->loss_type;
	curv_var curv_type = param->curv_type;
	double loss_param = param->loss_param;
	double curv_param = param->curv_param;
	double reg_param = param->reg_param;
	struct model* model_init = NULL;
	double v;
	double & objective_value = v;
	clock_t cpu_now;
	double cpu_time_elapsed = 0;

	double C[3] = {Cn,0,Cp};
	int nnz;

	if (param->init_model_file) {
		if((model_init=load_model(param->init_model_file))==0)
		{
			fprintf(stderr,"can't open model file %s\n",param->init_model_file);
			exit(1);
		}
	}

	// initialize w and z either from a model file or from w=0
	if (model_init) {
		if (model_init->nr_feature != w_size) { info("initial model file weight dimension does not match the data."); exit(0);}

		for(j=0; j<l; j++)
		{
			z[j] = 0;  // we  initialize z vector with z=0
			if(prob_col->y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;
		}

		for(j=0; j<w_size; j++)
		{
			w[j] = model_init->w[j]; // we initialize with model w
			index[j] = j;
			xj_sq[j] = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				double val = x->value;
				x->value *= y[ind]; // x->value now stores yi*xij, xtilde in paper
				xj_sq[j] += C[GETI(ind)]*val*val;
				x++;
			}
		}
		// if w is nonzero, need to find initial z
		for(int i=0; i<w_size; i++)
		{	
			if(w[i]==0) continue;
			x = prob_col->x[i];
			while(x->index != -1)
			{
				z[x->index-1] += w[i]*x->value;
				x++;
			}
		}
	} else {
		for(j=0; j<l; j++)
		{
			z[j] = 0;  // we  initialize with w=0
			if(prob_col->y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;
		}

		for(j=0; j<w_size; j++)
		{
			w[j] = 0; // we  initialize with w=0
			index[j] = j;
			xj_sq[j] = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				double val = x->value;
				x->value *= y[ind]; // x->value now stores yi*xij, xtilde in paper
				xj_sq[j] += C[GETI(ind)]*val*val;
				x++;
			}
		}
	}

	if (curv_type == MC) {
		PREC = true;
		if (loss_type == L2) {
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
			for(j=0; j<l; j++) curv[j] = 2.0;
		}else if (loss_type == L1){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/loss_param;
			for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == HU1){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/(2.0*loss_param);
			for(j=0; j<l; j++) curv[j] = 1.0/(2.0*loss_param);
		}else if (loss_type == HU2){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1/loss_param;
			for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == LOG){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/4.0;
			for(j=0; j<l; j++) curv[j] = 1.0/4.0;
		}
	}

	if (loss_type == LS) {
		PREC = true;
		for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	} else if (loss_type == L2 && curv_type == OC) {
		PREC = true;
		for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	}

	// begin calculating groups
	std::vector<int*> matcheds;
	std::vector<int> unmatcheds;
	int *matched;
	double *d1x = new double[l];
	double *means = new double[w_size];
	feature_node *d1node, *d2node;
	double maxvar, thisvar;
	int maxvarindex;
	bool ismatched;
	
	for(int i=0; i<w_size; i++)
	{
		means[i] = 0;
		d1node = prob_col->x[i];
		while(d1node->index != -1)
		{
			means[i] += d1node->value;
			d1node++;
		}
		means[i] /= l;
	}
	
	for(int i=0; i<w_size; i++)
	{
		ismatched = false;
		for(int im=0; im<matcheds.size(); im++)
		{
			if(matcheds[im][1] == i)
			{
				ismatched = true;
				break;
			}			
		}
		if(ismatched)
			continue;
			
		for(int ind = 0; ind<l; ind++)
			d1x[ind] = 0;
		
		d1node = prob_col->x[i];
		while(d1node->index != -1)
		{
			d1x[d1node->index-1] = d1node->value;
			d1node++;
		}
		
		maxvar = -1;
		for(int i2=i+1; i2<w_size; i2++)
		{
			ismatched = false;
			for(int im=0; im<matcheds.size(); im++)
			{
				if(matcheds[im][1] == i2)
				{
					ismatched = true;
					break;
				}			
			}
			if(ismatched)
				continue;			
			
			thisvar = 0;
			d2node = prob_col->x[i2];
			while(d2node->index != -1)
			{
				//thisvar += (d1x[d2node->index-1] - means[i])*(d2node->value - means[i2]);
				thisvar += (d1x[d2node->index-1])*(d2node->value);
				d2node++;
			}
			if(fabs(thisvar)>maxvar)
			{
				maxvar = fabs(thisvar);
				maxvarindex = i2;
			}
		}
		
		if(maxvar != -1)
		{
			matched = new int[2];
			matched[0] = i;
			matched[1] = maxvarindex;
			matcheds.push_back(matched);
		}
		else
			unmatcheds.push_back(i);
	}
	
	for(int im=0; im<unmatcheds.size(); im++)
	{
		matched = new int[2];
		matched[0] = 0;
		matched[1] = unmatcheds[im];
		matcheds.push_back(matched);
	}

	std::vector<int*>  & groups = matcheds;
	delete[] means;
	delete[] d1x;
	// end calculating groups
	
	// begin variables for update
	double Hess[4]; // (1,1) (1,2) (2,1) (2,2)
	double detHess;
	double invHess[4]; // (1,1) (1,2) (2,1) (2,2)
	double b[2];
	double delw[2];
	double *xVec1 = new double[l], *xVec2 = new double[l];
	double *all_delw=new double[w_size];
	// end variables for update
	
	if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,iter); save_model(filenametosave,model_);}
	if (param->chat_level > 0)
	{
		cpu_now = clock();
		cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
		v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);																										
		__QMMCHAT__
		cpu_begin += (clock() - cpu_now);
	}

	while(iter < max_iter)
	{
		if (loss_type == L2) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -2.0*(1.0-z[j]) : 0; }
		} else if (loss_type == L1) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -1.0 : 0; }
		} else if (loss_type == HU1) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1 + t) ? 1.0/(2.0*t)*(z[j]-1.0-t) : 0.0; }
		} else if (loss_type == HU2) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1) ? 1.0/(t)*(z[j]-1) : 0; }
		} else if (loss_type == LOG) {
			for(j=0; j<l; j++) { hdot[j] = -1.0/(1+exp(z[j])); }
		} else if (loss_type == LS) {
			for(j=0; j<l; j++) { hdot[j] = -2.0*(1-z[j]); }
		}
		// initialize qdot
		for(j=0; j<l; j++) { qdot[j] = hdot[j]; }

		// compute curv's if necessary
		if (PREC==false) {
			if (loss_type == L2 && curv_type == NC) {
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (z[j] < 1) ? 2.0 : epsi; }
			} else if (loss_type == L1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == L1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == HU1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU2 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : 1 / (2*fabs(1-t/2-z[j]));}
			} else if (loss_type == HU2 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : epsi;}
			} else if (loss_type == LOG && curv_type == OC) {
				for(j=0; j<l; j++) { 
					double ez = exp(z[j]);
					double emz = exp(-z[j]);
					double eme = ez - emz;
					curv[j] = (fabs(eme)<1e-3) ? 1.0/4.0 : fabs(eme/z[j])/(2*(2+ez+emz));
				}
			} else if (loss_type == LOG && curv_type == NC) {
				for(j=0; j<l; j++) { 
					double ee = exp(z[j])+exp(-z[j]);
					curv[j] = 1.0/(2+ee);
				}
			} else {
				info("invalid loss and curv type combination\n");
				exit(0); //revisit
			}
		}

		Dmax_new = 0;
		wmax_new = 0;
		
		// reset some arrays
		for(int i=0; i<w_size; i++)
			all_delw[i]=0;
		
		// begin update groupwise
		for(int i=0; i<groups.size(); i++)
		{
			d1node = prob_col->x[matcheds[i][0]];
			d2node = prob_col->x[matcheds[i][1]];
			Hess[0] = reg_param; Hess[3] = reg_param; Hess[2] = 0; Hess[1] = 0; b[0] = reg_param*w[matcheds[i][0]]; b[1] = reg_param*w[matcheds[i][1]];
			while(1)
			{
				if(d1node->index == -1 && d2node->index == -1)
					break;
				else if(d1node->index != -1 && d2node->index != -1 && d1node->index==d2node->index)
				{
					Hess[0] += d1node->value * C[GETI(d1node->index-1)] * curv[d1node->index-1] * d1node->value;
					Hess[3] += d2node->value * C[GETI(d2node->index-1)] * curv[d2node->index-1] * d2node->value;
					Hess[2] += d1node->value * C[GETI(d1node->index-1)] * curv[d1node->index-1] * d2node->value;
					
					b[0] += C[GETI(d1node->index-1)] * d1node->value * qdot[d1node->index-1];
					b[1] += C[GETI(d2node->index-1)] * d2node->value * qdot[d2node->index-1];
	
					d1node++;
					d2node++;
				}
				else if(d2node->index != -1 && (d1node->index == -1 || d1node->index>d2node->index))
				{
					Hess[3] += d2node->value * C[GETI(d2node->index-1)] * curv[d2node->index-1] * d2node->value;
					b[1] += C[GETI(d2node->index-1)] * d2node->value * qdot[d2node->index-1];
					d2node++;
				}
				else if(d1node->index != -1 && (d2node->index == -1 || d1node->index<d2node->index))
				{
					Hess[0] += d1node->value * C[GETI(d1node->index-1)] * curv[d1node->index-1] * d1node->value;
					b[0] += C[GETI(d1node->index-1)] * d1node->value * qdot[d1node->index-1];
					d1node++;
				}
				else
				{
					printf("\ncheck indexes (1)\n");
					return;
				}
			}
			
			b[0] = -b[0];
			b[1] = -b[1];
			Hess[1] = Hess[2];
		
			detHess = Hess[0]*Hess[3] - Hess[1]*Hess[2];
			invHess[0] = (1/detHess) * Hess[3];
			invHess[1] = (1/detHess) * Hess[1] * (-1);
			invHess[2] = (1/detHess) * Hess[2] * (-1);
			invHess[3] = (1/detHess) * Hess[0];
			
			delw[0] = invHess[0] * b[0] + invHess[1] * b[1];
			delw[1] = invHess[2] * b[0] + invHess[3] * b[1];
			all_delw[matcheds[i][0]]+=delw[0];
			all_delw[matcheds[i][1]]+=delw[1];

			d1node = prob_col->x[matcheds[i][0]];
			d2node = prob_col->x[matcheds[i][1]];
			while(1)
			{
				if(d1node->index == -1 && d2node->index == -1)
					break;
				else if(d1node->index != -1 && d2node->index != -1 && d1node->index==d2node->index)
				{
					qdot[d1node->index-1] += delw[0] * d1node->value * curv[d1node->index-1] ;
					qdot[d2node->index-1] += delw[1] * d2node->value * curv[d2node->index-1] ;
					
					d1node++;
					d2node++;
				}
				else if(d2node->index != -1 && (d1node->index == -1 || d1node->index>d2node->index))
				{
					qdot[d2node->index-1] += delw[1] * d2node->value * curv[d2node->index-1] ;
					
					d2node++;
				}
				else if(d1node->index != -1 && (d2node->index == -1 || d1node->index<d2node->index))
				{
					qdot[d1node->index-1] += delw[0] * d1node->value * curv[d1node->index-1] ;
					
					d1node++;
				}
				else
				{
					printf("\ncheck indexes (2)\n");
					return;
				}
			}
		}
		// end update groupwise
		
		// update w
		for(int i=0; i<w_size; i++)
		{
			w[i] += all_delw[i];
			
			Dmax_new = max(Dmax_new, fabs(all_delw[i]));
			wmax_new = max(wmax_new, fabs(w[i]));
		}
		
		// begin update z
		for(int i=0; i<l; i++)
			z[i] = 0;

		for(int i=0; i<w_size; i++)
		{
			if(w[i]==0) continue;
			x = prob_col->x[i];
			while(x->index != -1)
			{
				z[x->index-1] += w[i]*x->value;
				x++;
			}
		}
		// end update z
		
		iter++;
		
		if(iter % 10 == 0) {
			if (param->chat_level == 0) // display a . every 10th when no other info displayed
				info(".");
		}

		if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,iter); save_model(filenametosave,model_);}
		if (param->chat_level > 0)
		{
			cpu_now = clock();
			cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
			v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);																										
			__QMMCHAT__
			cpu_begin += (clock() - cpu_now);
		}
		
		if(Dmax_new <= eps*wmax_new) 
			break;
			
		if(param->turn_to_nc==1)
		{
			PREC = false;
			curv_type = NC;
		}
	}


	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);


	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
	}

	delete [] index;
	delete [] y;
	delete [] z;
	delete [] hdot;
	delete [] qdot;
	delete [] curv;
	delete [] xj_sq;
	delete [] Djpp;
}


// solve mm with cg; modified from solve_mmcd_simple
static void solve_mmcg(
					   const problem *prob_row, double *w, double eps, 
					   double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin)
{
	printf("\n......solving with mmcg.......\n");
	
	problem * prob_col = (problem*) prob_row; // this is done to print chat messages, otherwise row order is always used
	
	int l = prob_col->l;  // N in paper
	int w_size = prob_col->n;  // M in paper
	int j, s, iter = 0;
	int max_iter = 1000;
	
	//double sigma = 0.01;
	double d;
	//double G_loss, G, H;
	double Gmax_old = 0;
	double Gmax_new;
	double wmax_old = INF;
	double wmax_new;
	double Dmax_old = INF;
	double Dmax_new;
	double G2_old = INF;
	double G2_new;
	double Gmax_init;
	//double d_old, d_diff;
	//double loss_old, loss_new;
	//double appxcond, cond;

	schar *y = new schar[l];
	double *z = new double[l];		// z in paper
	double *hdot = new double[l];	// hdot
	double *qdot = new double[l];	// qdot
	double *curv = new double[l];	// curv
	double *xj_sq = new double[w_size];
	double *Djpp = new double[w_size];
	double *Diff = new double[w_size];
	feature_node *x;
	bool PREC = false, DENUPDATE = true;
	loss_var loss_type = param->loss_type;
	curv_var curv_type = param->curv_type;
	double loss_param = param->loss_param;
	double curv_param = param->curv_param;
	double reg_param = param->reg_param;
	struct model* model_init = NULL;
	double v;
	double & objective_value = v;
	clock_t cpu_now;
	double cpu_time_elapsed = 0;

	double C[3] = {Cn,0,Cp};
	int nnz;

	if (param->init_model_file) {
		if((model_init=load_model(param->init_model_file))==0)
		{
			fprintf(stderr,"can't open model file %s\n",param->init_model_file);
			exit(1);
		}
	}

	// initialize w and z either from a model file or from w=0
	if (model_init) {
		if (model_init->nr_feature != w_size) { info("initial model file weight dimension does not match the data."); exit(0);}

		for(j=0; j<l; j++)
		{
			z[j] = 0;  // we  initialize z vector with z=0
			if(prob_row->y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;
		}

		for(j=0; j<w_size; j++)
		{
			w[j] = model_init->w[j]; // we initialize with model w
			xj_sq[j] = 0;
		}
		
		for(j=0; j<l; j++)
		{
			x = prob_row->x[j];
			while(x->index != -1)
			{
				int ind = j;
				double val = x->value;
				x->value *= y[ind]; // x->value now stores yi*xij, xtilde in paper
				xj_sq[x->index-1] += C[GETI(ind)]*val*val;
				x++;
			}
		}
		// if w is nonzero, need to find initial z
		for(int i=0; i<l; i++)
		{	
			x = prob_row->x[i];
			while(x->index != -1)
			{
				z[i] += w[x->index-1]*x->value;
				x++;
			}
		}
	} else {
		for(j=0; j<l; j++)
		{
			z[j] = 0;  // we  initialize with w=0
			if(prob_row->y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;
		}

		for(j=0; j<w_size; j++)
		{
			w[j] = 0; // we  initialize with w=0
			xj_sq[j] = 0;
		}
		
		for(j=0; j<l; j++)
		{
			x = prob_row->x[j];
			while(x->index != -1)
			{
				int ind = j;
				double val = x->value;
				x->value *= y[ind]; // x->value now stores yi*xij, xtilde in paper
				xj_sq[x->index-1] += C[GETI(ind)]*val*val;
				x++;
			}
		}
	}

	if (true) {
		if (curv_type == MC) 
			PREC = true;
		if (loss_type == L2) {
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
			if (curv_type == MC) for(j=0; j<l; j++) curv[j] = 2.0;
		}else if (loss_type == L1){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/loss_param;
			if (curv_type == MC) for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == HU1){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/(2.0*loss_param);
			if (curv_type == MC) for(j=0; j<l; j++) curv[j] = 1.0/(2.0*loss_param);
		}else if (loss_type == HU2){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1/loss_param;
			if (curv_type == MC) for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == LOG){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/4.0;
			if (curv_type == MC) for(j=0; j<l; j++) curv[j] = 1.0/4.0;
		}
	}

	if (loss_type == LS) {
		PREC = true;
		for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	} else if (loss_type == L2 && curv_type == OC) {
		PREC = true;
		for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	}

	// begin variables for update
	double *delw=new double[w_size];
    double *rk=new double[w_size];
	double *rkold=new double[w_size];
	double *zk=new double[w_size];
	double *zkold=new double[w_size];
	double *pk=new double[w_size];
	double *Minv=new double[w_size];
	double mean_Djpp=0;
	double *b=new double[w_size];
	double *Apk=new double[w_size];
	double *xpk=new double[l];
	mean_Djpp=0;
	for(int i=0; i<w_size; i++)
		mean_Djpp+=Djpp[i];
	mean_Djpp/=w_size;
	for(int i=0; i<w_size; i++)
		Minv[i]=1/(0.9*Djpp[i]+0.1*mean_Djpp);
	// end variables for update
	
	if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,iter); save_model(filenametosave,model_);}
	if (param->chat_level > 0)
	{
		cpu_now = clock();
		cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
		v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);																										
		__QMMCHAT__
		cpu_begin += (clock() - cpu_now);
	}

	while(iter < max_iter)
	{
		if (loss_type == L2) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -2.0*(1.0-z[j]) : 0; }
		} else if (loss_type == L1) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -1.0 : 0; }
		} else if (loss_type == HU1) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1 + t) ? 1.0/(2.0*t)*(z[j]-1.0-t) : 0.0; }
		} else if (loss_type == HU2) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1) ? 1.0/(t)*(z[j]-1) : 0; }
		} else if (loss_type == LOG) {
			for(j=0; j<l; j++) { hdot[j] = -1.0/(1+exp(z[j])); }
		} else if (loss_type == LS) {
			for(j=0; j<l; j++) { hdot[j] = -2.0*(1-z[j]); }
		}
		// initialize qdot
		for(j=0; j<l; j++) { qdot[j] = hdot[j]; }

		// compute curv's if necessary
		if (PREC==false) {
			if (loss_type == L2 && curv_type == NC) {
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (z[j] < 1) ? 2.0 : epsi; }
			} else if (loss_type == L1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == L1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == HU1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU2 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : 1 / (2*fabs(1-t/2-z[j]));}
			} else if (loss_type == HU2 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : epsi;}
			} else if (loss_type == LOG && curv_type == OC) {
				for(j=0; j<l; j++) { 
					double ez = exp(z[j]);
					double emz = exp(-z[j]);
					double eme = ez - emz;
					curv[j] = (fabs(eme)<1e-3) ? 1.0/4.0 : fabs(eme/z[j])/(2*(2+ez+emz));
				}
			} else if (loss_type == LOG && curv_type == NC) {
				for(j=0; j<l; j++) { 
					double ee = exp(z[j])+exp(-z[j]);
					curv[j] = 1.0/(2+ee);
				}
			} else {
				info("invalid loss and curv type combination\n");
				exit(0); //revisit
			}
		}

		Dmax_new = 0;
		wmax_new = 0;		
	
		// begin conjugate gradient 
		for(int i=0; i<w_size; i++)
			b[i]=-w[i];

		feature_node *dnode;
		for(int i=0; i<l; i++)
		{
			dnode = prob_row->x[i];
			while(dnode->index != -1)
			{
				b[dnode->index-1] -= C[GETI(i)] * dnode->value * hdot[i];
				dnode++;
			}
		}
			
		for(int i=0; i<w_size; i++)
		{
			delw[i]=0;
			rk[i]=(-1)*b[i];
			zk[i]=Minv[i]*rk[i];
			pk[i]=(-1)*zk[i];
		}
		
		double first_norm_rk = 0;
        for(int k=0;k<w_size-1;k++)
		{
            for(int i=0; i<w_size; i++)
			{
				rkold[i]=rk[i];
				zkold[i]=zk[i];
				Apk[i]=pk[i];
			}

			for(int i=0; i<l; i++)
				xpk[i]=0;

			for(int i=0; i<l; i++)
			{
				//for(int j=0; j<w_size; j++)
				//	row_vec[j] = 0;
				
				dnode = prob_row->x[i];
				while(dnode->index != -1)
				{
					xpk[i] += dnode->value * pk[dnode->index-1];
					dnode++;
				}
				
				/*
				for(int j=0; j<w_size; j++)
					if(row_vec[j] != 0)
						Apk[j] += row_vec[j] * C[GETI(i)] * curv[i] * xpk[i];					
				*/
			}
			for(int i=0; i<l; i++)
			{
				dnode = prob_row->x[i];
				while(dnode->index != -1)
				{
					Apk[dnode->index-1] += dnode->value * C[GETI(i)] * curv[i] * xpk[i];					
					dnode++;
				}
			}
			
			double dalphak=0, nalphak=0, alphak=0;
			for(int i=0; i<w_size; i++)
			{
				dalphak+=(rk[i]*zk[i]);
				nalphak+=(pk[i]*Apk[i]);
			}
            alphak=dalphak/nalphak;

			for(int i=0; i<w_size; i++)
				delw[i]+=alphak*pk[i];

			double norm_rk=0;
			for(int i=0; i<w_size; i++)
			{
				rk[i]+=alphak*Apk[i];
				norm_rk+=rk[i]*rk[i];
			}
			
			if(k==0)
                first_norm_rk = sqrt(norm_rk);
			else if(sqrt(norm_rk)<=first_norm_rk*param->cg_tol)
                break;

			for(int i=0; i<w_size; i++)
				zk[i]=Minv[i]*rk[i];
            
			double dbetak=0, nbetak=0, betak=0;
			for(int i=0; i<w_size; i++)
			{
				dbetak+=(zk[i]*rk[i]);
				nbetak+=(zkold[i]*rkold[i]);
			}
			betak=dbetak/nbetak;
            
			for(int i=0; i<w_size; i++)
				pk[i]=((-1)*zk[i]) + (betak*pk[i]);
		}

		for(int i=0;i<w_size;i++)
		{
			w[i]+=delw[i];

			Dmax_new = max(Dmax_new, fabs(delw[i]));
			wmax_new = max(wmax_new, fabs(w[i]));
		}
		// end conjugate gradient
		
		// begin update z
		for(int i=0; i<l; i++)
			z[i] = 0;

		for(int i=0; i<l; i++)
		{
			dnode = prob_row->x[i];
			while(dnode->index != -1)
			{
				z[i] += w[dnode->index-1]*dnode->value;
				dnode++;
			}
		}
		// end update z
		
		
		iter++;
		
		if(iter % 10 == 0) {
			if (param->chat_level == 0) // display a . every 10th when no other info displayed
				info(".");
		}

		if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,iter); save_model(filenametosave,model_);}
		if (param->chat_level > 0)
		{
			cpu_now = clock();
			cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
			v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);																										
			__QMMCHAT__
			cpu_begin += (clock() - cpu_now);
		}
		
		if(Dmax_new <= eps*wmax_new) 
			break;
				
		if(param->turn_to_nc==1)
		{
			PREC = false;
			curv_type = NC;
		}
	}


	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);


	for(int j=0; j<l; j++)
	{
		x = prob_row->x[j];
		while(x->index != -1)
		{
			x->value *= prob_row->y[j]; // restore x->value
			x++;
		}
	}

	delete [] y;
	delete [] z;
	delete [] hdot;
	delete [] qdot;
	delete [] curv;
	delete [] xj_sq;
	delete [] Djpp;
}


static void solve_mmcd(
					   problem *prob_col, double *w, double eps, 
					   double Cp, double Cn, const parameter *param, const struct model* model_, clock_t cpu_begin)
{
	int l = prob_col->l;  // N in paper
	int w_size = prob_col->n;  // M in paper
	int j, s, iter = 0;
	int max_iter = 1000;
	int active_size = w_size;

	//double sigma = 0.01;
	double d;
	//double G_loss, G, H;
	double Gmax_old = 0;
	double Gmax_new;
	double wmax_old = INF;
	double wmax_new;
	double Dmax_old = INF;
	double Dmax_new;
	double G2_old = INF;
	double G2_new;
	double Gmax_init;
	//double d_old, d_diff;
	//double loss_old, loss_new;
	//double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *z = new double[l];		// z in paper
	double *hdot = new double[l];	// hdot
	double *qdot = new double[l];	// qdot
	double *curv = new double[l];	// curv
	double *xj_sq = new double[w_size];
	double *Djpp = new double[w_size];
	double *Diff = new double[w_size];
	feature_node *x;
	bool PREC = false, DENUPDATE = true;
	loss_var loss_type = param->loss_type;
	curv_var curv_type = param->curv_type;
	double loss_param = param->loss_param;
	double curv_param = param->curv_param;
	double reg_param = param->reg_param;
	struct model* model_init = NULL;
	double v;
	double & objective_value = v;
	clock_t cpu_now;
	double cpu_time_elapsed = 0;

	double C[3] = {Cn,0,Cp};
	int nnz;
	int Maxruns = 1;

	if (param->init_model_file) {
		if((model_init=load_model(param->init_model_file))==0)
		{
			fprintf(stderr,"can't open model file %s\n",param->init_model_file);
			exit(1);
		}
	}

	// initialize w and z either from a model file or from w=0
	if (model_init) {
		if (model_init->nr_feature != w_size) { info("initial model file weight dimension does not match the data."); exit(0);}

		for(j=0; j<l; j++)
		{
			z[j] = 0;  // we  initialize z vector with z=0
			if(prob_col->y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;
		}

		for(j=0; j<w_size; j++)
		{
			w[j] = model_init->w[j]; // we initialize with model w
			index[j] = j;
			xj_sq[j] = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				double val = x->value;
				x->value *= y[ind]; // x->value now stores yi*xij, xtilde in paper
				xj_sq[j] += C[GETI(ind)]*val*val;
				x++;
			}
		}
		// if w is nonzero, need to find initial z
		for(int i=0; i<w_size; i++)
		{	
			if(w[i]==0) continue;
			x = prob_col->x[i];
			while(x->index != -1)
			{
				z[x->index-1] += w[i]*x->value;
				x++;
			}
		}
	} else {
		for(j=0; j<l; j++)
		{
			z[j] = 0;  // we  initialize with w=0
			if(prob_col->y[j] > 0)
				y[j] = 1;
			else
				y[j] = -1;
		}

		for(j=0; j<w_size; j++)
		{
			w[j] = 0; // we  initialize with w=0
			index[j] = j;
			xj_sq[j] = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				double val = x->value;
				x->value *= y[ind]; // x->value now stores yi*xij, xtilde in paper
				xj_sq[j] += C[GETI(ind)]*val*val;
				x++;
			}
		}
	}

	if (curv_type == MC) {
		PREC = true;
		if (loss_type == L2) {
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
			for(j=0; j<l; j++) curv[j] = 2.0;
		}else if (loss_type == L1){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/loss_param;
			for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == HU1){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/(2.0*loss_param);
			for(j=0; j<l; j++) curv[j] = 1.0/(2.0*loss_param);
		}else if (loss_type == HU2){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1/loss_param;
			for(j=0; j<l; j++) curv[j] = 1.0/loss_param;
		}else if (loss_type == LOG){
			for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*1.0/4.0;
			for(j=0; j<l; j++) curv[j] = 1.0/4.0;
		}
	}

	if (loss_type == LS) {
		PREC = true;
		for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	} else if (loss_type == L2 && curv_type == OC) {
		PREC = true;
		for(j=0; j<w_size; j++) Djpp[j] = xj_sq[j]*2.0;
		for(j=0; j<l; j++) curv[j] = 2.0;
	}

	if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,iter); save_model(filenametosave,model_);}
	if (param->chat_level > 0)
	{
		cpu_now = clock();
		cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
		v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);																										
		__QMMCHAT__
		cpu_begin += (clock() - cpu_now);
	}

	while(iter < max_iter)
	{
		//printf("Iter: %d\n",iter);
		Gmax_new = 0;
		wmax_new = 0;
		Dmax_new = 0;
		G2_new = 0;

		if (loss_type == L2) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -2.0*(1.0-z[j]) : 0; }
		} else if (loss_type == L1) {
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1) ? -1.0 : 0; }
		} else if (loss_type == HU1) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1 + t) ? 1.0/(2.0*t)*(z[j]-1.0-t) : 0.0; }
		} else if (loss_type == HU2) {
			double t = loss_param; 
			for(j=0; j<l; j++) { hdot[j] = (z[j] < 1-t) ? -1.0 : (z[j] < 1) ? 1.0/(t)*(z[j]-1) : 0; }
		} else if (loss_type == LOG) {
			for(j=0; j<l; j++) { hdot[j] = -1.0/(1+exp(z[j])); }
		} else if (loss_type == LS) {
			for(j=0; j<l; j++) { hdot[j] = -2.0*(1-z[j]); }
		}
		// initialize qdot
		for(j=0; j<l; j++) { qdot[j] = hdot[j]; }

		// compute curv's if necessary
		if (PREC==false) {
			if (loss_type == L2 && curv_type == NC) {
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (z[j] < 1) ? 2.0 : epsi; }
			} else if (loss_type == L1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == L1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU1 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : 1 / (2*fabs(1-z[j])); }
			} else if (loss_type == HU1 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(z[j] - 1) < t) ? 1/(2*t) : epsi; }
			} else if (loss_type == HU2 && curv_type == OC) {
				double t = loss_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : 1 / (2*fabs(1-t/2-z[j]));}
			} else if (loss_type == HU2 && curv_type == NC) {
				double t = loss_param;
				double epsi = param->curv_param;
				for(j=0; j<l; j++) { curv[j] = (fabs(1 - t/2 - z[j]) < t/2) ? 1/t : epsi;}
			} else if (loss_type == LOG && curv_type == OC) {
				for(j=0; j<l; j++) { 
					double ez = exp(z[j]);
					double emz = exp(-z[j]);
					double eme = ez - emz;
					curv[j] = (fabs(eme)<1e-3) ? 1.0/4.0 : fabs(eme/z[j])/(2*(2+ez+emz));
				}
			} else if (loss_type == LOG && curv_type == NC) {
				for(j=0; j<l; j++) { 
					double ee = exp(z[j])+exp(-z[j]);
					curv[j] = max((1.0/(2+ee)),param->curv_param);					
				}
			} else {
				info("invalid loss and curv type combination\n");
				exit(0); //revisit
			}
		}

		// randomly choose order of wj's to update
		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(int run=0; run < Maxruns; run++) {
			for(s=0; s<active_size; s++)
			{
				j = index[s];

				double Djp = 0;
				double Djppnow = 0;

				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index-1;
					double val = x->value;
					//double ctv = C[GETI(ind)]*val;
					if(qdot[ind] != 0)
					{
						Djp += C[GETI(ind)]*val*qdot[ind];
					}
					if (PREC == false && curv[ind] != 0)
						Djppnow += C[GETI(ind)]*val*val*curv[ind];
					x++;
				}

				if (PREC == true) {
					Djppnow = Djpp[j];
				}

				if (reg_param > 0) {
					Djp += reg_param * w[j];
					Djppnow += reg_param;
				}
				// newton step
				d = -Djp/Djppnow;

				if(fabs(d) < 1.0e-12)
					continue;

				double gradwj=0; // gradient entry for wj at w[j]

				if (w[j] == 0) {
					gradwj = max(0.0, Djp - (1-reg_param));
					gradwj = max(gradwj, -(1-reg_param)-Djp);
				} else {
					gradwj = Djp + (1-reg_param)*mysign(w[j]);
				}

				double abs_gradwj = fabs(gradwj);

				if (false && reg_param < 1) { //never do
					if(w[j] == 0)
					{
						if( abs_gradwj < Gmax_old / l)
						{
							active_size--;
							swap(index[s], index[active_size]);
							s--;
							Diff[j]=0; // to prevent from activating later, except for whole activation
							continue;
						}
					}
				}

				Gmax_new = max(Gmax_new, abs_gradwj);
				// compute gradient information if its going to be displayed
				// note that this is not right since active size will change now
				//if (param->chat_level > 1) {
				//	G2_new   += gradwj*gradwj;
				//}

				double wjhat = w[j] + d;

				if (reg_param < 1) {
					wjhat = shrink(wjhat, (1-reg_param)/(Djppnow));
				}

				double difwj  = wjhat-w[j]; // difference
				double adifwj = fabs(difwj);

				// if it has not changed much for previous two updates
				if( iter >= 1 && adifwj < Dmax_old * param->cd_tol && Diff[j] < Dmax_old * param->cd_tol)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				
				w[j]=wjhat;
				Diff[j] = adifwj;


				Dmax_new = max(Dmax_new, adifwj);
				wmax_new = max(wmax_new, fabs(wjhat));

				// update qdot
				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index-1;
					double val = x->value;
					qdot[ind] += val*curv[ind]*difwj;
					x++;
				}
			}
		}

		// // update z[i]
		if (curv_param > 1e-12) {
			for(j=0;j<l;j++)
				z[j] += (qdot[j]-hdot[j])/curv[j];
		} else { // if there is zero curvature value (possible for some loss if param->curv_param=0)
			info("#");
			for(int i=0; i<l; i++)
				z[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				while(x->index != -1)
				{
					z[x->index-1] += w[i]*x->value;
					x++;
				}
			}
		}


		if(iter == 0) {
			Gmax_init = Gmax_new;
		}
		iter++;
		
		//// try not updating denominator, does not work
		//if (iter > 3) {
		//	DENUPDATE = false;
		//	if (iter % 10 == 0)
		//		DENUPDATE = true;
		//}

		if(iter % 10 == 0) {
			if (param->chat_level == 0) // display a . every 10th when no other info displayed
				info(".");
		}

		if (iter % param->cd_reset == 0) {
			active_size = w_size; // make it to reconsider updating all wj's including 0's every 20th
		}

		// reconsider deactived list
		if (iter < 0) { // never do
			for (int i=active_size;i<w_size;i++) 
				if( Diff[index[i]] > Dmax_new / 30)
				{
					swap(index[i], index[active_size]);
					active_size++;
				}
		}

		if(param->save_each_iter!=0) {char filenametosave[1024]; sprintf(filenametosave,"%s%d.model",param->save_each_iter,iter); save_model(filenametosave,model_);}
		if (param->chat_level > 0)
		{
			cpu_now = clock();
			cpu_time_elapsed = ((double) (cpu_now - cpu_begin)) / ((double) CLOCKS_PER_SEC);																					
			v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);																										
			__QMMCHAT__
			cpu_begin += (clock() - cpu_now);
		}

		//if(Gmax_new <= eps*Gmax_init) break;
		if(Dmax_new <= eps*wmax_new) {
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = 0;
				//continue;
			}
		} else {

			// these currently unused
			Dmax_old = Dmax_new;
			Gmax_old = Gmax_new;
			G2_old   = G2_new;
			wmax_old = wmax_new;
		}
		
		if(param->turn_to_nc==1)
		{
			PREC = false;
			curv_type = NC;
		}
	}


	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	v = mmcd_obj(prob_col,param,w,z,y,C,&nnz);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);


	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
	}

	delete [] index;
	delete [] y;
	delete [] z;
	delete [] hdot;
	delete [] qdot;
	delete [] curv;
	delete [] xj_sq;
	delete [] Djpp;
}
