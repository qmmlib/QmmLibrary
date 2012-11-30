#ifndef _TRON_H
#define _TRON_H

/**
Interface for function objects that are called by TRON. 
Classes that implement this are used by TRON during training
*/
class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
};

/**
Class for L2 regularized SVC training using TRON.

Initialized with an object that implements function interface and epsilon parameter.
*/
class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
	~TRON();

	/**
	Trains 
	*/
	void tron(double *w, const parameter *param, const struct model* model_, clock_t cpu_begin);
	void set_print_string(void (*i_print) (const char *buf));

private:
	/**
	Conjugate gradient within TRON
	*/
	int trcg(double delta, double *g, double *s, double *r);
	double norm_inf(int n, double *x);

	double eps;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif
