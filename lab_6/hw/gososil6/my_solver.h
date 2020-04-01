#ifndef _MY_SOLVER
#define _MY_SOLVER

#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

const double EPSILON = 0.0000001;

extern "C"
{
	void gespp_(int *, float(*)[3], int *, int *, float *);
	void solve_(int *, float(*)[3], int *, int *, float *, float *);
	int rpoly_(double *, int *, double *, double *, long int *);
	int hybrj1_(void fcn(int *, double *, double *, double *, int *, int *), int *, double *, double *, double *, int *, double *, int *, double *, int *);
	int hybrd1_(void fcn(int *, double *, double *, int *), int *, double *, double *, double *, int *, double *, int *);
	void lmdif1_(void(*fcn)(int *m, int *n, double *x, double *fvec, int *iflag),
		int *m, int *n, double *x, double *fvec,
		double *tol, int *info, int *iwa, double *wa, int *lwa);
};

#endif