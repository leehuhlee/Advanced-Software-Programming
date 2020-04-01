#include "my_solver.h"
#include <math.h>

#define SOLNUMS 2
#define MATCOLS SOLNUMS
#define MATROWS SOLNUMS
#define TOLERANCE 0.0000001
#define M_PI 3.14159265359

void fcn3_3(int *m, int *n, double *x, double *fvec, int *iflag)
{
	fvec[0] = (sin(x[0] * x[1] + (M_PI / 6)) + sqrt(x[0] * x[0] * x[1] * x[1] + 1)) / cos(x[0] - x[1]) + 2.8;
	fvec[1] = (x[0] * exp(x[0] * x[1] + M_PI / 6) - sin(x[0] - x[1])) / sqrt(x[0] * x[0] * x[1] * x[1] + 1) - 1.66;
}

void program3_3(void)
{
	int n = SOLNUMS;
	int m = SOLNUMS;
	double x[SOLNUMS] = { 20.0, 0.0 };	//need to initilize x0
	double fvec[SOLNUMS], fvec1[SOLNUMS], fjac[(MATCOLS + 1) * (MATROWS + 1)];
	int ldfjac = SOLNUMS;
	double tol = TOLERANCE;
	int info;
	double wa[(SOLNUMS * (3 * SOLNUMS + 13)) / 2];
	int lwa = m * n + 5 * n + 2 * m;
	int iwa[SOLNUMS];
	double wa1[(SOLNUMS * (3 * SOLNUMS + 13)) / 2];
	int lwa1 = (SOLNUMS * (3 * SOLNUMS + 13)) / 2;
	int i;

	FILE *fp_w = fopen("roots_found_3-3.txt", "w");

	lmdif1_(fcn3_3, &m, &n, x, fvec, &tol, &info, iwa, wa, &lwa);

	fprintf(fp_w, "x = %.15lf \ny = %.15lf \n", x[0], x[1]);
	for (i = 0; i < 2; i++)
		fprintf(fp_w, "f%d(x, y) = %.15lf\n", i, fvec[i]);
}