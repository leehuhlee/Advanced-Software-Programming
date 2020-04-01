#include "my_solver.h"
#define _USE_MATH_DEFINES
#include <math.h>
#define SOLNUMS 3
#define MATCOLS SOLNUMS
#define MATROWS SOLNUMS
#define TOLERANCE 0.0000001

void fcn3_8(int *n, double *x, double *fvec, double *fjac, int *ldfjac, int *iflag)
{
	// origin function F(x)
	if (*iflag == 1) {
		fvec[0] = 3 * x[0] - cos(x[1] * x[2]) - 0.5;
		fvec[1] = x[0] * x[0] - 81.0*(x[1] + 0.1)*(x[1] + 0.1) + sin(x[2]) + 1.06;
		fvec[2] = exp(-x[0] * x[1]) + 20.0*x[2] + (10.0*M_PI - 3.0) / 3.0;
	}
	// Jacobi matrix J(x)
	else if (*iflag == 2) {
		fjac[0] = 3.0;						fjac[3] = x[2] * sin(x[1] * x[2]);		fjac[6] = x[1] * sin(x[1] * x[2]);
		fjac[1] = 2.0*x[0];					fjac[4] = -162.0*(x[1] + 0.1);		fjac[7] = cos(x[2]);
		fjac[2] = -x[1] * exp(-x[0] * x[1]);	fjac[5] = -x[0] * exp(-x[0] * x[1]);	fjac[8] = 20.0;
	}
}

void program3_8(void)
{
	int n = SOLNUMS;
	double x[SOLNUMS] = {0.1, 0.1, -0.1};
	double fvec[SOLNUMS];
	double fjac[10];
	double wa[25]; // (n*(n+13))/2
	double tol = TOLERANCE;
	int ldfjac = SOLNUMS;
	int info;
	int lwa = 25;
	FILE *fp_w = fopen("roots_3-8.txt", "w");

	if(fp_w == NULL) 
	{
		printf("%s file open error...\n", "roots_3-8.txt");
		return;
	}

	printf("%s\n", "roots_3-8.txt");
	hybrj1_(fcn3_8, &n, x, fvec, fjac, &ldfjac, &tol, &info, wa, &lwa);
	fprintf(fp_w, "x* = (");

	for(int i=0; i<n; i++)
	{
		if(i+1<n){
			fprintf(fp_w, "%.15lf, ", x[i]);
		}
		else {
			fprintf(fp_w, "%.15lf", x[i]);
		}
	}

	fprintf(fp_w, ")\n");

	for(int i=0; i<n; i++)
	{
		fprintf(fp_w, "f(x*) = %.15lf\n", fvec[i]);
	}

	if(fp_w != NULL) fclose(fp_w);
}