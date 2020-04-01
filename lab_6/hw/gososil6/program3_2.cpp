#include "my_solver.h"
#define SOLNUMS 4
#define TOLERANCE 0.0000001

void rf(int *n, double *x, double *fvec, double *fjac, int *ldfjac, int *iflag) {
	if (*iflag == 1) {
		fvec[0] = x[0] + 10 * x[1] + 9;
		fvec[1] = sqrt(5) * (x[2] - x[3]) - 2*sqrt(5);
		fvec[2] = pow(x[1]-(2*x[2]), 2.0) - 9;
		fvec[3] = sqrt(10)*pow(x[0]-x[3], 2.0) - 2*sqrt(10);
	}
	else if (*iflag == 2) {
		fjac[0] = 1.0;									 fjac[4] = 10;					fjac[8] = 0;				fjac[12] = 0;
		fjac[1] = 0;									 fjac[5] = 0;					fjac[9] = sqrt(5);			fjac[13] = -sqrt(5);
		fjac[2] = 0;									 fjac[6] = 2 * x[1] - 4 * x[2];	fjac[10] = -4 * x[1] + 4;	fjac[14] = 0;
		fjac[3] = 2 * sqrt(10)*x[0] - 2 * sqrt(10)*x[3]; fjac[7] = 0;					fjac[11] = 0;				fjac[15] = -2 * sqrt(10)*x[0] + 2 * sqrt(10)*x[3];
	}
}

void program3_2(void)
{
	FILE *fp_w = fopen("roots_found_3-2.txt", "w");
	int n = SOLNUMS;
	double x[SOLNUMS] = { 0.9, -0.9, 1.25, -1.25 };
	double fvec[SOLNUMS];
	double fjac[SOLNUMS*SOLNUMS];
	double wa[34]; // (n*(n+13))/2
	double tol = TOLERANCE;
	int ldfjac = SOLNUMS;
	int info;
	int lwa = 34;
	hybrj1_(rf, &n, x, fvec, fjac, &ldfjac, &tol, &info, wa, &lwa);
	fprintf(fp_w, "x* = (");
	for (int i = 0; i < n; i++)
	{
		if (i + 1 < n) {
			fprintf(fp_w, "%.15lf, ", x[i]);
		}
		else {
			fprintf(fp_w, "%.15lf", x[i]);
		}
	}
	fprintf(fp_w, ")\n");
	for (int i = 0; i < n; i++)
	{
		fprintf(fp_w, "f(x*) = %.15lf\n", fvec[i]);
	}
	if (fp_w != NULL) fclose(fp_w);
}