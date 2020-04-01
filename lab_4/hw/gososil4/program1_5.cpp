#include "my_solver.h"

extern double(*_f)(double, double, double);
extern double(*_fp)(double, double, double);

/*********************************************
 Secant Method - Three-leaved clover  -- HOMEWORK
**********************************************/

void program1_5(FILE *fp)
{
	double x0, x1, a, b, temp;
	int n;

	fprintf(fp, "n              xn1                  |f(xn1)|\n");
	scanf("%lf %lf %lf %lf", &x0, &x1, &a, &b);

	for (n = 0;; n++) {
		fprintf(fp, "%2d  %20.18e  %12.10e\n", n, x1, fabs(_f(x1, a, b)));

		if (fabs(_f(x1,a,b)) < DELTA || n >= Nmax || abs(x1-x0) < EPSILON) break;

		temp = x1;
		x1 = x1 - _f(x1,a,b)*(x1-x0)/(_f(x1,a,b)-_f(x0,a,b));
		x0 = temp;
	}
	printf("%2d  %20.18e  %12.10e\n", n, x1, fabs(_f(x1, a, b)));
}