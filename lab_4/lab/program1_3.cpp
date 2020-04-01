#include "my_solver.h"

extern double(*_f)(double, double, double);
extern double(*_fp)(double, double, double);

/*********************************************
 Newton-Raphson Method - Three-leaved clover
**********************************************/

void program1_3(FILE *fp)
{
	double x0, x1, a, b;
	int n;

	fprintf(fp, " n              xn1                  |f(xn1)|\n");
	scanf("%lf %lf %lf", &x1, &a, &b);
	x0=DBL_MAX;
	for (n = 0;; n++){
		fprintf(fp, "%2d  %20.18e  %12.10e\n", n, x1, fabs(_f(x1, a, b)));
		if( fabs(_f(x1,a,b))<DELTA || n>=Nmax||fabs(x0-x1)<EPSILON)
		//TODO
		break;
		x0=x1;
		x1 = x0 - (_f(x0,a,b)/_fp(x0,a,b));
	}
	printf("%2d  %20.18e  %12.10e\n", n, x1, fabs(_f(x1, a, b)));
}