#include "my_solver.h"

extern double(*_F)(double);
extern double(*_FP)(double);

/*********************************************
 Bisection Method -- HOMEWORK
**********************************************/

void program1_4(FILE *fp)
{
	double a0, b0, x0, x1, temp;
	int n;

	fprintf(fp, " n              xn1                  |f(xn1)|\n");
	scanf("%lf %lf", &a0, &b0);
	x1 = (a0 + b0) / 2.0;

	for (n = 0; ; n++) {
		fprintf(fp, "%2d  %20.18e  %12.10e\n", n, x1, fabs(_F(x1)));

		x0 = x1;
		x1 = (a0 + b0) / 2.0;
		if (fabs(_F(x1)) < DELTA || abs(b0 - a0)/2 < EPSILON || n >= Nmax ) break;

		if( _F(x1)*_F(a0) <0) b0 = x1;
		else if( _F(x1)*_F(b0)<0) a0 = x1;


	}
	printf("%2d  %20.18e  %12.10e\n", n, x1, fabs(_F(x1)));
}