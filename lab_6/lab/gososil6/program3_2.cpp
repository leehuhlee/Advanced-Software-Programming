#include "my_solver.h"
#include <math.h>

#define SOLNUMS 3
#define MATCOLS SOLNUMS
#define MATROWS SOLNUMS
#define TOLERANCE 0.0000001

void fcn3_2( int *n, double *x, double *fvec, double *fjac, int *ldfjac, int *iflag )
{
	// origin function F(x)
	if( *iflag == 1 ){
		//TODO
		fvec[0] = exp(2*x[0]) - x[1] + 4;
		fvec[1] = x[1] - x[2]*x[2] - 1;
		fvec[2] = x[2] - sin(x[0]);
	}
	// Jacobi matrix J(x)
	else if( *iflag == 2 ){
		//TODO
		fjac[0] = 2*exp(2*x[0]);	fjac[3] = -1.0;		fjac[6] = 0.0;
		fjac[1] = 0.0;				fjac[4] = 1.0;		fjac[7] = -2*x[2];
		fjac[2] = -cos(x[0]);		fjac[5] = 0.0;		fjac[8] = 1.0;
	}
}

void program3_2(void)
{
	//TODO
	
}