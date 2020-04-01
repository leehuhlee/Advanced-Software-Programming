#include "my_solver.h"
#include <math.h>

#define SOLNUMS 3
#define MATCOLS SOLNUMS
#define MATROWS SOLNUMS
#define TOLERANCE 0.0000001

void fcn3_3( int *n, double *x, double *fvec, double *fjac, int *ldfjac, int *iflag )
{
	// origin function F(x)
	if( *iflag == 1 ){
		fvec[0] = x[0] + x[1] + x[2] - 3;
		fvec[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] - 5;
		fvec[2] = exp(x[0]) + x[0]*x[1] - x[0]*x[2] - 1;
	}
	// Jacobi matrix J(x)
	else if( *iflag == 2 ){
		fjac[0] = 1.0;						fjac[3] = 1.0;		fjac[6] = 1.0;
		fjac[1] = 2.0*x[0];					fjac[4] = 2.0*x[1];	fjac[7] = 2.0*x[2];
		fjac[2] = exp(x[0]) + x[1] - x[2];	fjac[5] = x[0];		fjac[8] = -x[0];
	}
}

void program3_3(void)
{
	//TODO
	
}