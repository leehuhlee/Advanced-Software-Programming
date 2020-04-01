#include "my_solver.h"
#include <math.h>

#define SOLNUMS 2
#define MATCOLS SOLNUMS
#define MATROWS SOLNUMS
#define TOLERANCE 0.0000001

void fcn3_5( int *n, double *x, double *fvec, double *fjac, int *ldfjac, int *iflag )
{
	// origin function F(x)
	if( *iflag == 1 ){
		//TODO	
		fvec[0] = 3.0*x[0]*x[0] - 2.0*x[1]*x[1] - 1.0;
		fvec[1] = x[0]*x[0]-2.0*x[0]+x[1]*x[1]+2.0*x[1]-8.0;

	}
	// Jacobi matrix J(x)
	else if( *iflag == 2 ){
		//TODO
		fjac[0] = 6.0*x[0];					fjac[2] = -4.0*x[1];
		fjac[1] = 2.0*x[0]-2.0;				fjac[3] = 2.0*x[1]+2.0;

	}
}

void program3_5(void)
{
	//TODO

}