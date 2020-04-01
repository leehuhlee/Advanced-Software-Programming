#include "my_solver.h"
#include <math.h>

#define SOLNUMS 3
#define MATCOLS SOLNUMS
#define MATROWS SOLNUMS
#define TOLERANCE 0.0000001

void fcn3_4( int *n, double *x, double *fvec, int *iflag )
{	
	fvec[0] = 10.0*x[0] - 2.0*x[1]*x[1] + x[1] -2.0*x[2] - 5.0;
	fvec[1] = 8.0*x[1]*x[1] + 4.0*x[2]*x[2] - 9.0;
	fvec[2] = 8.0*x[1]*x[2] + 4.0;	
}

void program3_4(void)
{
	//TODO

	
}