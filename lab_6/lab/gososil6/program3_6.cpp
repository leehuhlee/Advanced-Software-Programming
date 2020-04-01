#include "my_solver.h"
#include <math.h>

#define SOLNUMS 2
#define MATCOLS SOLNUMS
#define MATROWS SOLNUMS
#define TOLERANCE 0.0000001

void fcn3_6( int *n, double *x, double *fvec, int *iflag )
{	
	fvec[0] = 2.0*x[0]*x[0]*x[0] - 12.0*x[0] - x[1] - 1.0;
	fvec[1] = 3.0*x[1]*x[1] - 6.0*x[1] - x[0] - 3.0;		
}

void program3_6(void)
{
	//TODO
	
}