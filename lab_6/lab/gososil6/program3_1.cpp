#include "my_solver.h"
double F_(double* poly, int degree, double root)
{
	int i,j;
	double result=0,sum;
	for(i=0;i<=degree;i++)
	{
		sum=1;
		for(j=0;j<degree-i;j++)
			sum*=root;
		result+=sum*poly[i];
	}

	return result;
}

void rpoly(const char *readfilename, const char *writefilename)
{
	int NCOEF, DEGREE;

	double *poly;
	double *zeror, *zeroi;
	long int fail;
	int i,j;	

	FILE *fp_r = fopen(readfilename, "r");
	if(fp_r == NULL) 
	{
		printf("%s file open error...\n", readfilename);
		return;
	}

	FILE *fp_w = fopen(writefilename, "w");
	if(fp_w == NULL) 
	{
		printf("%s file open error...\n", writefilename);
		return;
	}	
	// TODO

	fscanf(fp_r, "%d",&DEGREE);
	NCOEF = DEGREE +1;

	poly = (double *)malloc(sizeof(double) * (NCOEF));
	zeroi = (double *)malloc(sizeof(double) * (DEGREE));
	zeror = (double *)malloc(sizeof(double) * (DEGREE));

	for(i=0; i<NCOEF; i++){
		fscanf(fp_r, "%lf", &poly[i]);
	}

	rpoly_(poly, &DEGREE, zeror, zeroi, &fail);

	if(fail){
		fprintf(fp_w, "fail\n");
	}
	else{
		fprintf(fp_w, "zeror :\n");
		for(i=0; i<DEGREE; i++){
			fprintf(fp_w, "%.15f ", zeror[i]);
		}
		fprintf(fp_w,"\n");
		fprintf(fp_w, "zeroi : \n");
		for(i=0; i<DEGREE; i++){
			fprintf(fp_w, "%.15f ", zeroi[i]);
		}
		fprintf(fp_w, "\n\n");

		for(i=0; i<DEGREE; i++){
			if(zeroi[i]==0){
				fprintf(fp_w, "f(%.15f) = %.15f\n", zeror[i], F_(poly, DEGREE, zeror[i]));
			}
		}
	}
	
	free(zeroi);
	free(zeror);
	free(poly);

	if(fp_r != NULL) fclose(fp_r);
	if(fp_w != NULL) fclose(fp_w);
}

void program3_1()
{
	rpoly("polynomial_3-1_1.txt", "roots_1.txt");
	rpoly("polynomial_3-1_2.txt", "roots_2.txt");
	rpoly("polynomial_3-1_3.txt", "roots_3.txt");
	rpoly("polynomial_3-1_4.txt", "roots_4.txt");
	rpoly("polynomial_3-1_5.txt", "roots_5.txt");
	rpoly("polynomial_3-1_6.txt", "roots_6.txt");
}