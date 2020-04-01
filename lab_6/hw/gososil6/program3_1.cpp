#include "my_solver.h"
#define SOLNUMS 4
#define TOLERANCE 0.0000001

double C, b;
double p[SOLNUMS][3];
double t[SOLNUMS];
double tr[SOLNUMS];


void fcn3_1(int *n, double *x, double *fvec, double *fjac, int *ldfjac, int *iflag)
{
	if (*iflag == 1) {
		fvec[0] = pow(x[0] - p[0][0], 2.0) + pow(x[1] - p[0][1], 2.0) + pow(x[2] - p[0][2], 2.0) - pow(C*(tr[0] + x[3] - t[0]), 2.0);
		fvec[1] = pow(x[0] - p[1][0], 2.0) + pow(x[1] - p[1][1], 2.0) + pow(x[2] - p[1][2], 2.0) - pow(C*(tr[1] + x[3] - t[1]), 2.0);
		fvec[2] = pow(x[0] - p[2][0], 2.0) + pow(x[1] - p[2][1], 2.0) + pow(x[2] - p[2][2], 2.0) - pow(C*(tr[2] + x[3] - t[2]), 2.0);
		fvec[3] = pow(x[0] - p[3][0], 2.0) + pow(x[1] - p[3][1], 2.0) + pow(x[2] - p[3][2], 2.0) - pow(C*(tr[3] + x[3] - t[3]), 2.0);
	}
	else if (*iflag == 2) {
		fjac[0] = 2 * (x[0] - p[0][0]);	fjac[4] = 2 * (x[1] - p[0][1]);	fjac[8] = 2 * (x[2] - p[0][2]);	fjac[12] = C * -2 * C*(tr[0] + x[3] - t[0]);
		fjac[1] = 2 * (x[0] - p[1][0]);	fjac[5] = 2 * (x[1] - p[1][1]);	fjac[9] = 2 * (x[2] - p[1][2]);	fjac[13] = C * -2 * C*(tr[1] + x[3] - t[1]);
		fjac[2] = 2 * (x[0] - p[2][0]);	fjac[6] = 2 * (x[1] - p[2][1]);	fjac[10] = 2 * (x[2] - p[2][2]);	fjac[14] = C * -2 * C*(tr[2] + x[3] - t[2]);
		fjac[3] = 2 * (x[0] - p[3][0]);	fjac[7] = 2 * (x[1] - p[3][1]);	fjac[11] = 2 * (x[2] - p[3][2]);	fjac[15] = C * -2 * C*(tr[3] + x[3] - t[3]);
	}
}

void fcn3_2(int *n, double *x, double *fvec, int *iflag)
{
	if (*iflag == 1) {
		fvec[0] = pow(x[0] - p[0][0], 2.0) + pow(x[1] - p[0][1], 2.0) + pow(x[2] - p[0][2], 2.0) - pow(C*(tr[0] + x[3] - t[0]), 2.0);
		fvec[1] = pow(x[0] - p[1][0], 2.0) + pow(x[1] - p[1][1], 2.0) + pow(x[2] - p[1][2], 2.0) - pow(C*(tr[1] + x[3] - t[1]), 2.0);
		fvec[2] = pow(x[0] - p[2][0], 2.0) + pow(x[1] - p[2][1], 2.0) + pow(x[2] - p[2][2], 2.0) - pow(C*(tr[2] + x[3] - t[2]), 2.0);
		fvec[3] = pow(x[0] - p[3][0], 2.0) + pow(x[1] - p[3][1], 2.0) + pow(x[2] - p[3][2], 2.0) - pow(C*(tr[3] + x[3] - t[3]), 2.0);
	}
}

void program3_1(void)
{
	FILE *fp_r0 = fopen("GPS_signal_0.txt", "r");
	FILE *fp_r1 = fopen("GPS_signal_1.txt", "r");
	FILE *fp_r2 = fopen("GPS_signal_2.txt", "r");

	FILE *fp_w1_0 = fopen("GPS position 3-1_0.txt", "w");
	FILE *fp_w1_1 = fopen("GPS position 3-1_1.txt", "w");
	FILE *fp_w1_2 = fopen("GPS position 3-1_2.txt", "w");
	FILE *fp_w2_0 = fopen("GPS position 3-2_0.txt", "w");
	FILE *fp_w2_1 = fopen("GPS position 3-2_1.txt", "w");
	FILE *fp_w2_2 = fopen("GPS position 3-2_2.txt", "w");

	int flag = 0;
	int i;

	int n = SOLNUMS;
	double x[SOLNUMS] = { 10, 20, 30, 40 };
	double fvec[SOLNUMS];
	double fjac[SOLNUMS*SOLNUMS];
	double wa[34]; // (n*(n+13))/2
	double tol = TOLERANCE;
	int ldfjac = SOLNUMS;
	int info;
	int lwa = 34;

	for (i = 0; i < 3; i++) {

		if (flag == 0) {
			fscanf(fp_r0, "%lf %lf", &C, &b);
			fscanf(fp_r0, "%lf %lf %lf %lf %lf", &p[0][0], &p[0][1], &p[0][2], &t[0], &tr[0]);
			fscanf(fp_r0, "%lf %lf %lf %lf %lf", &p[1][0], &p[1][1], &p[1][2], &t[1], &tr[1]);
			fscanf(fp_r0, "%lf %lf %lf %lf %lf", &p[2][0], &p[2][1], &p[2][2], &t[2], &tr[2]);
			fscanf(fp_r0, "%lf %lf %lf %lf %lf", &p[3][0], &p[3][1], &p[3][2], &t[3], &tr[3]);

			hybrj1_(fcn3_1, &n, x, fvec, fjac, &ldfjac, &tol, &info, wa, &lwa);
			fprintf(fp_w1_0, "x* = (");
			for (int i = 0; i < n; i++)
			{
				if (i + 1 < n) {
					fprintf(fp_w1_0, "%.15lf, ", x[i]);
				}
				else {
					fprintf(fp_w1_0, "%.15lf", x[i]);
				}
			}
			fprintf(fp_w1_0, ")\n");
			for (int i = 0; i < n; i++)
			{
				fprintf(fp_w1_0, "f(x*) = %.15lf\n", fvec[i]);
			}

			if (fp_w1_0 != NULL) fclose(fp_w1_0);

			hybrd1_(fcn3_2, &n, x, fvec, &tol, &info, wa, &lwa);
			fprintf(fp_w2_0, "x* = (");
			for (int i = 0; i < n; i++)
			{
				if (i + 1 < n) {
					fprintf(fp_w2_0, "%.15lf, ", x[i]);
				}
				else {
					fprintf(fp_w2_0, "%.15lf", x[i]);
				}
			}
			fprintf(fp_w2_0, ")\n");
			for (int i = 0; i < n; i++)
			{
				fprintf(fp_w2_0, "f(x*) = %.15lf\n", fvec[i]);
			}

			if (fp_w2_0 != NULL) fclose(fp_w1_0);
		}

		else if (flag == 1) {
			fscanf(fp_r1, "%lf %lf", &C, &b);
			fscanf(fp_r1, "%lf %lf %lf %lf %lf", &p[0][0], &p[0][1], &p[0][2], &t[0], &tr[0]);
			fscanf(fp_r1, "%lf %lf %lf %lf %lf", &p[1][0], &p[1][1], &p[1][2], &t[1], &tr[1]);
			fscanf(fp_r1, "%lf %lf %lf %lf %lf", &p[2][0], &p[2][1], &p[2][2], &t[2], &tr[2]);
			fscanf(fp_r1, "%lf %lf %lf %lf %lf", &p[3][0], &p[3][1], &p[3][2], &t[3], &tr[3]);

			hybrj1_(fcn3_1, &n, x, fvec, fjac, &ldfjac, &tol, &info, wa, &lwa);
			fprintf(fp_w1_1, "x* = (");
			for (int i = 0; i < n; i++)
			{
				if (i + 1 < n) {
					fprintf(fp_w1_1, "%.15lf, ", x[i]);
				}
				else {
					fprintf(fp_w1_1, "%.15lf", x[i]);
				}
			}
			fprintf(fp_w1_1, ")\n");
			for (int i = 0; i < n; i++)
			{
				fprintf(fp_w1_1, "f(x*) = %.15lf\n", fvec[i]);
			}

			if (fp_w1_1 != NULL) fclose(fp_w1_1);

			hybrd1_(fcn3_2, &n, x, fvec, &tol, &info, wa, &lwa);
			fprintf(fp_w2_1, "x* = (");
			for (int i = 0; i < n; i++)
			{
				if (i + 1 < n) {
					fprintf(fp_w2_1, "%.15lf, ", x[i]);
				}
				else {
					fprintf(fp_w2_1, "%.15lf", x[i]);
				}
			}
			fprintf(fp_w2_1, ")\n");
			for (int i = 0; i < n; i++)
			{
				fprintf(fp_w2_1, "f(x*) = %.15lf\n", fvec[i]);
			}

			if (fp_w2_1 != NULL) fclose(fp_w2_1);
		}

		else {
			fscanf(fp_r2, "%lf %lf", &C, &b);
			fscanf(fp_r2, "%lf %lf %lf %lf %lf", &p[0][0], &p[0][1], &p[0][2], &t[0], &tr[0]);
			fscanf(fp_r2, "%lf %lf %lf %lf %lf", &p[1][0], &p[1][1], &p[1][2], &t[1], &tr[1]);
			fscanf(fp_r2, "%lf %lf %lf %lf %lf", &p[2][0], &p[2][1], &p[2][2], &t[2], &tr[2]);
			fscanf(fp_r2, "%lf %lf %lf %lf %lf", &p[3][0], &p[3][1], &p[3][2], &t[3], &tr[3]);
			hybrj1_(fcn3_1, &n, x, fvec, fjac, &ldfjac, &tol, &info, wa, &lwa);
			fprintf(fp_w1_2, "x* = (");
			for (int i = 0; i < n; i++)
			{
				if (i + 1 < n) {
					fprintf(fp_w1_2, "%.15lf, ", x[i]);
				}
				else {
					fprintf(fp_w1_2, "%.15lf", x[i]);
				}
			}
			fprintf(fp_w1_2, ")\n");
			for (int i = 0; i < n; i++)
			{
				fprintf(fp_w1_2, "f(x*) = %.15lf\n", fvec[i]);
			}
			if (fp_w1_2 != NULL) fclose(fp_w1_2);

			hybrd1_(fcn3_2, &n, x, fvec, &tol, &info, wa, &lwa);
			fprintf(fp_w2_2, "x* = (");
			for (int i = 0; i < n; i++)
			{
				if (i + 1 < n) {
					fprintf(fp_w2_2, "%.15lf, ", x[i]);
				}
				else {
					fprintf(fp_w2_2, "%.15lf", x[i]);
				}
			}
			fprintf(fp_w2_2, ")\n");
			for (int i = 0; i < n; i++)
			{
				fprintf(fp_w2_2, "f(x*) = %.15lf\n", fvec[i]);
			}
			if (fp_w2_2 != NULL) fclose(fp_w2_2);
		}

		flag++;
	}
	fclose(fp_r0);
	fclose(fp_r1);
	fclose(fp_r2);
}
