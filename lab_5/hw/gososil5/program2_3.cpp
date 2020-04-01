#include "my_solver.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>

// global variables
const double DELTA = 0.000000000001;
const int Nmax = 100;
const double EPSILON = 0.00000000001;

double _F2(double X, double *data_x, double *data_y, int n) {
	int i;
	for (i = 0; i < n - 1; i++)
		if (data_x[i] <= X && X < data_x[i + 1]) break;

	double y_base = data_y[i];
	double y_diff = data_y[i + 1] - data_y[i];

	double r_base = data_x[i + 1] - data_x[i];
	double r_0 = (X - data_x[i]);
	double r_1 = (data_x[i + 1] - X);

	return y_base + y_diff * r_0 / r_base;
}

// HOMEWORK
void program2_4()
{
	printf("\n-------------------------------------------------------------\n");
	printf("program2_4\n");
	double Lambda[3] = { 1, 2, 3 };
	double U;
	double X1, X2;

	int n = 1000;
	for (int i = 0; i < 3; i++) {
		U = (double)rand() / RAND_MAX; //uniform random numbers
		X1 = -1 / Lambda[i] * log(1 - U); //역변환법
		X2 = exp(Lambda[i]);// R function rexp(n,lambda)
		printf("exp(%.1f) = %f : log(exp(%.1f)) = %.1f\n", Lambda[i], X2, Lambda[i], X1);
	}
}

// HOMEWORK
void program2_3()
{
	FILE *fp_r_pdf, *fp_r_rand, *fp_w;
	fp_r_pdf = fopen("pdf_table.txt", "r");
	fp_r_rand = fopen("random_event_table_2_b.txt", "r");
	fp_w = fopen("histogram.txt", "w");

	int n, N;
	double h;
	fscanf(fp_r_pdf, "%d %lf", &n, &h);
	fscanf(fp_r_rand, "%d", &N);

	double *x = new double[n];
	double *fx = new double[n];
	double *rand = new double[N];
	int *histo = new int[n];

	for (int i = 0; i < n; i++) {
		fscanf(fp_r_pdf, "%lf %lf", &x[i], &fx[i]);
		histo[i] = 0;
	}

	for (int i = 0; i < N; i++) {
		fscanf(fp_r_rand, "%lf", &rand[i]);

		for (int k = 0; k < n; k++) {
			if (x[k] <= rand[i] && rand[i] < x[k + 1]) {
				histo[k]++;
				break;
			}
		}
	}

	for (int i = 0; i < n - 1; i++) {
		fprintf(fp_w, "%lf ~ %lf : %d\n", x[i], x[i + 1], histo[i]);
	}

	delete[] x;
	delete[] fx;
	delete[] rand;
	delete[] histo;
}

// HOMEWORK
void program2_2_a()
{
	printf("\n-------------------------------------------------------------\n");
	printf("program2_2_a: bisection\n");

	__int64 start, freq, end;
	float resultTime = 0;
	double U;

	FILE *fp_r, *fp_w;
	int n_r;

	fp_r = fopen("pdf_table.txt", "r");
	fp_w = fopen("random_event_table_2_a.txt", "w");

	printf("n_r: ");
	scanf("%d", &n_r);
	fprintf(fp_w, "%d\n", n_r);

	srand(time(NULL));

	int n;
	double fre;
	fscanf(fp_r, "%d %lf", &n, &fre);
	double *x = new double[n];
	double *y_tmp = new double[n];
	double *y = new double[n];

	for (int i = 0; i < n; i++) {
		fscanf(fp_r, "%lf %lf", &x[i], &y_tmp[i]);
	}

	double fre_half = fre / 2.0;
	y[0] = 0;
	y[1] = fre_half * (y_tmp[0] + y_tmp[1]);
	for (int i = 2; i < n; i++) {
		y[i] = y[i - 1] + fre_half * (y_tmp[i - 1] + y_tmp[i]);
	}

	CHECK_TIME_START;

	// something to do...
	while (n_r--) {
		U = (double)rand() / RAND_MAX;

		double x0 = INT_MAX;
		double a0 = 0, b0 = 1, temp;
		double x1 = (a0 + b0) / 2;

		for (;;) {
			if (fabs(_F2(x1, x, y, 100) - U) < DELTA)
				break;

			if (fabs(x1 - x0) < EPSILON)
				break;

			double value_a = _F2(a0, x, y, 100) - U;
			double value_b = _F2(b0, x, y, 100) - U;

			temp = (a0 + b0) / 2.0;
			double value_ab = _F2(temp, x, y, 100) - U;

			if (value_a * value_ab > 0)
				a0 = temp;

			else if (value_b * value_ab > 0)
				b0 = temp;

			else {
				if (fabs(value_a) < fabs(value_b))
					b0 = temp;

				else
					a0 = temp;
			}

			x0 = x1;
			x1 = (a0 + b0) / 2;
		}
		fprintf(fp_w, "%.15lf\n", x1);
	}

	CHECK_TIME_END(resultTime);
	delete[] x;
	delete[] y_tmp;
	delete[] y;

	if (fp_r != NULL) fclose(fp_r);
	if (fp_w != NULL) fclose(fp_w);

	printf("The program2_2_a run time is %f(ms)..\n", resultTime*1000.0);
}

int idx;

double _F2_(double X, double *data_x, double *data_y, int n) {
	double r1, r2;
	double y;

	for (int i = 0; i < n - 1; i++) {
		if (data_x[i] <= X && X <= data_x[i + 1]) {
			idx = i;
			break;
		}
	}

	r1 = X - data_x[idx];
	r2 = data_x[idx + 1] - X;
	y = (r2 * data_y[idx] + r1 * data_y[idx + 1]) / (r1 + r2);

	return y;
}

double _F2P_(double X, double *data_x, double *data_y, int n) {
	double s;
	double r1, r2;
	double y;

	for (int i = 0; i < n - 1; i++) {
		if (data_x[i] <= X && X <= data_x[i + 1]) {
			idx = i;
			break;
		}
	}

	s = (X - data_x[idx]) / (data_x[idx + 1] - data_x[idx]);
	return (1 - s) * data_y[idx] + s * data_y[idx + 1];
}

void program2_2_b()
{
	printf("\n-------------------------------------------------------------\n");
	printf("program2_2_b: bisection + secant\n");

	__int64 start, freq, end;
	float resultTime = 0;
	double U;

	FILE *fp_r, *fp_w;
	int n_r, bi_n;

	fp_r = fopen("pdf_table.txt", "r");
	fp_w = fopen("random_event_table_2_b.txt", "w");

	printf("total number: ");
	scanf("%d", &n_r);
	printf("bisection method number: ");
	scanf("%d", &bi_n);
	//fprintf(fp_w, "%d / %d\n", n_r, bi_n);
	fprintf(fp_w, "%d\n", n_r);

	srand(time(NULL));

	int n;
	double fre;
	fscanf(fp_r, "%d %lf", &n, &fre);
	double *x = new double[n];
	double *y_tmp = new double[n];
	double *y = new double[n];
	for (int i = 0; i < n; i++) {
		fscanf(fp_r, "%lf %lf", &x[i], &y_tmp[i]);
	}

	double fre_half = fre / 2.0;
	y[0] = 0;
	y[1] = fre_half * (y_tmp[0] + y_tmp[1]);
	for (int i = 2; i < n; i++) {
		y[i] = y[i - 1] + fre_half * (y_tmp[i - 1] + y_tmp[i]);
	}


	CHECK_TIME_START;

	// something to do...
	int i = 0;
	while (n_r--) {
		U = (double)rand() / RAND_MAX;

		double x0 = INT_MAX;
		double a0 = 0, b0 = 1, temp;
		double x1 = (a0 + b0) / 2;

		for (int j = 0; j < bi_n; j++) {
			if (fabs(_F2(x1, x, y, 100) - U) < DELTA)
				break;

			if (fabs(x1 - x0) < EPSILON)
				break;

			double value_a = _F2(a0, x, y, 100) - U;
			double value_b = _F2(b0, x, y, 100) - U;

			temp = (a0 + b0) / 2.0;
			double value_ab = _F2(temp, x, y, 100) - U;

			if (value_a * value_ab > 0)
				a0 = temp;

			else if (value_b * value_ab > 0)
				b0 = temp;

			else {
				if (fabs(value_a) < fabs(value_b))
					b0 = temp;

				else
					a0 = temp;
			}

			x0 = x1;
			x1 = (a0 + b0) / 2;
		}

		x0 = x1;
		x1 = x0;

		for (int k = 0; k < 50; k++) {
			if (fabs(_F2(x1, x, y, 100) - U) < DELTA)
				break;

			if (fabs(x1 - x0) < EPSILON)
				break;

			x0 = x1;
			x1 = x0 - (_F2_(x1, x, y, 100) - U) / _F2P_(x1, x, y, 100);

			if (x1 > 1)
				x1 = 1;

			if (x1 < 0)
				x1 = 0;
		}
		//fprintf(fp_w, "%.15lf / %.15lf\n", U, x1);
		fprintf(fp_w, "%.15lf\n", x1);
	}
	CHECK_TIME_END(resultTime);

	if (fp_r != NULL) fclose(fp_r);
	if (fp_w != NULL) fclose(fp_w);

	printf("The program2_2_b run time is %f(ms)..\n", resultTime*1000.0);
}