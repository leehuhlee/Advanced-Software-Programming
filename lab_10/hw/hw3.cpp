#include <stdio.h>
#include <time.h>
#include <math.h>
#include <random>
#include <Windows.h>

__int64 start, freq, end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f))
float compute_time;

#define TWO_23 223

void loop_interchange() {
	float Array_1[TWO_23][TWO_23],Array_2[TWO_23][TWO_23];
	int i, j;
	float run_time;

	printf("******************************\n");
	printf("*      Loop interchange      *\n");
	printf("******************************\n");

	srand(time(NULL));

	for (int i = 0; i < TWO_23; i++) {
		for (int j = 0; j < TWO_23; j++) {
			Array_1[i][j] = 0;
			Array_2[i][j] = rand() - RAND_MAX / 2.0f;
		}
	}
	printf("\nThe problem is to add one array of %d * %d elements...\n\n", TWO_23, TWO_23);

	///////////////////////////////////
	CHECK_TIME_START;
	for (int j = 0; j < TWO_23; j++) {
		for (int i = 0; i < TWO_23; i++) {
			Array_1[i][j] = Array_2[i][j];
		}
	}

	CHECK_TIME_END(run_time);

	printf("The runtime using a multiple loop is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_1[TWO_23-1][TWO_23 - 1]);
	///////////////////////////////////
	
	for (int i = 0; i < TWO_23; i++) {
		for (int j = 0; j < TWO_23; j++) {
			Array_1[i][j] = 0;
		}
	}
	//////////////////////////////////
	CHECK_TIME_START;
	for (int i = 0; i < TWO_23; i++) {
		for (int j = 0; j < TWO_23; j++) {
			Array_1[i][j] = Array_2[i][j];
		}
	}
	CHECK_TIME_END(run_time);

	printf("The runtime using a multiple loop is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_1[TWO_23 - 1][TWO_23 - 1]);
	///////////////////////////////////
	printf("******************************\n\n");
}


void loop_inversion() {
	float Array_1[TWO_23], Array_2[TWO_23];
	int i=0;
	float run_time;

	printf("******************************\n");
	printf("*       Loop inversion       *\n");
	printf("******************************\n");

	srand(time(NULL));

	for (int i = 0; i < TWO_23; i++) {
			Array_1[i] = 0;
			Array_2[i] = rand() - RAND_MAX / 2.0f;
	}
	i = 0;
	printf("\nThe problem is to add one array of %d elements...\n\n", TWO_23);

	///////////////////////////////////
	CHECK_TIME_START;
	while (i < TWO_23) {
		Array_1[i] = Array_2[i];
		i++;
	}

	CHECK_TIME_END(run_time);

	printf("The runtime using a while loop is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_1[TWO_23 - 1]);
	///////////////////////////////////

	for (int i = 0; i < TWO_23; i++) {
		Array_1[i] = 0;
	}
	i = 0;
	//////////////////////////////////
	CHECK_TIME_START;
	if (i < TWO_23) {
		do {
			Array_1[i] = Array_2[i];
			i++;
		} while (i < TWO_23);
	}
	CHECK_TIME_END(run_time);

	printf("The runtime using a do-while loop is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_1[TWO_23 - 1]);
	///////////////////////////////////
	printf("******************************\n\n");
}

void loop_invariant_code_motion() {
	float Array_1[TWO_23];
	int i = 0;
	float run_time;
	float x, y, z;
	float t1;
	printf("******************************\n");
	printf("* Loop-invariant code motion *\n");
	printf("******************************\n");

	srand(time(NULL));

	x =0;
	y = rand() - RAND_MAX / 2.0f;
	z = rand() - RAND_MAX / 2.0f;
	for (int i = 0; i < TWO_23; i++) {
		Array_1[i] = 0;
	}
	i = 0;
	printf("\nThe problem is to add one array of %d elements...\n\n", TWO_23);

	///////////////////////////////////
	CHECK_TIME_START;
	for (i = 0; i < TWO_23; ++i) {
		x = y + z;
		Array_1[i] = 6 * i + x * x;
	}

	CHECK_TIME_END(run_time);

	printf("The runtime using a loop is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_1[TWO_23 - 1]);
	///////////////////////////////////

	for (int i = 0; i < TWO_23; i++) {
		Array_1[i] = 0;
	}
	i = 0;
	x = 0;
	//////////////////////////////////
	CHECK_TIME_START;
	x = y + z;
	t1 = x * x;
	for (i = 0; i < TWO_23; ++i) {
		Array_1[i] = 6 * i + t1;
	}
	CHECK_TIME_END(run_time);

	printf("The runtime using a loop is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_1[TWO_23 - 1]);
	///////////////////////////////////
	printf("******************************\n\n");
}

void loop_splitting() {
	float Array_1[TWO_23], Array_2[TWO_23];
	int i = 0, j;
	float run_time;

	printf("******************************\n");
	printf("*       Loop splitting       *\n");
	printf("******************************\n");

	srand(time(NULL));

	for (int i = 0; i < TWO_23; i++) {
		Array_1[i] = 0;
		Array_2[i] = rand() - RAND_MAX / 2.0f;
	}
	i = 0;
	printf("\nThe problem is to add one array of %d elements...\n\n", TWO_23);

	///////////////////////////////////
	CHECK_TIME_START;
	j = TWO_23 - 1;
	for (i = 0; i < TWO_23; ++i) {
		Array_1[i] = Array_2[i] + Array_2[j];
		j = i;
	}

	CHECK_TIME_END(run_time);

	printf("The runtime using a loop is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_1[TWO_23 - 1]);
	///////////////////////////////////

	for (int i = 0; i < TWO_23; i++) {
		Array_1[i] = 0;
	}
	i = 0;

	//////////////////////////////////
	CHECK_TIME_START;
	Array_1[0] = Array_2[0] + Array_2[TWO_23-1];
	for (i = 1; i < TWO_23; ++i) {
		Array_1[i] = Array_2[i] + Array_2[i-1];
	}
	CHECK_TIME_END(run_time);

	printf("The runtime using a loop is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_1[TWO_23 - 1]);
	///////////////////////////////////
	printf("******************************\n\n");
}

void loop_tiling() {
	float Array_1[TWO_23][TWO_23], Array_2[TWO_23], Array_3[TWO_23];
	int i = 0, j, x, y;
	float run_time;

	printf("******************************\n");
	printf("*       Loop tiling       *\n");
	printf("******************************\n");

	srand(time(NULL));

	for (int i = 0; i < TWO_23; i++) {
		Array_2[i] = rand() - RAND_MAX / 2.0f;
		for (j = 0; j < TWO_23; j++) {
			Array_1[i][j] = rand() - RAND_MAX / 2.0f;
		}
	}
	printf("\nThe problem is to add one array of %d elements...\n\n", TWO_23);

	///////////////////////////////////
	CHECK_TIME_START;
	for (i = 0; i < TWO_23; i++) {
		Array_3[i] = 0;
		for (j = 0; j < TWO_23; j++) {
			Array_3[i] = Array_3[i] + Array_1[i][j] * Array_2[j];
		}
	}

	CHECK_TIME_END(run_time);

	printf("The runtime using a multiple loop is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_3[TWO_23 - 1]);
	///////////////////////////////////

	//////////////////////////////////
	CHECK_TIME_START;
	for (i = 0; i < TWO_23; i += 2) {
		Array_3[i] = 0;
		Array_3[i + 1] = 0;
		for (j = 0; j < TWO_23; j += 2) {
			for (x = i; x < min(i + 2, TWO_23); x++) {
				for (y = j; y < min(j + 2, TWO_23); y++) {
					Array_3[x] = Array_3[x] + Array_1[x][y] * Array_2[y];
				}
			}
		}

	}
	CHECK_TIME_END(run_time);

	printf("The runtime using a multiple loops is %.3f(ms).\n", run_time * 1000);
	printf("The sum is %e.\n\n", Array_3[TWO_23 - 1]);
	///////////////////////////////////
	printf("******************************\n\n");
}
void main() {
	loop_interchange();
	loop_inversion();
	loop_invariant_code_motion();
	loop_splitting();
	loop_tiling();
	Sleep(10000);
}