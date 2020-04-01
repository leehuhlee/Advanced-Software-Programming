#ifndef _MY_SOLVER
#define _MY_SOLVER

#include <stdio.h>
#include <Windows.h>

#define CHECK_TIME_START QueryPerformanceFrequency((_LARGE_INTEGER*) &freq); QueryPerformanceCounter((_LARGE_INTEGER*)&start);
#define CHECK_TIME_END(a) QueryPerformanceCounter((_LARGE_INTEGER*)&end); a = (float)((float) (end - start)/freq);

void program2_1(void);
void program2_2(void);

// HOMEWORK
void program2_2_a(void);
void program2_2_b(void);
void program2_3(void);

#endif // _MY_SOLVER