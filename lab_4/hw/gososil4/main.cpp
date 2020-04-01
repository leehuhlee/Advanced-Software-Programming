#include "my_solver.h"

extern void program1_1(FILE *fp); // Newton-Raphson Method
extern void program1_2(FILE *fp); // Secant Method
extern void program1_3(FILE *fp); // Newton-Raphson Method - Three-leaved clover
extern void program1_4(FILE *fp); // Bisection Method -- HOMEWORK
extern void program1_5(FILE *fp); // HOMEWORK

double (*_F)(double);
double (*_FP)(double);

double(*_f)(double, double, double);
double(*_fp)(double, double, double);

void main()
{
	FILE *fp;

	fp = fopen("result.txt", "w");

	/**********************
	  Problem 1-1
     **********************/
	// f1 = x^2 -4x +4 -lnx = 0
	_F = _F1;
	_FP = _FP1;

	program1_1(fp);
	program1_2(fp);


	 // f2 = x +1 -2sin(PI*x) = 0
	_F = _F2;
	_FP = _FP2;

	program1_1(fp);
	program1_2(fp);


	/**********************
	  Problem 1-2
	**********************/
	// f3 = x^4 -11.0x^3 +42.35x^2 -66.55x +35.1384 = 0
	_F = _F3;
	_FP = _FP3;

	for(int i = 3; i >= 0; i--)
		program1_1(fp);
	
	/**********************
	Problem 1-3
	**********************/
	// f4(x,y) = x^4 + 2x^2y^2 + y^4 - x^3 + 3xy^2 = 0, y = ax + b
	_f = _F4;
	_fp = _FP4;

	for (int i = 3; i >= 0; i--)
		program1_3(fp);

		for (int i = 3; i >= 0; i--)
		program1_3(fp);

			for (int i = 3; i >= 0; i--)
		program1_3(fp);


	/**********************
	Problem 2  -- HOMEWORK
	**********************/

	//hw1
	_F = _F1;
	_FP = _FP1;

	program1_4(fp);


	 // f2 = x +1 -2sin(PI*x) = 0
	_F = _F2;
	_FP = _FP2;

	program1_4(fp);

	_F = _F3;
	_FP = _FP3;

	program1_4(fp);


	//hw2
	_F = _F5;
	_FP = _FP5;
	program1_1(fp);


	//hw3
	_f = _F6;
	_fp = _FP6;
	program1_5(fp);

	program1_3(fp);

	fclose(fp);	
}