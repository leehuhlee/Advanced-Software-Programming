#include <stdio.h>
#include <time.h>
#include <math.h>
#include <random>
#include <Windows.h>

double sol1, sol2;
double fx_1, fx_2;

void equation(double a, double b, double c)
{
	fx_1 = a*pow(sol1, 2.0) + b*sol1 + c;
	fx_2 = a*pow(sol2, 2.0) + b*sol2 + c;
	printf("f(%lf) = %lf\nf(%lf) = %lf\n", sol1, fx_1, sol2, fx_2);
}

void solution2(double a, double b, double c)
{
	if (b*b - 4 * a*c < 0) {
		printf("해가 없다.\n");
	}
	else if (b*b - 4 * a*c == 0) {
		sol1 = (-1 * 4 * a * c) / (b + sqrt(b*b - 4 * a*c));
		sol1 = sol1 / (2 * a);
		sol2 = (-1 * b - sqrt(b*b - 4 * a*c)) / (2 * a);
		printf("중근: %.20lf, %.20lf\n", sol1, sol2);
		equation(a, b, c);
	}
	else {
		sol1 = (-1 * 4 * a * c) / (b + sqrt(b*b - 4 * a*c));
		sol1 = sol1 / (2 * a);
		sol2 = (-1 * b - sqrt(b*b - 4 * a*c)) / (2 * a);
		printf("solution: %.20lf, %.20lf\n", sol1, sol2);
		equation(a, b, c);
	}
}

void solution(double a, double b, double c)
{
	if (b*b - 4 * a*c < 0) {
		printf("해가 없다.\n");
	}
	else if (b*b - 4 * a*c == 0) {
		sol1 = (-1 * b + sqrt(b*b - 4 * a*c)) / (2 * a);
		sol2 = (-1 * b - sqrt(b*b - 4 * a*c)) / (2 * a);
		printf("중근: %.20lf, %.20lf\n", sol1, sol2);
		equation(a, b, c);
	}
	else {
		sol1 = (-1 * b + sqrt(b*b - 4 * a*c)) / (2 * a);
		sol2 = (-1 * b - sqrt(b*b - 4 * a*c)) / (2 * a);
		printf("solution: %.20lf, %.20lf\n", sol1, sol2);
		equation(a, b, c);
	}
}

void main()
{
	//while (1) {
		double a, b, c;
		printf("\nInput a, b, c: ");
		scanf("%lf %lf %lf", &a, &b, &c);
		printf("\n");
		solution(a, b, c);
		printf("------------------------------------------\n");
		solution2(a, b, c);
		printf("------------------------------------------\n");
	//}
}
