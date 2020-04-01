#include "my_solver.h"
#define M_PI 3.141592

/*
	½Ç½À
*/

// f1 = x^2 -4x +4 -lnx = 0
double _F1(double x) {
	return x*x - 4*x + 4 - log(x);
}
double _FP1(double x) {
	return 2*x - 4 - (1/x);
}

// f2 = x +1 -2sin(PI*x) = 0
double _F2(double x) {
	return x + 1 - 2*sin(M_PI*x);
}
double _FP2(double x) {
	return 1 - 2*M_PI*cos(M_PI*x);
}

// f3 = x^4 -11.0x^3 +42.35x^2 -66.55x +35.1384 = 0
double _F3(double x) {
	return x*x*x*x - 11.0*x*x*x + 42.35*x*x -66.55*x +35.1384;
}
double _FP3(double x) {
	return 4*x*x*x - 33.0*x*x + 84.70*x -66.55;
}

// f5(x,y) = x^4 + 2x^2y^2 + y^4 - x^3 + 3xy^2 = 0, y = ax + b
double _F4(double x, double a, double b) {
	return x*x*x*x + 2.0 * x*x*(a*x+b)*(a*x+b) 
		+ (a*x+b)*(a*x+b)*(a*x+b)*(a*x+b) 
		-x*x*x + 3.0 * x*(a*x+b)*(a*x+b);
}
double _FP4(double x, double a, double b) {
	return 4.0*a*(a*x+b)*(a*x+b)*(a*x+b) 
		+ 4.0*x*(a*x+b)*(a*x+b)
		+ 3.0*(a*x+b)*(a*x+b) 
		+ 4.0*x*x*x - 3.0*x*x 
		+ 4.0*a*x*x*(a*x+b) + 6.0*a*x*(a*x+b);
}

// f5 = 89*sin*(11.5)*sin(x)*cos(x) +89*cos(11.5)*cos(x)*cos(x) -((49+0.5*55)*sin(11.5) - 0.5*55*tan(11.5))*cos(x) - ((49+0.5*55)*cos(11.5) - 0.5*55)*sin(x) = 0
double _F5(double x) {
	return 89*sin(11.5/180*M_PI)*sin(x)*cos(x) +89*cos(11.5/180*M_PI)*sin(x)*sin(x) -((49+0.5*55)*sin(11.5/180*M_PI) - 0.5*55*tan(11.5/180*M_PI))*cos(x) - ((49+0.5*55)*cos(11.5/180*M_PI) - 0.5*55)*sin(x);
}
double _FP5(double x) {
	return 89*sin(11.5/180*M_PI)*(cos(x)*cos(x) - sin(x)*sin(x))+ 2*89*cos(11.5/180*M_PI)*sin(x)*cos(x) +  ((49+0.5*55)*sin(11.5/180*M_PI) - 0.5*55*tan(11.5/180*M_PI))*sin(x) - ((49+0.5*55)*cos(11.5/180*M_PI) - 0.5*55)*cos(x);
}

double _F6(double x, double a, double b) {
	return cos(3*x)*sin(x)-a*cos(3*x)*cos(x) -b;
}
double _FP6(double x, double a, double b) {
	return -3*sin(3*x)*sin(x) + cos(3*x)*cos(x) - 3*a*sin(3*x)*cos(x) + a*cos(3*x)*sin(x);
}