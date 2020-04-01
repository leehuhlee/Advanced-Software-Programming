#include "my_solver.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>

// global variables
const double DELTA = 0.000000001;
const int Nmax = 100;
const double EPSILON = 0.000000001;
#define START 0
#define END 1
void program2_2()
{
	FILE *fp_r, *fp_w;	
	int n_r;


	int n_sampling;    
	double delta_h;   // (b-a)/n
	double *x,*y; //array for x,y values
	int i,j;
	double cumulative;
	double *cdf;
	int n;
	double x0,x1;
	double middle;
	double temp;
	double U;
	//double *random;
	double under, upper;
	double compare;

	srand(time(NULL));

	fp_r = fopen("pdf_table.txt", "r");
	fp_w = fopen("random_event_table.txt", "w");

	scanf("%d", &n_r);
	fprintf(fp_w,"%d\n", n_r);
	
	// Step 0: pdf_table.txt ������ ������ �Է� �޴´�.
	fscanf(fp_r, "%d %lf", &n_sampling,&delta_h);
	x=(double*)malloc(sizeof(double)*n_sampling);
	y=(double*)malloc(sizeof(double)*n_sampling);

	cdf=(double*)malloc(sizeof(double)*n_sampling);

	for(i=0;i<n_sampling;i++)
	{
		fscanf(fp_r,"%lf %lf", &x[i],&y[i]);
	}

	// Step 1: ������ f(x) = F(x) - U = 0�� �����ϴ� f(x)�� ����� ���� ���� ���� ���� �Լ��� F(x)�� �����Ѵ�.
	cumulative=0;
	for(i=0;i<n_sampling;i++)
	{
		cdf[i]=0;
		for(j=0;j<i;j++)
		{
			cdf[i]+=(y[j]+y[j+1])*(x[2]-x[1])/2;
		}
	}
	
	// ���� ���� �Լ��� ���Ǵ� 4���� �����ϸ�, ������ �ռ� ��ٸ��� ������ ����ϸ� 7�� 3.2 �κ� ����

	while (n_r--) 
	{

		U = (double)rand() / RAND_MAX; // [0, 1] ���̿� �����ϴ� ������ �� U
		x0=0.0;
		x1=1.0;

		under=cdf[0];
		upper=cdf[n_sampling-1];
		middle = (x0+x1)/2;
		
		for (n = 0; ; n++) 
		{
			for(i=0;i<n_sampling;i++)
			{
				if(x[i]<middle && x[i+1]>=middle)
					break;
			}

			temp=cdf[i]+(cdf[i+1]-cdf[i])*(middle-x[i])/(x[i+1]-x[i]);//���� ���� - ������ �̿�.
			
			compare=middle;//compare= problem 1_2������ x0����

			
			if((under-U)*(temp-U)<0)
			{
				x1=middle; //!! not x0
				upper=temp;//?
			}
			else
			{
				x0=middle;
				under=temp;
			}
			middle = (double)(x0+x1)/2;
			//middle= problem 1_2������ x1����
			if( fabs(middle) < DELTA || n>=Nmax || fabs(middle-compare)<EPSILON )
			{				
				n++;
				break;
			}
			
			
		}
		fprintf(fp_w,"%.15lf \n",temp);


		// Step 2: 4���� ������ �ۼ��ߴ� Bisection ����� ����� f(x*) = 0���� ����� �ִ� x* ���� ���Ѵ�. (����� DELTA, Nmax, EPSILON�� ���� ������� ���� ����)
		// �̶� Bisection�� �ʱ� ������ x�� [0, 1]�� ����ȭ �Ǿ����Ƿ� �ʱⰪ�� 0�� 1�� �д�. �̶��� ������ x*�̶�� ����.
		// �׷��� �츮�� ���Ϸ��� �ϴ� x*�� �̹� ��� ������ �� ���� sampling�� �����ϴ� x�� �ƴ� �ٸ� ���� ���� x*�� ã�� ���̴�.
		// ���� x*�� �ش��ϴ� F(x*)���� �𸣹Ƿ� f(x) = 0 �������� Ǯ�� ����ϴ�.
		// Step 3: ���� F(x*) ���� ��������(linear interpolation)�� �����Ͽ� �ٻ������� F(x*)�� ���Ѵ�.(�������� ������ �˻�, �ǽ��ð� ���� ����, 17�� 5.2 �κ� ����)
		
		// Step 4: �� ����� ���� ���� x*�� ��� ���˿� �°� random_event_table.txt�� ����Ѵ�.(15�� ����)
	}

	if(fp_r != NULL) fclose(fp_r);
	if(fp_w != NULL) fclose(fp_w);
}
