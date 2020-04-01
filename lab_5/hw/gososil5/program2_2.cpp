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
	double under, upper;
	double compare;

	srand(time(NULL));

	fp_r = fopen("pdf_table.txt", "r");
	fp_w = fopen("random_event_table.txt", "w");

	scanf("%d", &n_r);
	fprintf(fp_w,"%d\n", n_r);
	
	// Step 0: pdf_table.txt 파일의 내용을 입력 받는다.
	fscanf(fp_r, "%d %lf", &n_sampling,&delta_h);
	x=(double*)malloc(sizeof(double)*n_sampling);
	y=(double*)malloc(sizeof(double)*n_sampling);

	cdf=(double*)malloc(sizeof(double)*n_sampling);

	for(i=0;i<n_sampling;i++)
	{
		fscanf(fp_r,"%lf %lf", &x[i],&y[i]);
	}

	// Step 1: 방정식 f(x) = F(x) - U = 0을 정의하는 f(x)를 만들기 위해 먼저 누적 분포 함수인 F(x)를 정의한다.
	cumulative=0;
	for(i=0;i<n_sampling;i++)
	{
		cdf[i]=0;
		for(j=0;j<i;j++)
		{
			cdf[i]+=(y[j]+y[j+1])*(x[2]-x[1])/2;
		}
	}
	
	// 누적 분포 함수의 정의는 4쪽을 참고하며, 구현은 합성 사다리꼴 공식을 사용하며 7쪽 3.2 부분 참고

	while (n_r--) 
	{

		U = (double)rand() / RAND_MAX; // [0, 1] 사이에 존재하는 임의의 값 U
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

			temp=cdf[i]+(cdf[i+1]-cdf[i])*(middle-x[i])/(x[i+1]-x[i]);//선형 보간 - 닮음을 이용.
			
			compare=middle;//compare= problem 1_2에서의 x0역할

			
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
			//middle= problem 1_2에서의 x1역할
			if( fabs(middle) < DELTA || n>=Nmax || fabs(middle-compare)<EPSILON )
			{				
				n++;
				break;
			}
			
			
		}
		fprintf(fp_w,"%.15lf \n",temp);


		// Step 2: 4주차 숙제로 작성했던 Bisection 방법을 사용해 f(x*) = 0으로 만들어 주는 x* 값을 구한다. (상수인 DELTA, Nmax, EPSILON은 본인 마음대로 조정 가능)
		// 이때 Bisection의 초기 구간은 x가 [0, 1]로 정규화 되었으므로 초기값을 0과 1로 둔다. 이때의 중점을 x*이라고 하자.
		// 그러나 우리가 구하려고 하는 x*은 이미 곡선을 설계할 때 사용된 sampling에 존재하는 x가 아닌 다른 유사 난수 x*를 찾는 것이다.
		// 따라서 x*에 해당하는 F(x*)값을 모르므로 f(x) = 0 방정식을 풀기 곤란하다.
		// Step 3: 따라서 F(x*) 값은 선형보간(linear interpolation)을 적용하여 근사적으로 F(x*)을 구한다.(선형보간 내용은 검색, 실습시간 설명 내용, 17쪽 5.2 부분 참고)
		
		// Step 4: 위 방법을 통해 구한 x*을 출력 포맷에 맞게 random_event_table.txt에 출력한다.(15쪽 참고)
	}

	if(fp_r != NULL) fclose(fp_r);
	if(fp_w != NULL) fclose(fp_w);
}
