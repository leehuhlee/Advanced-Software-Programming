#include "my_solver.h"

void program2_1()
{
	FILE *fp_r, *fp_w;
	__int64 start, freq, end;
	float resultTime = 0;

	int n_sampling;    
	double delta_h;   // (b-a)/n
	double *x,*y; //array for x,y values
	int i;
	double sum=0.0;
	double sub_s[4]={0.0};	
	double total=0.0;
	double start_x,end_x;
	fp_r = fopen("sampling_table.txt", "r");		
	if(fp_r == NULL) {
		printf("input file not found....\n");
		exit(0);
	}

	fp_w = fopen("pdf_table.txt", "w");

	
	// Step 0: sampling_table.txt 파일의 내용을 입력 받는다.
	fscanf(fp_r,"%d %lf",&n_sampling,&delta_h);
	x=(double*)malloc(sizeof(double)*n_sampling);
	y=(double*)malloc(sizeof(double)*n_sampling);
	for(i=0;i<n_sampling;i++)
	{
		fscanf(fp_r,"%lf %lf", &x[i],&y[i]);
		total+=y[i];
	}	
	start_x = x[0];
	end_x   = x[n_sampling-1];

	// Step 1: x의 구간을 [0, 1] 사이로 정규화한다.
	// 이때 정규화 하면서 x의 간격이 바뀌였기 때문에 나중에 적분시 사다리꼴의 높이가 바뀐 것을 주의해야 한다!!
	for(i=0;i<n_sampling;i++)
	{
		x[i]=(x[i]-start_x)/(end_x-start_x);
	}
	delta_h=delta_h/(end_x-start_x);
	
	
	// Step 2: 곡선 함수를 확률밀도함수(pdf)로 만들기 위해 정규화한 x의 전체 구간에서 y값을 수치적으로 적분했을 때 1이 나오게 y값을 변환한다. 
	// (이제 이 곡선 함수는 확률밀도 함수의 조건을 만족했으므로 y를 p(x)라고 하겠다.)
	// y -> p(x)로 바꾸는 식은 강의자료 11쪽 참고 (분모의 x의 전체구간을 적분하는 수식은 합성 사다리꼴 공식을 사용하며 7쪽 3.2 부분 참고)
	total = 2*total-y[0]-y[n_sampling-1];
	total = total * delta_h * (double)0.5;

	for(i=0;i<n_sampling;i++)
		y[i]/=total;

	fprintf(fp_w,"%d %lf\n",n_sampling,delta_h); //for program2_2

	// Step 3: 위에서 구한 x와 p(x) 값을 sampling_table.txt와 같은 포맷으로 pdf_table.txt에 저장한다.
	for(i=0;i<n_sampling;i++)
	{
		fprintf(fp_w,"%lf %lf\n",x[i],y[i]);//normalized x, y (y->p(x))
		//sum+=y[i];							   in this case, the result is less than 1
		sum+=(y[i+1]+y[i])*(x[2]-x[1])/2;		// in this case, the result is 1
	}

	// Step 4: 위에서 구한 p(x)에 대해서 x의 전체구간 적분을 했을 때 1에 근사하게 나오는지 확인한다.
	printf("Integrating the pdf from 0.00 to 1.00 %lf \n",sum);

	for(i=0;i<(int)(n_sampling*0.25);i++)
	{
		sub_s[0]+=y[i];
	}
	printf("Integrating the pdf from 0.00 to 0.25 %lf \n",sub_s[0]/(n_sampling));

	for(i=(int)(n_sampling*0.25);i<(int)(n_sampling*0.5);i++)
	{
		sub_s[1]+=y[i];
	}
	printf("Integrating the pdf from 0.25 to 0.05 %lf \n",sub_s[1]/(n_sampling));

	for(i=(int)(n_sampling*0.5);i<(int)(n_sampling*0.75);i++)
	{
		sub_s[2]+=y[i];
	}
	printf("Integrating the pdf from 0.50 to 0.75 %lf \n",sub_s[2]/(n_sampling));

	for(i=(int)(n_sampling*0.75);i<(n_sampling);i++)
	{
		sub_s[3]+=y[i];
	}
	printf("Integrating the pdf from 0.75 to 1.00 %lf \n",sub_s[3]/(n_sampling));
	
	// Console 창에 강의자료 12쪽 포맷을 참고해서 출력한다.
	if(fp_r != NULL) fclose(fp_r);
	if(fp_w != NULL) fclose(fp_w);
	free(x);
	free(y);
}