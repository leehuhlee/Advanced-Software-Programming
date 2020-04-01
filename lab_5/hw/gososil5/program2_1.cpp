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

	
	// Step 0: sampling_table.txt ������ ������ �Է� �޴´�.
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

	// Step 1: x�� ������ [0, 1] ���̷� ����ȭ�Ѵ�.
	// �̶� ����ȭ �ϸ鼭 x�� ������ �ٲ�� ������ ���߿� ���н� ��ٸ����� ���̰� �ٲ� ���� �����ؾ� �Ѵ�!!
	for(i=0;i<n_sampling;i++)
	{
		x[i]=(x[i]-start_x)/(end_x-start_x);
	}
	delta_h=delta_h/(end_x-start_x);
	
	
	// Step 2: � �Լ��� Ȯ���е��Լ�(pdf)�� ����� ���� ����ȭ�� x�� ��ü �������� y���� ��ġ������ �������� �� 1�� ������ y���� ��ȯ�Ѵ�. 
	// (���� �� � �Լ��� Ȯ���е� �Լ��� ������ ���������Ƿ� y�� p(x)��� �ϰڴ�.)
	// y -> p(x)�� �ٲٴ� ���� �����ڷ� 11�� ���� (�и��� x�� ��ü������ �����ϴ� ������ �ռ� ��ٸ��� ������ ����ϸ� 7�� 3.2 �κ� ����)
	total = 2*total-y[0]-y[n_sampling-1];
	total = total * delta_h * (double)0.5;

	for(i=0;i<n_sampling;i++)
		y[i]/=total;

	fprintf(fp_w,"%d %lf\n",n_sampling,delta_h); //for program2_2

	// Step 3: ������ ���� x�� p(x) ���� sampling_table.txt�� ���� �������� pdf_table.txt�� �����Ѵ�.
	for(i=0;i<n_sampling;i++)
	{
		fprintf(fp_w,"%lf %lf\n",x[i],y[i]);//normalized x, y (y->p(x))
		//sum+=y[i];							   in this case, the result is less than 1
		sum+=(y[i+1]+y[i])*(x[2]-x[1])/2;		// in this case, the result is 1
	}

	// Step 4: ������ ���� p(x)�� ���ؼ� x�� ��ü���� ������ ���� �� 1�� �ٻ��ϰ� �������� Ȯ���Ѵ�.
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
	
	// Console â�� �����ڷ� 12�� ������ �����ؼ� ����Ѵ�.
	if(fp_r != NULL) fclose(fp_r);
	if(fp_w != NULL) fclose(fp_w);
	free(x);
	free(y);
}