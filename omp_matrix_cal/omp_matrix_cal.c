#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

/*
* º∆À„æÿ’ÛA = B + Tr(B) °¡ C^2 + Tr(B^2) °¡ C^4
*/

//define size of matrixe
#define SIZE 350

//define number of threads
#define NUM_THREAD 12

void GenMatrix(double matrix[][SIZE]);
void MatrixMultiply(double result[SIZE][SIZE], double mat1[SIZE][SIZE], double mat2[SIZE][SIZE]);
double Trace(double matrix[SIZE][SIZE]);
void MulNumMatrix(double result[SIZE][SIZE], double matrix[SIZE][SIZE], double num);

int main(void) {
	double A[SIZE][SIZE];
	double B[SIZE][SIZE];
	double C[SIZE][SIZE];
	srand((unsigned int)time(NULL));
	GenMatrix(B);
	GenMatrix(C);
	double start, end;
	double totalTime = 0.0;

	double TrB = 0.0;
	double TrB2 = 0.0;	
	double matrix_C2[SIZE][SIZE];
	double matrix_TrBC2[SIZE][SIZE];
	double matrix_B2[SIZE][SIZE];
	double matrix_C4[SIZE][SIZE];
	double matrix_TrB2C4[SIZE][SIZE];

	start = omp_get_wtime();
	#pragma omp parallel num_threads(NUM_THREAD)
	{
		//caculate c^2
		#pragma omp single nowait
		{
			MatrixMultiply(matrix_C2, C, C);
		}
		
		//caculate Tr(B)
		TrB = Trace(B);
		#pragma omp barrier

		//caculate Tr(B)C^2
		MulNumMatrix(matrix_TrBC2, matrix_C2, TrB);
			
		
		#pragma omp single nowait
		{
			//caculate B^2
			MatrixMultiply(matrix_B2, B, B);
			//caculate Tr(B^2)
			TrB2 = Trace(matrix_B2);
		}

		//caculate C^4
		#pragma omp single nowait
		{
			MatrixMultiply(matrix_C4, matrix_C2, matrix_C2);
		}
		#pragma omp barrier

		//caculate Tr(B^2)C^4
		MulNumMatrix(matrix_TrB2C4, matrix_C4, TrB2);

		//caculate matrix A
		#pragma omp for collapse(2)
		for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < SIZE; j++) {
				A[i][j] = B[i][j] + matrix_TrBC2[i][j] + matrix_TrB2C4[i][j];
			}
		}
	}
	end = omp_get_wtime();
	totalTime = (end - start) * 1000000;
	printf("total time: %.0lf\n", totalTime);
	return 0;
}

void GenMatrix(double matrix[][SIZE]) {
	int i = 0;
	int j = 0;
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			matrix[i][j] = rand() % 5;
		}
	}
}

void MatrixMultiply(double result[SIZE][SIZE], double mat1[SIZE][SIZE], double mat2[SIZE][SIZE]) {
    	for (int i = 0; i < SIZE; i++) {
			#pragma omp task
        	for (int j = 0; j < SIZE; j++) {
            	result[i][j] = 0.0;
            	for (int k = 0; k < SIZE; k++) {
                		result[i][j] += mat1[i][k] * mat2[k][j];
            		}
        	}
    	}
}

double Trace(double matrix[SIZE][SIZE]) {
    	double trace = 0.0;
	#pragma omp parallel for reduction(+:trace)
    	for (int i = 0; i < SIZE; i++) {
        	trace += matrix[i][i];
    	}

    	return trace;
}

void MulNumMatrix(double result[SIZE][SIZE], double matrix[SIZE][SIZE], double num) {
	#pragma omp for collapse(2)
	for (int i = 0; i < SIZE; i++) {
        	for (int j = 0; j < SIZE; j++) {
            	result[i][j] = num * matrix[i][j];
        	}
    	}
}
