#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 512


/*
* 计算矩阵A = B + Tr(B) × C^2 + Tr(B^2) × C^4
*/

void GenMatrix(double matrix[SIZE][SIZE]);
void MatrixMultiply(double result[SIZE][SIZE], double mat1[SIZE][SIZE], double mat2[SIZE][SIZE], int start_row, int end_row);
double Trace(double matrix[SIZE][SIZE], int start_row, int end_row);
void MulNumMatrix(double result[SIZE][SIZE], double matrix[SIZE][SIZE], double num, int start_row, int end_row);

int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (SIZE % size != 0) {
		if (rank == 0) printf("矩阵大小应是进程数的倍数。\n");
		MPI_Finalize();
		return -1;
    	}

	int rows_per_proc = SIZE / size;

	double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
	double matrix_C2[SIZE][SIZE];
	double matrix_TrBC2[SIZE][SIZE];
	double matrix_B2[SIZE][SIZE]; 
	double matrix_C4[SIZE][SIZE];
	double matrix_TrB2C4[SIZE][SIZE];

	double TrB = 0.0, TrB2 = 0.0;

	double start_time = 0.0, end_time = 0.0, total_time = 0.0;

	if (rank == 0) {
		srand(time(NULL));
		GenMatrix(B);
		GenMatrix(C);

		start_time = MPI_Wtime();
	}

	MPI_Bcast(B, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(C, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int start_row = rank * rows_per_proc;
	int end_row = (rank + 1) * rows_per_proc;

	// Calculate Tr(B)
	double local_TrB = Trace(B, start_row, end_row);
	MPI_Reduce(&local_TrB, &TrB, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&TrB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Calculate C^2
	MatrixMultiply(matrix_C2, C, C, start_row, end_row);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, matrix_C2, rows_per_proc * SIZE, MPI_DOUBLE, MPI_COMM_WORLD);

	// Calculate Tr(B) * C^2
	MulNumMatrix(matrix_TrBC2, matrix_C2, TrB, start_row, end_row);

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, matrix_TrBC2, rows_per_proc * SIZE, MPI_DOUBLE, MPI_COMM_WORLD);

	// Calculate B^2
	MatrixMultiply(matrix_B2, B, B, start_row, end_row);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, matrix_B2, rows_per_proc * SIZE, MPI_DOUBLE, MPI_COMM_WORLD);

	// Calculate Tr(B^2)
	double local_TrB2 = Trace(matrix_B2, start_row, end_row);
	MPI_Reduce(&local_TrB2, &TrB2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&TrB2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Calculate C^4
	MatrixMultiply(matrix_C4, matrix_C2, matrix_C2, start_row, end_row);	 
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, matrix_C4, rows_per_proc * SIZE, MPI_DOUBLE, MPI_COMM_WORLD);

	// Calculate Tr(B^2) * C^4
	MulNumMatrix(matrix_TrB2C4, matrix_C4, TrB2, start_row, end_row);

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, matrix_TrB2C4, rows_per_proc * SIZE, MPI_DOUBLE, MPI_COMM_WORLD);

	// Calculate A
	for (int i = start_row; i < end_row; i++) {
		for (int j = 0; j < SIZE; j++) {
	    		A[i][j] = B[i][j] + matrix_TrBC2[i][j] + matrix_TrB2C4[i][j];
		}
	}
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, A, rows_per_proc * SIZE, MPI_DOUBLE, MPI_COMM_WORLD);	
	if (rank == 0) {
		end_time = MPI_Wtime();
		total_time = (end_time - start_time) * 1000;
		printf("%.0lf\n", total_time);
	}

	MPI_Finalize();
	return 0;
}

void GenMatrix(double matrix[SIZE][SIZE]) {
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
	    		matrix[i][j] = rand() % 5;
		}
	}
}

void MatrixMultiply(double result[SIZE][SIZE], double mat1[SIZE][SIZE], double mat2[SIZE][SIZE], int start_row, int end_row) {
	for (int i = start_row; i < end_row; i++) {
		for (int j = 0; j < SIZE; j++) {
	    		result[i][j] = 0.0;
	    		for (int k = 0; k < SIZE; k++) {
				result[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
}

double Trace(double matrix[SIZE][SIZE], int start_row, int end_row) {
	double trace = 0.0;
	for (int i = start_row; i < end_row; i++) {
		trace += matrix[i][i];
	}
	return trace;
}

void MulNumMatrix(double result[SIZE][SIZE], double matrix[SIZE][SIZE], double num, int start_row, int end_row) {
	for (int i = 0; i < SIZE; i++) {
        	for (int j = 0; j < SIZE; j++) {
            	result[i][j] = num * matrix[i][j];
        	}
    	}
}
