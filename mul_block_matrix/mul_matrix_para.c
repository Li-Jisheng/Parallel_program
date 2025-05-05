#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
//#include <mpi.h>

// define the size of matrix
#ifndef MATRIX
#define MATRIX 2880
#endif

// define the size of block
// 块矩阵大小通过环境变量修改，测试取值如下：1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 32, 36, 40, 45, 48, 60, 64, 72, 80, 90, 96, 120, 144, 160, 180, 192, 240, 288, 320, 360, 480, 576, 720, 960, 1440, 2880}
#ifndef BLOCK
#define BLOCK 32
#endif

// define the type of elements
#define TYPE int
// define the number of threads
#ifndef THREADS
#define THREADS 16
#endif

void MultipyMatrix(TYPE** result, TYPE** A, TYPE** B, int matrixSize, int blockSize);

int main() {
    srand(time(NULL));
    TYPE **A, **B, **C; //二维数组
    

    //动态分配数组空间
    int sizeMatrix = sizeof(TYPE*) * MATRIX; //二维数组大小
    int sizeRow = sizeof(TYPE*) * MATRIX; //二维数组一行的大小
    

    A = (TYPE**)malloc(sizeMatrix);
    B = (TYPE**)malloc(sizeMatrix);
    C = (TYPE**)malloc(sizeMatrix);
    

    //初始化矩阵
    int i = 0, j = 0, k = 0;
    for (i = 0; i < MATRIX; i++) {
        A[i] = (TYPE*)malloc(sizeRow);
        B[i] = (TYPE*)malloc(sizeRow);
        C[i] = (TYPE*)malloc(sizeRow);
        for (j = 0; j < MATRIX; j++) {
            A[i][j] = (i <= j) ? rand() % 2 * 2 - 1 : 0;
            B[i][j] = (i <= j) ? rand() % 2 * 2 - 1 : 0;
            C[i][j] = 0;
        }
    }

    //计算C矩阵
    MultipyMatrix(C, A, B, MATRIX, BLOCK);
   
    //释放空间
    for (int i = 0; i < MATRIX; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}

void MultipyMatrix(TYPE** C, TYPE** A, TYPE** B, int matrixSize, int blockSize) {
    TYPE *AA, *BB; //一维数组

    //动态分配一维数组空间
    int size1D = sizeof(TYPE) * ((matrixSize * matrixSize + matrixSize * blockSize) / 2); //一维数组大小
    AA = (TYPE*)malloc(size1D);
    BB = (TYPE*)malloc(size1D);

    //将二维数组转换成一维数组
    int kk = 0;
    for (int i = 0; i < matrixSize; i += blockSize) {
        for (int j = i; j < matrixSize; j += blockSize) {
            for (int ii = 0; ii < blockSize; ii++) {
                for (int jj = 0; jj < blockSize; jj++) {
                    AA[kk] = A[i + ii][j + jj];
                    kk++;
                }
            }
        }
    }

    kk = 0;
    for (int j = 0; j < matrixSize; j += blockSize) {
        for (int i = 0; i <= j; i += blockSize) {
            for (int jj = 0; jj < blockSize; jj++) {
                for (int ii = 0; ii < blockSize; ii++) {
                    BB[kk] = B[i + ii][j + jj];
                    kk++;
                }
            }
        }
    }

    

    //计算C矩阵
    double start = omp_get_wtime();
    int n_blocks = matrixSize / blockSize;
    #pragma omp parallel num_threads(THREADS) 
    {
	    #pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < matrixSize; i += blockSize) {
            for (int j = 0; j < matrixSize; j += blockSize) {
                for (int k = i; k <= j; k += blockSize) {
                    int i_block = i / blockSize;
                    int j_block = j / blockSize;
                    int k_block = k / blockSize;

                    //块的起始坐标
                    int A0 = blockSize * blockSize * (i_block * (2 * n_blocks - i_block + 1) / 2 + (k_block - i_block));
                    int B0 = blockSize * blockSize * ((j_block + 1) * j_block / 2 + k_block);

                    // Process block (i,j,k_block)  
                    //循环顺序i1 -> k1 -> j1的顺序要比i1 -> j1 ->k1的顺序更好，
                    //因为这个顺序能保证对AA和BB的访问模式均为连续访问
                    //另外，在k1固定时，a_val的值在j1循环中可被寄存器缓存，无需反复从内存或缓存中加载
                    for (int i1 = 0; i1 < blockSize; i1++) {
                        for (int k1 = 0; k1 < blockSize; k1++) {
                            int a_offset = A0 + i1 * blockSize + k1;
                            TYPE a_val = AA[a_offset];
                            for (int j1 = 0; j1 < blockSize; j1++) {
                                int b_offset = B0 + j1 * blockSize + k1;
                                C[i + i1][j + j1] += a_val * BB[b_offset];
                            }
                        }
                    }
                }
            }
        }
        
    }
    double end = omp_get_wtime();

    printf("Thread: %d, Block size: %d, Time: %.2f ms\n", THREADS, BLOCK, (end - start) * 1000);


    free(AA);
    free(BB);
}
