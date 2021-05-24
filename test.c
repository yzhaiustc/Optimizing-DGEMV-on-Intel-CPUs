#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "utils.h"
#include "mkl.h"

int main(int argc, char *argv[])
{
    if (argc != 4) {
        printf("please input [m] [n] [kernel_num].\n");
        printf("kernel_num == 1: mydgemv (default).\n");
        printf("kernel_num == 2: MKL DGEMV.\n");
        exit(-1);
    }
    int m, n, kernel_num = 1;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    kernel_num = atoi(argv[3]);

    if ( (kernel_num!=1&&kernel_num!=2) || m <= 0 || n <= 0 ) {
        printf("Illegal input, returned.\n");
        exit(-1);
    }
    printf("m = %d, n = %d.\n", m, n);
    if (kernel_num == 1) printf("Testing my dgemv.\n");
    else printf("Testing MKL DGEMV.\n");
    double *A, *X, *Y, *Y_ref;
    double t0, t1, elapsed_time;
    A = (double*)malloc(sizeof(double) * m * n);
    X = (double*)malloc(sizeof(double) * n);
    Y = (double*)malloc(sizeof(double) * m);
    Y_ref = (double*)malloc(sizeof(double) * m);
    double alpha = 1., beta = 1.;
    int N = 5;
    randomize_matrix(A, m, n);
    randomize_matrix(X, n, 1);
    randomize_matrix(Y, m, 1);
    copy_matrix(Y, Y_ref, m);

    if (kernel_num == 1){
        printf("Start the sanity check...\n");
        fflush(stdout);
        mydgemv_t(A, X, Y, m, n);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, X, 1, beta, Y_ref, 1);

        if (!verify_matrix(Y, Y_ref, m)){
            printf("did not pass the sanity check, returned.\n");
            exit(-2);
        }else{
            printf("Sanity check passed. Start performance benchmarking...\n");
            fflush(stdout);
        }
    }

    t0 = get_sec();
    if (kernel_num == 1){
        for (int i = 0; i < N; i++){
            mydgemv_t(A, X, Y, m, n);
        }
    }else {
        for (int i = 0; i < N; i++){
            cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, X, 1, beta, Y, 1);
        }
    }
    t1 = get_sec();
    elapsed_time = t1 - t0;
    printf("Average elasped time: %f second, performance: %f GFLOPS.\n", elapsed_time/N,2.*N*1e-9*m*n/elapsed_time);
    free(A);free(X);free(Y);free(Y_ref);
    return 0;
}