#include <stdio.h>
#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <stdbool.h>

void print_vector(double *vec, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        //if (i % 5 == 0) printf("\n");
        printf(" %5.2f, ", vec[i]);
    }
    printf("\n");
}

void print_matrix(const double *A, int m, int n)
{
    int i;
    printf("[");
    for (i = 0; i < m * n; i++)
    {

        if ((i + 1) % n == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % n == 0)
        {
            if (i + 1 < m * n)
                printf(";\n");
        }
    }
    printf("]\n");
}

double get_sec()
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec + 1e-6 * time.tv_usec);
}

void randomize_matrix(double *A, int m, int n)
{
    srand(time(NULL));
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i * n + j] = (double)(rand() % 100) + 0.01 * (rand() % 100);
            A[i * n + j] = i * n + j;
            if (rand() % 2 == 1)
            {
                //A[i * n + j] *= -1.0;
            }
        }
    }
}

void copy_matrix(double *src, double *dest, int n)
{
    int i;
    for (i = 0; src + i && dest + i && i < n; i++)
    {
        *(dest + i) = *(src + i);
    }
    if (i != n)
    {
        printf("copy failed at %d while there are %d elements in total.\n", i, n);
    }
}

bool verify_matrix(double *mat1, double *mat2, int n)
{
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < n; i++)
    {
        diff = fabs(mat1[i] - mat2[i]);
        if (diff > 1e-6) break;
    }
    if (i != n) return false;
    else return true;
}