#include "utils.h"
#include "immintrin.h"
#include "omp.h"
#define A(i, j) A[(i)*lda+(j)]//row-major

#define INIT_n1(no)\
	"vxorps %%zmm"#no",%%zmm"#no",%%zmm"#no";"
#define INIT_n4\
	INIT_n1(4)\
	INIT_n1(5)\
	INIT_n1(6)\
	INIT_n1(7)
#define LOAD_X\
	"vmovups (%%r14,%%r15,8), %%zmm12;"
#define FMA(a, b)\
	"vfmadd231pd (%%"#a", %%r15, 8), %%zmm12,%%zmm"#b";"
#define FMA_4(a1, b1, a2, b2, a3, b3, a4, b4)\
	FMA(a1, b1)\
    FMA(a2, b2)\
    FMA(a3, b3)\
    FMA(a4, b4)
#define EXTRACT_0(vr1,vr2)\
	"vextractf64x4 $0, %%zmm"#vr1", %%ymm"#vr2";"
#define EXTRACT_0_4(a,b,c,d,e,f,g,h)\
	EXTRACT_0(a,b)\
    EXTRACT_0(c,d)\
    EXTRACT_0(e,f)\
    EXTRACT_0(g,h)
#define EXTRACT_1(vr1,vr2)\
	"vextractf64x4 $1, %%zmm"#vr1", %%ymm"#vr2";"
#define EXTRACT_1_4(a,b,c,d,e,f,g,h)\
	EXTRACT_1(a,b)\
    EXTRACT_1(c,d)\
    EXTRACT_1(e,f)\
    EXTRACT_1(g,h)
#define EXTRACT128(vr1,vr2)\
	"vextractf128 $1, %%ymm"#vr1", %%xmm"#vr2";"
#define EXTRACT_128_4(a,b,c,d,e,f,g,h)\
	EXTRACT128(a,b)\
    EXTRACT128(c,d)\
    EXTRACT128(e,f)\
    EXTRACT128(g,h)
#define VADD_1(vr1,vr2,vr3)\
    "vaddpd %%"#vr1", %%"#vr2", %%"#vr3";"
#define VADD_4(vr11,vr12,vr13,vr21,vr22,vr23,vr31,vr32,vr33,vr41,vr42,vr43)\
    VADD_1(vr11,vr12,vr13)\
    VADD_1(vr21,vr22,vr23)\
    VADD_1(vr31,vr32,vr33)\
    VADD_1(vr41,vr42,vr43)
#define VHADD_1(vr)\
    "vhaddpd %%xmm"#vr", %%xmm"#vr", %%xmm"#vr";"
#define VHADD_4(vr1,vr2,vr3,vr4)\
    VHADD_1(vr1)\
    VHADD_1(vr2)\
    VHADD_1(vr3)\
    VHADD_1(vr4)
#define VSTORE_1(vr, offset)\
    "vmovsd %%xmm"#vr", "#offset"(%2);"
#define VSTORE_4(vr1, offset1, vr2, offset2, vr3, offset3, vr4, offset4)\
    VSTORE_1(vr1,offset1)\
    VSTORE_1(vr2,offset2)\
    VSTORE_1(vr3,offset3)\
    VSTORE_1(vr4,offset4)
#define DGEMV_KERNEL {\
	__asm__ __volatile__(\
        INIT_n4 "xorq %%r15,%%r15;"\
        "1:\n\t"\
        "vmovups (%1,%%r15,8), %%zmm12;"\
        "vfmadd231pd (%3,%%r15,8), %%zmm12, %%zmm4;"\
        "vfmadd231pd (%4,%%r15,8), %%zmm12, %%zmm5;"\
        "vfmadd231pd (%5,%%r15,8), %%zmm12, %%zmm6;"\
        "vfmadd231pd (%6,%%r15,8), %%zmm12, %%zmm7;"\
        "addq $8, %%r15;subq $8, %0;jnz 1b;"\
        EXTRACT_0_4(4,16,5,17,6,18,7,19)\
        EXTRACT_1_4(4,4,5,5,6,6,7,7)\
        VADD_4(ymm4,ymm16,ymm4,ymm5,ymm17,ymm5,ymm6,ymm18,ymm6,ymm7,ymm19,ymm7)\
        EXTRACT_128_4(4,12,5,13,6,14,7,15)\
        VADD_4(xmm4,xmm12,xmm4,xmm5,xmm13,xmm5,xmm6,xmm14,xmm6,xmm7,xmm15,xmm7)\
        VHADD_4(4,5,6,7)\
        VSTORE_4(4,0,5,8,6,16,7,24)\
        "vzeroupper			 \n\t"\
        :"+r"(n8)\
        :"r"(x),"r"(y_buffer),"r"(a1),"r"(a2),"r"(a3),"r"(a4)\
        :"r15",\
          "cc","memory","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5",\
          "zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12",\
          "zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19",\
          "zmm20","zmm21","zmm22","zmm23","zmm24","zmm25",\
          "zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
}


void mydgemv_compute(double *A, double *x, double *y, long int m, long int n){
    int lda = n;//row-major
    int m4 = m & -4;
    int n4 = n & -4;
    long int n8 = n & -8;
    int i, j;
    double y_buffer[4] = {0.};
    double r_reg[4];
    double a_reg[4];
    double *A_ptr;
    double *a1, *a2, *a3, *a4;
    double xj;
    for (i = 0; i < m4; i += 4){
        A_ptr = A + i*lda;
        _mm256_storeu_pd(r_reg, _mm256_loadu_pd(y+i));
        n8 = n & -8;
        if (n8){
            a1 = A_ptr; A_ptr += lda;
            a2 = A_ptr; A_ptr += lda;
            a3 = A_ptr; A_ptr += lda;
            a4 = A_ptr; A_ptr += lda;
            DGEMV_KERNEL
            _mm256_storeu_pd(r_reg, _mm256_add_pd(_mm256_loadu_pd(r_reg), _mm256_loadu_pd(y_buffer)) );
        }

        j = (n & -8);
        A_ptr = &A(i, j);
        while (j < n){
            xj = x[j];
            r_reg[0] += *A_ptr * xj; A_ptr+=lda;
            r_reg[1] += *A_ptr * xj; A_ptr+=lda;
            r_reg[2] += *A_ptr * xj; A_ptr+=lda;
            r_reg[3] += *A_ptr * xj;
            j++; A_ptr += (1 - lda*3);
        }
        _mm256_storeu_pd(y+i, _mm256_loadu_pd(r_reg));
    }

    while (i < m)//naive version
    {
        for (j = 0; j < n; j++)
            y[i] += A[i * lda + j] * x[j];
        i++;
    }

}

void mydgemv_t(double *A, double *x, double *y, int m, int n)
{
    int tid;
#pragma omp parallel for schedule(static)
    for (tid = 0; tid < omp_get_num_threads(); tid++)
    {
        int TOTAL_THREADS = omp_get_num_threads();
        long int NUM_DIV_NUM_THREADS = m / TOTAL_THREADS * TOTAL_THREADS;
        long int DIM_LEN = m / TOTAL_THREADS;
        long int EDGE_LEN = (NUM_DIV_NUM_THREADS == m) ? m / TOTAL_THREADS : m - NUM_DIV_NUM_THREADS + DIM_LEN;
        if (tid == 0)
            mydgemv_compute(A,x,y,EDGE_LEN,n);
        else
            mydgemv_compute(A+EDGE_LEN*n + (tid - 1) * DIM_LEN*n,x,y + EDGE_LEN + (tid - 1) * DIM_LEN,DIM_LEN,n);
    }
    return;
}