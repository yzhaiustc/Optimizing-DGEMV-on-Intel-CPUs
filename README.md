## Hardware and software requirments

* Hardware: Intel Xeon Skylake or Cascade Lake processors.
* Software: Intel icc compiler - updated to be able to compile SIMD intrinsics and AVX512 assembly codes.
* You may need to make the provided shell scripts executable by typing: ```chmod +x xxx.sh``` before running any of them.
* Be sure to set thread number and dispatch AVX-512 instruction from the terminal before testing.
```
export MKL_ENABLE_INSTRUCTIONS=AVX512
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
 ```

## How to reproduce the data

* Modify ***PATH-TO-YOUR-MKL*** and ***PATH-TO-YOUR-OPENBLAS*** in [***Makefile***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/dgemv/Makefile). Then one can just type in ***make*** to compile the code. If the paths are correctly configured, six executable binaries will be generated: ***init_data***, ***verify***, ***ori_blas***, ***ftblas***, ***mkl*** and ***openblas***. ***ori_blas*** is our self-implemented routine without fault-tolerant (FT) capability and ***ftblas*** is our self-implemented routine with FT capability. ***openblas*** and ***mkl*** correspond to OpenBLAS and MKL implementation. ***init_data*** writes all randomly-generated data and parameters into a file so that all the tested BLAS implementations can load from it. ***verify*** validates the correctness of each implementation.
* Run the script [***run.sh***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/dgemv/run.sh) in the same directory by typing in ```./run.sh```. In [***run.sh***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/dgemv/run.sh), we first generate random input and write into a file. All the BLAS routines then load from the same input file and write result matrix/vector into their own output files. Finally, we verify the correctness by running the binary ***verify*** . 
* When benchmark completes and performance data are saved, grab the performance information in the unit of GFLOPS by running the shell script [***data.sh***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/dgemv/data.sh) (```./data.sh```).
* Plot figures using the MATLAB script [***data_blas2.m***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/data_blas2.m). If you want to plot with your own data, you need to move the generated data to the corresponding directories in [***raw_data/***](https://github.com/anonymous-sc-20/routines-to-verify/tree/master/raw_data/dgemv) and overwrite my raw data before running [***data_blas2.m***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/data_blas2.m).
* One more note. The syntax of the binaries to test DGEMV of OpenBLAS/MKL/FT-BLAS:FT and FT-BLAS:Ori are the same: ***./[binary_name] [matrix_row_size] [matrix_col_size]*** .More details can be found in [***run.sh***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/dgemv/run.sh).

## How our codes are organized

We first generate and store random double-precision floating point numbers using the binary ***init_data*** compiled from ***generate_data.c*** into the file system. For the routines containing coefficients as paramemter, we also write these random coefficients into ***matrix_data.txt*** to ensure all the routines for test load the same random data and parameters.

Once input data are generated and ready to load, we can start to test the performance of our self-implemented routines, which have the same interface as OpenBLAS and MKL. When compiling [***test_perf.c***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/dgemv/test_perf.c), we select different BLAS implementations to link using different pre-defined macros sent from [***Makefile***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/dgemv/Makefile). This ensures all the routines are tested under the same setting. To be more specific, there are only 6 lines of codes in the main function of [***test_perf.c***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/dgemv/test_perf.c).
```
int main(int argc, char* argv[])
{
    init(&vec_x, &vec_y, &vec_a, &vec_b, &A, &mn, &m, &n, &inc_x, &inc_y, &alpha, &beta, argc, argv);
    t0 = get_sec();
    cblas_dgemv(CblasRowMajor, CblasTrans, m, n, alpha, A, n, vec_x, inc_x, beta, vec_y, inc_y);
    t1 = get_sec();
    post_processing(&vec_x, &vec_y, &vec_a, &vec_b, &A, t0, t1, m, n, c);
    return 0;
}
```
In the ***init()*** function, we initialize the matrices and arrays with randomly-generated double-precision floating numbers by loading from ***matrix_data.txt*** . We also flush the cache by traversing 256 MB data before calling BLAS routines in the same function. ***get\_sec()*** is our self-wrapped function counting the time in the precision of ***10\^-6*** second. Once the BLAS function finishes, we first stop the timer and then print the elapsed time and calculated performance in the unit of GFLOPS. We finally free the allocated space in the function ***post\_processing()*** . In the same function, we also store the result matrix/vector into the file system to verify the correctness.

Since all four BLAS implementations save the output matrix/array into their own text file, we can now verify the correctness by comparing their output data. This is implemented in [***verify.c***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/dgemv/verify.c).

 ## References
 * Compilation flags of sequential Intel MKL codes are suggested by [Intel MKL Link Advisor](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html) by selecting MKL product version: Intel MKL 2019.0, OS: Linux, compiler: Intel(R) C/C++, architecture: Intel64, dynamic linkage, interface layer: 64-bit integer and threading layer: sequential.
 * Dispatching instructions are suggested by [Intel Document](https://software.intel.com/content/www/us/en/develop/documentation/mkl-macos-developer-guide/top/managing-behavior-of-the-intel-math-kernel-library-with-environment-variables/instruction-set-specific-dispatching-on-intel-architectures.html).
 

## Our data collected on a Skylake processor

Hardware: Intel Xeon Gold 5122 @ 3.6 GHz, Skylake. Memory: 192 GB RAM, DDR4-2666.

Detailed raw data can be referred in [***raw_data/***](https://github.com/anonymous-sc-20/routines-to-verify/tree/master/raw_data/dgemv) . So I just show the figures generated by [***data_blas2.m***](https://github.com/anonymous-sc-20/routines-to-verify/blob/master/data_blas2.m) here.

![image](./dgemv.png)
