/**
 * Header for Highway SIMD pairwise sum wrapper functions.
 *
 * Declares extern "C" functions callable from C code. The implementations
 * (in loops_arithm_sum_hwy.dispatch.cpp) use NumPy's CPU dispatch to
 * compile per-target variants (SVE for ARM; X86_V4/V3/V2 for x86).
 * Each variant internally uses Highway's HWY_STATIC_DISPATCH for
 * sub-target selection.
 *
 * Callers use the plain function names (e.g. npy_highway_pairwise_sum_f32).
 * Highway automatically selects the best available instruction set at
 * runtime (e.g. NEON on baseline ARM64, SSE2/AVX on baseline x86).
 *
 * Supports float and double for real types, and complex float/double.
 */

#ifndef _NPY_UMATH_LOOPS_ARITHM_SUM_HWY_H_
#define _NPY_UMATH_LOOPS_ARITHM_SUM_HWY_H_

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Availability check - non-dispatched, defined in loops_arithm_sum_hwy_avail.c
 * Returns 1 if Highway SIMD is available for the given element size (4 or 8).
 */
NPY_VISIBILITY_HIDDEN int
npy_highway_pairwise_sum_available(int element_size);

/*
 * Real floating-point pairwise sum functions.
 * These compute the sum of a contiguous array using Highway SIMD,
 * employing pairwise recursion for numerical stability.
 */
NPY_VISIBILITY_HIDDEN float
npy_highway_pairwise_sum_f32_contig(const float *a, npy_intp n);

NPY_VISIBILITY_HIDDEN double
npy_highway_pairwise_sum_f64_contig(const double *a, npy_intp n);

/*
 * Complex floating-point pairwise sum functions.
 * rr, ri are output pointers for real and imaginary parts.
 * n is the total number of floating-point elements (2 * num_complex).
 */
NPY_VISIBILITY_HIDDEN void
npy_highway_complex_pairwise_sum_f32_contig(float *rr, float *ri,
                                             const float *a, npy_intp n);

NPY_VISIBILITY_HIDDEN void
npy_highway_complex_pairwise_sum_f64_contig(double *rr, double *ri,
                                             const double *a, npy_intp n);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_ARITHM_SUM_HWY_H_ */