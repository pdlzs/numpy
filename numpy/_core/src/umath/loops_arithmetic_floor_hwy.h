/**
 * Header for Highway SIMD floor divide wrapper functions.
 *
 * Declares extern "C" functions callable from C code that use Highway
 * SIMD for floor division when available, falling back to scalar otherwise.
 */

#ifndef _NPY_UMATH_LOOPS_ARITHMETIC_FLOOR_HWY_H_
#define _NPY_UMATH_LOOPS_ARITHMETIC_FLOOR_HWY_H_

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Check if Highway SIMD floor divide is available for the given element size.
 *
 * @param element_size Size of the integer type in bytes (1, 2, or 4)
 * @return 1 if Highway can provide SIMD implementation, 0 otherwise
 */
NPY_VISIBILITY_HIDDEN int
npy_highway_floor_divide_available(int element_size);

/**
 * Execute floor divide for 8-bit signed integers.
 * args[0] = numerator array, args[1] = divisor array, args[2] = result array
 *
 * @param args Array of pointers to input/output arrays
 * @param len Number of elements to process
 */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s8_contig(char **args, npy_intp len);

/**
 * Execute floor divide for 16-bit signed integers.
 */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s16_contig(char **args, npy_intp len);

/**
 * Execute floor divide for 32-bit signed integers.
 */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s32_contig(char **args, npy_intp len);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_ARITHMETIC_FLOOR_HWY_H_ */