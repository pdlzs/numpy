/*
 * Copyright (c) 2025, NumPy Developers
 * Distributed under the BSD-3-Clause license
 * See LICENSE.txt for more information
 *
 * Header for Highway SIMD maximum/minimum wrapper functions.
 *
 * Declares extern "C" functions callable from C code. The implementations
 * (in loops_minmax_hwy.dispatch.cpp) use NumPy's CPU dispatch to compile
 * per-target variants (SVE, ASIMD, NEON for ARM; X86_V4/V3/V2 for x86).
 *
 * Within each variant, Highway's HWY_STATIC_DISPATCH handles sub-target
 * selection at runtime (e.g. SVE vs SVE2, AVX2 vs AVX-512).
 *
 * Baseline (plain name, e.g. npy_highway_maximum_f64_contig) acts as a
 * trampoline: it uses NPY_CPU_DISPATCH_CALL_XB to dispatch to the best
 * available target-specific variant at runtime.
 *
 * Supports float (f32) and double (f64) with proper NaN propagation
 * following NumPy semantics.
 */

#ifndef _NPY_UMATH_LOOPS_MINMAX_HWY_H_
#define _NPY_UMATH_LOOPS_MINMAX_HWY_H_

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Availability check — returns 1 if Highway SIMD is available for the
 * given element size (4 for float, 8 for double).
 */
NPY_VISIBILITY_HIDDEN int
npy_highway_minmax_available(int element_size);

/*
 * Float maximum/minimum wrappers (NumPy-dispatched for SVE/ASIMD/NEON/x86)
 * These functions handle contiguous arrays and preserve NaN propagation
 * semantics: NaN in the first operand is propagated.
 */
NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_f32_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_f64_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_f32_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_f64_contig(char **args, npy_intp len);

/*
 * Integer maximum/minimum wrappers (SVE-vectorized for ARM platforms)
 * No NaN handling needed - simple max/min operations.
 * Supports 8-bit, 16-bit, 32-bit, and 64-bit signed/unsigned integers.
 */
NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_s8_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_u8_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_s16_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_u16_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_s32_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_u32_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_s64_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_maximum_u64_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_s8_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_u8_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_s16_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_u16_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_s32_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_u32_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_s64_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_minimum_u64_contig(char **args, npy_intp len);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_MINMAX_HWY_H_ */