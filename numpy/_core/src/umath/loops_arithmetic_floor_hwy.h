/**
 * Header for Highway SIMD floor divide wrapper functions.
 *
 * Declares extern "C" functions callable from C code. The implementations
 * (in loops_arithmetic_floor_hwy.dispatch.cpp) use NumPy's CPU dispatch
 * to compile per-target variants (SVE for ARM; X86_V4/V3/V2 for x86).
 * Each variant internally uses Highway's HWY_STATIC_DISPATCH for per-op
 * sub-target selection.
 *
 * Callers use the plain function names (e.g. npy_highway_floor_divide_s8_contig).
 * The baseline function checks for available targets at runtime via
 * NPY_CPU_DISPATCH_CALL_XB; if a target-specific version is available
 * (e.g. func_name_SVE) it dispatches to it, otherwise falls through to
 * the baseline HWY_STATIC_DISPATCH (typically NEON on ARM64).
 *
 * Supports signed (int8, int16, int32) and unsigned (uint8, uint16, uint32).
 */

#ifndef _NPY_UMATH_LOOPS_ARITHMETIC_FLOOR_HWY_H_
#define _NPY_UMATH_LOOPS_ARITHMETIC_FLOOR_HWY_H_

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Availability checks — non-dispatched, defined in *_avail.c */
NPY_VISIBILITY_HIDDEN int
npy_highway_floor_divide_available(int element_size);

NPY_VISIBILITY_HIDDEN int
npy_highway_floor_divide_unsigned_available(int element_size);

/* Signed integer wrappers (NumPy-dispatched for SVE/ASIMD/NEON/x86 targets) */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s8_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s16_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s32_contig(char **args, npy_intp len);

/* Unsigned integer wrappers */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_u8_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_u16_contig(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_u32_contig(char **args, npy_intp len);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_ARITHMETIC_FLOOR_HWY_H_ */
