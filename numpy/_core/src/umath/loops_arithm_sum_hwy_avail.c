/**
 * Availability check functions for Highway SIMD pairwise sum.
 *
 * These are compiled once (non-dispatched) as part of the _multiarray_umath
 * module and return whether Highway SIMD is available for a given element
 * size. When NPY_HAVE_HIGHWAY is not defined (Highway disabled or
 * unavailable), always returns 0 so the .c.src template falls through to
 * scalar paths.
 */

#include "loops_arithm_sum_hwy.h"

int
npy_highway_pairwise_sum_available(int element_size)
{
#ifdef NPY_HAVE_HIGHWAY
    /* Highway supports 32-bit and 64-bit floating-point types */
    return (element_size == 4 || element_size == 8) ? 1 : 0;
#else
    (void)element_size;
    return 0;
#endif
}