/*
 * Copyright (c) 2025, NumPy Developers
 * Distributed under the BSD-3-Clause license
 * See LICENSE.txt for more information
 *
 * Availability check functions for Highway SIMD min/max.
 *
 * Non-dispatched source file that provides runtime checks for whether
 * the Highway SIMD implementation is available for a given element size.
 */

#include "loops_minmax_hwy.h"

/*
 * Check if Highway SIMD min/max is available for the given element size.
 *
 * Parameters:
 *   element_size: Size of the element type in bytes (2, 4, or 8 for int types,
 *                 4 for float, 8 for double)
 *
 * Returns:
 *   1 if Highway SIMD is available and optimized for this element size
 *   0 otherwise
 */
int
npy_highway_minmax_available(int element_size)
{
#ifdef NPY_HAVE_HIGHWAY
    /* Highway supports all common element sizes: 1, 2, 4, 8 bytes */
    return (element_size >= 1 && element_size <= 8) ? 1 : 0;
#else
    /* Highway not available in this build */
    (void)element_size;
    return 0;
#endif
}