/**
 * Highway SIMD-optimized floor divide for signed integer types.
 *
 * This file provides extern "C" wrapper functions that can be called from
 * the C template (loops_arithmetic.dispatch.c.src) when native SIMD integer
 * division (like VSX4's vec_div) is not available.
 *
 * Highway's hn::Div implements the Granlund-Montgomery algorithm for
 * division by invariant integers, adapted for element-wise division.
 */

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <limits>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include "simd/simd.hpp"
#include <hwy/highway.h>

namespace np::highway::floor_div {

using namespace np::simd;

#if NPY_HWY

/**
 * SIMD floor divide kernel for contiguous signed integer arrays.
 *
 * Algorithm:
 * 1. Detect exceptions (div-by-zero, MIN/-1 overflow) before division
 * 2. Replace problematic divisors with 1 to avoid hardware exceptions
 * 3. Perform SIMD division using Highway's hn::Div
 * 4. Apply floor adjustment when signs differ and remainder != 0
 * 5. Apply exception results (div-by-zero -> 0, overflow -> MIN)
 *
 * Accumulates exception flags and sets float status once at end for efficiency.
 */
template <typename T>
HWY_ATTR static void
simd_floor_divide(const T *src1, const T *src2, T *dst, npy_intp len)
{
    constexpr T min_val = std::numeric_limits<T>::min();
    HWY_LANES_CONSTEXPR int vstep = static_cast<int>(Lanes<T>());

    const auto vneg_one = Set<T>(T(-1));
    const auto vzero = Zero<T>();
    const auto vmin = Set<T>(min_val);

    // Track exceptions
    bool warn_zero = false;
    bool warn_overflow = false;

    // Main SIMD loop
    for (; len >= vstep; len -= vstep, src1 += vstep, src2 += vstep, dst += vstep) {
        auto a = LoadU(src1);
        auto b = LoadU(src2);

        // Detect exceptions
        auto bzero = hn::Eq(b, vzero);
        auto amin = hn::Eq(a, vmin);
        auto bneg_one = hn::Eq(b, vneg_one);
        auto overflow = hn::And(amin, bneg_one);

        // Track div-by-zero using bool (avoid mask accumulation issues)
        if (!hn::AllFalse(_Tag<T>(), bzero)) {
            warn_zero = true;
        }
        // Track overflow (MIN / -1)
        if (!hn::AllFalse(_Tag<T>(), overflow)) {
            warn_overflow = true;
        }

        // Safe division: replace problematic divisors with 1
        auto safe_b = hn::IfThenElse(hn::Or(bzero, overflow), Set<T>(T(1)), b);
        auto quo = hn::Div(a, safe_b);

        // Floor adjustment: if signs differ and remainder != 0, subtract 1
        auto rem = hn::Sub(a, hn::Mul(quo, safe_b));
        auto a_pos = hn::Gt(a, vzero);
        auto b_pos = hn::Gt(b, vzero);
        auto same_sign = hn::Not(hn::Xor(a_pos, b_pos));
        auto rem_zero = hn::Eq(rem, vzero);
        auto needs_adj = hn::And(hn::Not(same_sign), hn::Not(rem_zero));
        quo = hn::Add(quo, hn::IfThenElse(needs_adj, vneg_one, vzero));

        // Apply exception results
        quo = hn::IfThenElse(bzero, vzero, quo);
        quo = hn::IfThenElse(overflow, vmin, quo);

        StoreU(quo, dst);
    }

    // Set exception status
    if (warn_zero) {
        npy_set_floatstatus_divbyzero();
    }
    if (warn_overflow) {
        npy_set_floatstatus_overflow();
    }

    // Scalar tail handles all overflow detection
    for (; len > 0; --len, ++src1, ++src2, ++dst) {
        const T a = *src1;
        const T b = *src2;
        if (NPY_UNLIKELY(b == 0)) {
            npy_set_floatstatus_divbyzero();
            *dst = 0;
        }
        else if (NPY_UNLIKELY(a == min_val && b == -1)) {
            npy_set_floatstatus_overflow();
            *dst = min_val;
        }
        else {
            T r = a / b;
            if (((a > 0) != (b > 0)) && ((r * b) != a)) {
                r--;
            }
            *dst = r;
        }
    }
}

#endif // NPY_HWY

} // namespace np::highway::floor_div

/*
 * Extern "C" wrapper functions callable from C template code.
 * These check NPY_HWY internally and fall back to scalar if Highway unavailable.
 */

extern "C" {

/*
 * Check if Highway SIMD floor divide is available for the given element size.
 * Returns 1 if Highway can provide SIMD implementation, 0 otherwise.
 */
NPY_VISIBILITY_HIDDEN int
npy_highway_floor_divide_available(int element_size)
{
#if NPY_HWY
    /* Highway supports 8, 16, 32-bit signed integers */
    return (element_size == 1 || element_size == 2 || element_size == 4) ? 1 : 0;
#else
    return 0;
#endif
}

/*
 * Execute Highway SIMD floor divide for 8-bit signed integers.
 * args[0] = numerator array, args[1] = divisor array, args[2] = result array
 */
NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s8_contig(char **args, npy_intp len)
{
#if NPY_HWY
    np::highway::floor_div::simd_floor_divide<int8_t>(
        reinterpret_cast<const int8_t *>(args[0]),
        reinterpret_cast<const int8_t *>(args[1]),
        reinterpret_cast<int8_t *>(args[2]),
        len);
#else
    /* Scalar fallback when Highway unavailable */
    const npy_int8 *src1 = (const npy_int8 *)args[0];
    const npy_int8 *src2 = (const npy_int8 *)args[1];
    npy_int8 *dst = (npy_int8 *)args[2];
    for (npy_intp i = 0; i < len; ++i) {
        const npy_int8 a = src1[i];
        const npy_int8 b = src2[i];
        if (NPY_UNLIKELY(b == 0)) {
            npy_set_floatstatus_divbyzero();
            dst[i] = 0;
        }
        else if (NPY_UNLIKELY(a == NPY_MIN_INT8 && b == -1)) {
            npy_set_floatstatus_overflow();
            dst[i] = NPY_MIN_INT8;
        }
        else {
            npy_int8 r = a / b;
            if (((a > 0) != (b > 0)) && ((r * b) != a)) {
                r--;
            }
            dst[i] = r;
        }
    }
#endif
}

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s16_contig(char **args, npy_intp len)
{
#if NPY_HWY
    np::highway::floor_div::simd_floor_divide<int16_t>(
        reinterpret_cast<const int16_t *>(args[0]),
        reinterpret_cast<const int16_t *>(args[1]),
        reinterpret_cast<int16_t *>(args[2]),
        len);
#else
    const npy_int16 *src1 = (const npy_int16 *)args[0];
    const npy_int16 *src2 = (const npy_int16 *)args[1];
    npy_int16 *dst = (npy_int16 *)args[2];
    for (npy_intp i = 0; i < len; ++i) {
        const npy_int16 a = src1[i];
        const npy_int16 b = src2[i];
        if (NPY_UNLIKELY(b == 0)) {
            npy_set_floatstatus_divbyzero();
            dst[i] = 0;
        }
        else if (NPY_UNLIKELY(a == NPY_MIN_INT16 && b == -1)) {
            npy_set_floatstatus_overflow();
            dst[i] = NPY_MIN_INT16;
        }
        else {
            npy_int16 r = a / b;
            if (((a > 0) != (b > 0)) && ((r * b) != a)) {
                r--;
            }
            dst[i] = r;
        }
    }
#endif
}

NPY_VISIBILITY_HIDDEN void
npy_highway_floor_divide_s32_contig(char **args, npy_intp len)
{
#if NPY_HWY
    np::highway::floor_div::simd_floor_divide<int32_t>(
        reinterpret_cast<const int32_t *>(args[0]),
        reinterpret_cast<const int32_t *>(args[1]),
        reinterpret_cast<int32_t *>(args[2]),
        len);
#else
    const npy_int32 *src1 = (const npy_int32 *)args[0];
    const npy_int32 *src2 = (const npy_int32 *)args[1];
    npy_int32 *dst = (npy_int32 *)args[2];
    for (npy_intp i = 0; i < len; ++i) {
        const npy_int32 a = src1[i];
        const npy_int32 b = src2[i];
        if (NPY_UNLIKELY(b == 0)) {
            npy_set_floatstatus_divbyzero();
            dst[i] = 0;
        }
        else if (NPY_UNLIKELY(a == NPY_MIN_INT32 && b == -1)) {
            npy_set_floatstatus_overflow();
            dst[i] = NPY_MIN_INT32;
        }
        else {
            npy_int32 r = a / b;
            if (((a > 0) != (b > 0)) && ((r * b) != a)) {
                r--;
            }
            dst[i] = r;
        }
    }
#endif
}

} // extern "C"