/*
 * Copyright (c) 2025, NumPy Developers
 * Distributed under the BSD-3-Clause license
 * See LICENSE.txt for more information
 *
 * Highway SIMD-optimized maximum/minimum for float and double types.
 *
 * Uses NumPy's CPU dispatch system (NPY_CPU_DISPATCH_CURFX) to compile
 * multiple target variants (SVE, ASIMD, NEON for ARM; X86_V4/V3/V2 for x86).
 * Within each variant, Highway's HWY_STATIC_DISPATCH handles sub-target
 * selection at runtime (e.g. SVE vs SVE2, AVX2 vs AVX-512).
 *
 * Key implementation details:
 * 1. NaN propagation follows NumPy semantics: propagate NaN from first operand
 *    - scalar_max(A, B) = ((A >= B || npy_isnan(A)) ? A : B)
 *    - scalar_min(A, B) = ((A <= B || npy_isnan(A)) ? A : B)
 * 2. Highway's hn::Min/hn::Max propagate NaN from second operand (opposite)
 *    - Inline mask operations correct this behavior
 * 3. SVE: Use svorr_b directly instead of Highway's svsel_b-based Or for speed
 * 4. 4x unrolled loop for throughput, scalar tail for remaining elements
 */

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <cmath>
#include <cstdint>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include "npy_cpu_dispatch.h"
#include "loops_minmax_hwy.dispatch.h"

#include <hwy/highway.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#include "loops_minmax_hwy.h"

HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

/*
 * SIMD binary maximum/minimum for integer types.
 * No NaN handling needed - direct max/min operations.
 *
 * Unroll strategy based on element size:
 * - 8-bit:  8x unroll (SVE 256-bit = 32 elements, need more throughput)
 * - 16-bit: 8x unroll (SVE 256-bit = 16 elements)
 * - 32-bit: 4x unroll (SVE 256-bit = 8 elements, balanced)
 * - 64-bit: 2x unroll (SVE 256-bit = 4 elements, avoid overhead)
 */
template <typename T, bool IsMax, int UnrollFactor>
HWY_ATTR static void
simd_minmax_int_unrolled(const T *HWY_RESTRICT src1, const T *HWY_RESTRICT src2,
                         T *HWY_RESTRICT dst, npy_intp len)
{
    const hn::ScalableTag<T> d;
    HWY_LANES_CONSTEXPR npy_intp vstep = static_cast<npy_intp>(hn::Lanes(d));
    const npy_intp vstep_unroll = vstep * UnrollFactor;

    // Main unrolled loop
    for (; len >= vstep_unroll;
         len -= vstep_unroll, src1 += vstep_unroll, src2 += vstep_unroll, dst += vstep_unroll) {
        for (int i = 0; i < UnrollFactor; ++i) {
            auto a = hn::LoadU(d, src1 + i * vstep);
            auto b = hn::LoadU(d, src2 + i * vstep);
            auto r = IsMax ? hn::Max(a, b) : hn::Min(a, b);
            hn::StoreU(r, d, dst + i * vstep);
        }
    }

    // Single vector loop
    for (; len >= vstep; len -= vstep, src1 += vstep, src2 += vstep, dst += vstep) {
        auto a = hn::LoadU(d, src1);
        auto b = hn::LoadU(d, src2);
        auto r = IsMax ? hn::Max(a, b) : hn::Min(a, b);
        hn::StoreU(r, d, dst);
    }

    // Scalar tail
    for (; len > 0; --len, ++src1, ++src2, ++dst) {
        const T a = *src1;
        const T b = *src2;
        *dst = IsMax ? ((a > b) ? a : b) : ((a < b) ? a : b);
    }
}

// Type aliases for integer types
using int8_t = std::int8_t;
using uint8_t = std::uint8_t;
using int16_t = std::int16_t;
using uint16_t = std::uint16_t;
using int32_t = std::int32_t;
using uint32_t = std::uint32_t;
using int64_t = std::int64_t;
using uint64_t = std::uint64_t;

// Convenience wrappers with optimal unroll factors
// 8-bit: 8x unroll for maximum throughput
HWY_ATTR static void simd_maximum_s8(const int8_t *src1, const int8_t *src2,
                                     int8_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<int8_t, true, 8>(src1, src2, dst, len);
}
HWY_ATTR static void simd_minimum_s8(const int8_t *src1, const int8_t *src2,
                                     int8_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<int8_t, false, 8>(src1, src2, dst, len);
}
HWY_ATTR static void simd_maximum_u8(const uint8_t *src1, const uint8_t *src2,
                                     uint8_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<uint8_t, true, 8>(src1, src2, dst, len);
}
HWY_ATTR static void simd_minimum_u8(const uint8_t *src1, const uint8_t *src2,
                                     uint8_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<uint8_t, false, 8>(src1, src2, dst, len);
}

// 16-bit: 8x unroll
HWY_ATTR static void simd_maximum_s16(const int16_t *src1, const int16_t *src2,
                                      int16_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<int16_t, true, 8>(src1, src2, dst, len);
}
HWY_ATTR static void simd_minimum_s16(const int16_t *src1, const int16_t *src2,
                                      int16_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<int16_t, false, 8>(src1, src2, dst, len);
}
HWY_ATTR static void simd_maximum_u16(const uint16_t *src1, const uint16_t *src2,
                                      uint16_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<uint16_t, true, 8>(src1, src2, dst, len);
}
HWY_ATTR static void simd_minimum_u16(const uint16_t *src1, const uint16_t *src2,
                                      uint16_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<uint16_t, false, 8>(src1, src2, dst, len);
}

// 32-bit: 4x unroll (balanced)
HWY_ATTR static void simd_maximum_s32(const int32_t *src1, const int32_t *src2,
                                      int32_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<int32_t, true, 4>(src1, src2, dst, len);
}
HWY_ATTR static void simd_minimum_s32(const int32_t *src1, const int32_t *src2,
                                      int32_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<int32_t, false, 4>(src1, src2, dst, len);
}
HWY_ATTR static void simd_maximum_u32(const uint32_t *src1, const uint32_t *src2,
                                      uint32_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<uint32_t, true, 4>(src1, src2, dst, len);
}
HWY_ATTR static void simd_minimum_u32(const uint32_t *src1, const uint32_t *src2,
                                      uint32_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<uint32_t, false, 4>(src1, src2, dst, len);
}

// 64-bit: 2x unroll (reduce overhead for smaller vector width)
HWY_ATTR static void simd_maximum_s64(const int64_t *src1, const int64_t *src2,
                                      int64_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<int64_t, true, 2>(src1, src2, dst, len);
}
HWY_ATTR static void simd_minimum_s64(const int64_t *src1, const int64_t *src2,
                                      int64_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<int64_t, false, 2>(src1, src2, dst, len);
}
HWY_ATTR static void simd_maximum_u64(const uint64_t *src1, const uint64_t *src2,
                                      uint64_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<uint64_t, true, 2>(src1, src2, dst, len);
}
HWY_ATTR static void simd_minimum_u64(const uint64_t *src1, const uint64_t *src2,
                                      uint64_t *dst, npy_intp len) {
    simd_minmax_int_unrolled<uint64_t, false, 2>(src1, src2, dst, len);
}

} // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

/*
 * C wrapper functions for dispatch.
 */
extern "C" {

/*
 * Baseline compilation: plain func_name is the caller-facing entry point.
 * Dispatches at runtime to the best available target (e.g. func_name_SVE)
 * via NPY_CPU_DISPATCH_CALL_XB. If no target is available, falls through
 * to HWY_STATIC_DISPATCH (typically NEON on ARM64).
 */
typedef void (*minmax_func)(char **args, npy_intp len);

/* Forward-declare target-specific variants so NPY_CPU_DISPATCH_CALL_XB can
 * reference them. FP types are NOT declared - baseline-only, bypass SVE. */
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_maximum_s8_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_minimum_s8_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_maximum_u8_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_minimum_u8_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_maximum_s16_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_minimum_s16_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_maximum_u16_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_minimum_u16_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_maximum_s32_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_minimum_s32_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_maximum_u32_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_minimum_u32_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_maximum_s64_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_minimum_s64_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_maximum_u64_contig,
                            (char **args, npy_intp len));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_minimum_u64_contig,
                            (char **args, npy_intp len));

#ifdef NPY_MTARGETS_CURRENT
/*
 * Target-specific compilation (e.g. SVE).
 * NPY_CPU_DISPATCH_CURFX(func_name) produces e.g. func_name_SVE.
 * Uses HWY_STATIC_DISPATCH for Highway's own sub-target selection
 * (e.g. SVE vs SVE2).  No NPY_CPU_DISPATCH_CALL_XB here to avoid
 * recursive dispatch on Kunpeng/ARM SVE targets.
 */
#define MINMAX_DISPATCH(func_name, simd_func, scalar_type) \
NPY_VISIBILITY_HIDDEN void \
NPY_CPU_DISPATCH_CURFX(func_name)(char **args, npy_intp len) \
{ \
    HWY_STATIC_DISPATCH(simd_func)( \
        reinterpret_cast<const scalar_type *>(args[0]), \
        reinterpret_cast<const scalar_type *>(args[1]), \
        reinterpret_cast<scalar_type *>(args[2]), len); \
}
#else
/*
 * Baseline compilation: plain func_name is the caller-facing entry point.
 * Dispatches at runtime to the best available target (e.g. func_name_SVE)
 * via NPY_CPU_DISPATCH_CALL_XB. If no target is available, falls through
 * to HWY_STATIC_DISPATCH (typically NEON on ARM64).
 */
/* Dispatch macro for integer types: runtime dispatch to best target. */
#define MINMAX_DISPATCH(func_name, simd_func, scalar_type) \
NPY_VISIBILITY_HIDDEN void \
NPY_CPU_DISPATCH_CURFX(func_name)(char **args, npy_intp len) \
{ \
    minmax_func _f = NULL; \
    NPY_CPU_DISPATCH_CALL_XB(_f = func_name); \
    if (_f != NULL) { \
        _f(args, len); \
        return; \
    } \
    HWY_STATIC_DISPATCH(simd_func)( \
        reinterpret_cast<const scalar_type *>(args[0]), \
        reinterpret_cast<const scalar_type *>(args[1]), \
        reinterpret_cast<scalar_type *>(args[2]), len); \
}
#endif

/* 8-bit to 64-bit integer maximum/minimum wrappers - SVE dispatched */
MINMAX_DISPATCH(npy_highway_maximum_s8_contig, simd_maximum_s8, std::int8_t)
MINMAX_DISPATCH(npy_highway_minimum_s8_contig, simd_minimum_s8, std::int8_t)
MINMAX_DISPATCH(npy_highway_maximum_u8_contig, simd_maximum_u8, std::uint8_t)
MINMAX_DISPATCH(npy_highway_minimum_u8_contig, simd_minimum_u8, std::uint8_t)

MINMAX_DISPATCH(npy_highway_maximum_s16_contig, simd_maximum_s16, std::int16_t)
MINMAX_DISPATCH(npy_highway_minimum_s16_contig, simd_minimum_s16, std::int16_t)
MINMAX_DISPATCH(npy_highway_maximum_u16_contig, simd_maximum_u16, std::uint16_t)
MINMAX_DISPATCH(npy_highway_minimum_u16_contig, simd_minimum_u16, std::uint16_t)

MINMAX_DISPATCH(npy_highway_maximum_s32_contig, simd_maximum_s32, std::int32_t)
MINMAX_DISPATCH(npy_highway_minimum_s32_contig, simd_minimum_s32, std::int32_t)
MINMAX_DISPATCH(npy_highway_maximum_u32_contig, simd_maximum_u32, std::uint32_t)
MINMAX_DISPATCH(npy_highway_minimum_u32_contig, simd_minimum_u32, std::uint32_t)

MINMAX_DISPATCH(npy_highway_maximum_s64_contig, simd_maximum_s64, std::int64_t)
MINMAX_DISPATCH(npy_highway_minimum_s64_contig, simd_minimum_s64, std::int64_t)
MINMAX_DISPATCH(npy_highway_maximum_u64_contig, simd_maximum_u64, std::uint64_t)
MINMAX_DISPATCH(npy_highway_minimum_u64_contig, simd_minimum_u64, std::uint64_t)

#undef MINMAX_DISPATCH
#undef MINMAX_DISPATCH_FP

} // extern "C"