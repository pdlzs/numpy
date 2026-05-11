/*
 * Copyright (c) 2025, NumPy Developers
 * Distributed under the BSD-3-Clause license
 * See LICENSE.txt for more information
 */

#include "common.hpp"

#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>

#include <hwy/highway.h>
#include <hwy/cache_control.h>
#include <cstring>

HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#if defined(__GNUC__) || defined(__clang__)
#define BRANCH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define BRANCH_UNLIKELY(x) (x)
#endif

constexpr size_t BOOL_BLOCK_SIZE_BYTES = 512;

int ComputeArgMinBool(const uint8_t* HWY_RESTRICT arr, npy_intp len)
{
    if (len <= 0) return 0;

    hn::ScalableTag<uint8_t> d;
    using V = hn::Vec<decltype(d)>;

    const size_t N = hn::Lanes(d);
    const size_t unroll_N = 4 * N;

    const size_t block_size = std::max<size_t>(unroll_N, (BOOL_BLOCK_SIZE_BYTES / unroll_N) * unroll_N);

    int base = 0;
    V v_zero = hn::Zero(d);

    for (; base <= len - (int)block_size; base += block_size) {
        V acc = hn::Set(d, hwy::HighestValue<uint8_t>());
        for (size_t i = 0; i < block_size; i += unroll_N) {
            V v0 = hn::LoadU(d, arr + base + i);
            V v1 = hn::LoadU(d, arr + base + i + N);
            V v2 = hn::LoadU(d, arr + base + i + 2 * N);
            V v3 = hn::LoadU(d, arr + base + i + 3 * N);

            // Fold 4 vectors into 1 (3 pure UMIN instructions)
            V v_min01 = hn::Min(v0, v1);
            V v_min23 = hn::Min(v2, v3);
            V v_min_all = hn::Min(v_min01, v_min23);

            // Accumulator update
            acc = hn::Min(acc, v_min_all);
        }

        auto mask_is_zero = hn::Eq(acc, v_zero);

        // If any element in mask is True, accumulator captured a zero
        if (BRANCH_UNLIKELY(!hn::AllFalse(d, mask_is_zero))) {
            // Hit! Use scalar loop to precisely locate first zero in L1 cache
            for (size_t i = 0; i < block_size; ++i) {
                if (arr[base + i] == 0) {
                    return base + i;
                }
            }
        }
    }

    for (; base <= len - (int)N; base += N) {
        V v = hn::LoadU(d, arr + base);
        auto mask_is_zero = hn::Eq(v, v_zero);
        if (BRANCH_UNLIKELY(!hn::AllFalse(d, mask_is_zero))) {
            for (size_t i = 0; i < N; ++i) {
                if (arr[base + i] == 0) return base + i;
            }
        }
    }

    for (; base < len; ++base) {
        if (arr[base] == 0) {
            return base;
        }
    }

    return 0;
}

// Find the index of the first non-zero element (True). Returns 0 if all False.
int ComputeArgMaxBool(const uint8_t* HWY_RESTRICT arr, npy_intp len)
{
    if (len <= 0) return 0;

    hn::ScalableTag<uint8_t> d;
    using V = hn::Vec<decltype(d)>;

    const size_t N = hn::Lanes(d);
    const size_t unroll_N = 4 * N;

    const size_t block_size = std::max<size_t>(unroll_N, (BOOL_BLOCK_SIZE_BYTES / unroll_N) * unroll_N);

    int base = 0;
    V v_zero = hn::Zero(d);

    for (; base <= len - (int)block_size; base += block_size) {
        V acc = v_zero;
        for (size_t i = 0; i < block_size; i += unroll_N) {

            V v0 = hn::LoadU(d, arr + base + i);
            V v1 = hn::LoadU(d, arr + base + i + N);
            V v2 = hn::LoadU(d, arr + base + i + 2 * N);
            V v3 = hn::LoadU(d, arr + base + i + 3 * N);

            V v_or01 = hn::Or(v0, v1);
            V v_or23 = hn::Or(v2, v3);
            V v_or_all = hn::Or(v_or01, v_or23);

            // As long as any byte is non-zero, corresponding acc byte stays non-zero permanently
            acc = hn::Or(acc, v_or_all);
        }

        // After block ends, perform single horizontal check
        if (BRANCH_UNLIKELY(!hn::AllTrue(d, hn::Eq(acc, v_zero)))) {
            // Hit! Use scalar loop to precisely locate first True in L1 cache
            for (size_t i = 0; i < block_size; ++i) {
                if (arr[base + i]) {
                    return base + i;
                }
            }
        }
    }

    for (; base <= len - (int)N; base += N) {
        V v = hn::LoadU(d, arr + base);
        if (BRANCH_UNLIKELY(!hn::AllTrue(d, hn::Eq(v, v_zero)))) {
            for (size_t i = 0; i < N; ++i) {
                if (arr[base + i]) return base + i;
            }
        }
    }

    for (; base < len; ++base) {
        if (arr[base]) {
            return base;
        }
    }
    return 0;
}

template <typename T, bool IsMax>
int ComputeArgMinMaxFloating(const T* HWY_RESTRICT arr, npy_intp len)
{
    if (len <= 0) return -1;

    hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    using M = hn::Mask<decltype(d)>;

    const size_t N = hn::Lanes(d);

    // ARM NEON: 4-way unroll to reduce register pressure (32x128-bit regs)
    // x86 AVX-512 / SVE: 8-way unroll for better OoO execution
#if (HWY_TARGET & HWY_NEON) || (HWY_TARGET & HWY_NEON_WITHOUT_AES) || (HWY_TARGET & HWY_NEON_BF16)
    constexpr size_t UNROLL_FACTOR = 4;
    constexpr size_t ITERATIONS_PER_BLOCK = 16;
#else
    constexpr size_t UNROLL_FACTOR = 8;
    constexpr size_t ITERATIONS_PER_BLOCK = 32;
#endif
    const size_t unroll_N = UNROLL_FACTOR * N;
    const size_t BLOCK_SIZE = unroll_N * ITERATIONS_PER_BLOCK;

    T global_best_val = arr[0];
    int global_best_idx = 0;
    if (std::isnan(global_best_val)) return 0;

    int base = 0;

    for (; base <= len - (int)BLOCK_SIZE; base += BLOCK_SIZE) {
        // Independent accumulator registers for OoO execution
        V best0 = hn::Set(d, IsMax ? hwy::LowestValue<T>() : hwy::HighestValue<T>());
        V best1 = best0;
        V best2 = best0;
        V best3 = best0;
        V best4 = best0;
        V best5 = best0;
        V best6 = best0;
        V best7 = best0;

        // Deferred NaN detection: accumulate validity mask
        M valid_acc = hn::Eq(best0, best0);

        const T* HWY_RESTRICT ptr = arr + base;
        const T* HWY_RESTRICT ptr_end = ptr + BLOCK_SIZE;

        for (; ptr < ptr_end; ptr += unroll_N) {
            V v0 = hn::LoadU(d, ptr);
            V v1 = hn::LoadU(d, ptr + N);
            V v2 = hn::LoadU(d, ptr + 2 * N);
            V v3 = hn::LoadU(d, ptr + 3 * N);

            // NaN check: x == x is false for NaN
            M e0 = hn::Eq(v0, v0);
            M e1 = hn::Eq(v1, v1);
            M e2 = hn::Eq(v2, v2);
            M e3 = hn::Eq(v3, v3);

#if (HWY_TARGET & HWY_NEON) || (HWY_TARGET & HWY_NEON_WITHOUT_AES) || (HWY_TARGET & HWY_NEON_BF16)
            // NEON optimized: 4-way unroll with simpler NaN check
            M e01 = hn::And(e0, e1);
            M e23 = hn::And(e2, e3);
            valid_acc = hn::And(valid_acc, hn::And(e01, e23));

            if constexpr (IsMax) {
                best0 = hn::Max(best0, v0);
                best1 = hn::Max(best1, v1);
                best2 = hn::Max(best2, v2);
                best3 = hn::Max(best3, v3);
            } else {
                best0 = hn::Min(best0, v0);
                best1 = hn::Min(best1, v1);
                best2 = hn::Min(best2, v2);
                best3 = hn::Min(best3, v3);
            }
#else
            V v4 = hn::LoadU(d, ptr + 4 * N);
            V v5 = hn::LoadU(d, ptr + 5 * N);
            V v6 = hn::LoadU(d, ptr + 6 * N);
            V v7 = hn::LoadU(d, ptr + 7 * N);

            M e4 = hn::Eq(v4, v4);
            M e5 = hn::Eq(v5, v5);
            M e6 = hn::Eq(v6, v6);
            M e7 = hn::Eq(v7, v7);

            // Tree-style mask merging
            M e01 = hn::And(e0, e1);
            M e23 = hn::And(e2, e3);
            M e45 = hn::And(e4, e5);
            M e67 = hn::And(e6, e7);
            M e0123 = hn::And(e01, e23);
            M e4567 = hn::And(e45, e67);

            valid_acc = hn::And(valid_acc, hn::And(e0123, e4567));

            if constexpr (IsMax) {
                best0 = hn::Max(best0, v0);
                best1 = hn::Max(best1, v1);
                best2 = hn::Max(best2, v2);
                best3 = hn::Max(best3, v3);
                best4 = hn::Max(best4, v4);
                best5 = hn::Max(best5, v5);
                best6 = hn::Max(best6, v6);
                best7 = hn::Max(best7, v7);
            } else {
                best0 = hn::Min(best0, v0);
                best1 = hn::Min(best1, v1);
                best2 = hn::Min(best2, v2);
                best3 = hn::Min(best3, v3);
                best4 = hn::Min(best4, v4);
                best5 = hn::Min(best5, v5);
                best6 = hn::Min(best6, v6);
                best7 = hn::Min(best7, v7);
            }
#endif
        }

        // If valid_acc is not all True, NaN was detected
        if (BRANCH_UNLIKELY(!hn::AllTrue(d, valid_acc))) {
            for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                if (std::isnan(arr[base + i])) return base + i;
            }
        }

        // Reduce accumulators to 1
#if (HWY_TARGET & HWY_NEON) || (HWY_TARGET & HWY_NEON_WITHOUT_AES) || (HWY_TARGET & HWY_NEON_BF16)
        // NEON: 4-way reduce
        if constexpr (IsMax) {
            best0 = hn::Max(best0, best1);
            best2 = hn::Max(best2, best3);
            best0 = hn::Max(best0, best2);
        } else {
            best0 = hn::Min(best0, best1);
            best2 = hn::Min(best2, best3);
            best0 = hn::Min(best0, best2);
        }
#else
        // AVX-512 / SVE: 8-way reduce
        if constexpr (IsMax) {
            best0 = hn::Max(best0, best1);
            best2 = hn::Max(best2, best3);
            best4 = hn::Max(best4, best5);
            best6 = hn::Max(best6, best7);
            best0 = hn::Max(best0, best2);
            best4 = hn::Max(best4, best6);
            best0 = hn::Max(best0, best4);
        } else {
            best0 = hn::Min(best0, best1);
            best2 = hn::Min(best2, best3);
            best4 = hn::Min(best4, best5);
            best6 = hn::Min(best6, best7);
            best0 = hn::Min(best0, best2);
            best4 = hn::Min(best4, best6);
            best0 = hn::Min(best0, best4);
        }
#endif

        T block_best_val;
        if constexpr (IsMax) {
            block_best_val = hn::GetLane(hn::MaxOfLanes(d, best0));
        } else {
            block_best_val = hn::GetLane(hn::MinOfLanes(d, best0));
        }

        bool is_better = IsMax ? (block_best_val > global_best_val) : (block_best_val < global_best_val);
        if (BRANCH_UNLIKELY(is_better)) {
            global_best_val = block_best_val;
            // Optimized index search: use SIMD to find first match
            const T* HWY_RESTRICT search_ptr = arr + base;
            const T* HWY_RESTRICT search_end = search_ptr + BLOCK_SIZE;
            for (; search_ptr < search_end; search_ptr += N) {
                V v = hn::LoadU(d, search_ptr);
                M match = hn::Eq(v, hn::Set(d, block_best_val));
                if (BRANCH_UNLIKELY(!hn::AllFalse(d, match))) {
                    for (size_t i = 0; i < N; ++i) {
                        if (search_ptr[i] == block_best_val) {
                            global_best_idx = base + (search_ptr - arr - base) + i;
                            goto block_done;
                        }
                    }
                }
            }
        }
        block_done:;
    }

    // --- Medium tail ---
    for (; base <= len - (int)N; base += N) {
        V v = hn::LoadU(d, arr + base);
        if (BRANCH_UNLIKELY(!hn::AllTrue(d, hn::Eq(v, v)))) {
            for (size_t i = 0; i < N; ++i) {
                if (std::isnan(arr[base + i])) return base + i;
            }
        }
        T v_best;
        if constexpr (IsMax) {
            v_best = hn::GetLane(hn::MaxOfLanes(d, v));
        } else {
            v_best = hn::GetLane(hn::MinOfLanes(d, v));
        }

        bool is_better = IsMax ? (v_best > global_best_val) : (v_best < global_best_val);
        if (BRANCH_UNLIKELY(is_better)) {
            global_best_val = v_best;
            for (size_t i = 0; i < N; ++i) {
                if (arr[base + i] == v_best) {
                    global_best_idx = base + i;
                    break;
                }
            }
        }
    }

    // --- Scalar remainder ---
    for (; base < len; ++base) {
        T val = arr[base];
        if (BRANCH_UNLIKELY(std::isnan(val))) return base;
        bool is_better = IsMax ? (val > global_best_val) : (val < global_best_val);
        if (is_better) {
            global_best_val = val;
            global_best_idx = base;
        }
    }

    return global_best_idx;
}

template <typename T, bool IsMax>
int ComputeArgMinMaxInteger(const T* HWY_RESTRICT arr, npy_intp len)
{
    if (len <= 0) return -1;

    hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);
    const size_t unroll_N = 4 * N;

    // Force inner loop to iterate at least 64 times.
    // For int8 (N=16), BLOCK_SIZE = 4096 elements, amortizing smaxv overhead.
    // For int64 (N=2), BLOCK_SIZE = 512 elements, staying in L1 cache fast zone.
    const size_t ITERATIONS_PER_BLOCK = 64;
    const size_t BLOCK_SIZE = unroll_N * ITERATIONS_PER_BLOCK;

    T global_best_val = arr[0];
    int global_best_idx = 0;

    int base = 0;

    for (; base <= len - (int)BLOCK_SIZE; base += BLOCK_SIZE) {
        V best0 = hn::Set(d, IsMax ? hwy::LowestValue<T>() : hwy::HighestValue<T>());
        V best1 = best0;
        V best2 = best0;
        V best3 = best0;

        const T* HWY_RESTRICT ptr = arr + base;
        const T* HWY_RESTRICT ptr_end = ptr + BLOCK_SIZE;

        // Fast inner loop: pure Load and Max/Min only
        for (; ptr < ptr_end; ptr += unroll_N) {
            V v0 = hn::LoadU(d, ptr);
            V v1 = hn::LoadU(d, ptr + N);
            V v2 = hn::LoadU(d, ptr + 2 * N);
            V v3 = hn::LoadU(d, ptr + 3 * N);

            if constexpr (IsMax) {
                best0 = hn::Max(best0, v0);
                best1 = hn::Max(best1, v1);
                best2 = hn::Max(best2, v2);
                best3 = hn::Max(best3, v3);
            } else {
                best0 = hn::Min(best0, v0);
                best1 = hn::Min(best1, v1);
                best2 = hn::Min(best2, v2);
                best3 = hn::Min(best3, v3);
            }
        }

        // Reduce 4 accumulators to 1
        if constexpr (IsMax) {
            best0 = hn::Max(best0, best1);
            best2 = hn::Max(best2, best3);
            best0 = hn::Max(best0, best2);
        } else {
            best0 = hn::Min(best0, best1);
            best2 = hn::Min(best2, best3);
            best0 = hn::Min(best0, best2);
        }

        // Extract the block's extreme scalar value
        T block_best_val;
        if constexpr (IsMax) {
            block_best_val = hn::GetLane(hn::MaxOfLanes(d, best0));
        } else {
            block_best_val = hn::GetLane(hn::MinOfLanes(d, best0));
        }

        bool is_better = IsMax ? (block_best_val > global_best_val) : (block_best_val < global_best_val);

        if (BRANCH_UNLIKELY(is_better)) {
            global_best_val = block_best_val;
            // Find index based on base offset
            for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                if (arr[base + i] == block_best_val) {
                    global_best_idx = base + i;
                    break;
                }
            }
        }
    }

    // Medium tail processing (less than BLOCK_SIZE)
    for (; base <= len - (int)N; base += N) {
        V v = hn::LoadU(d, arr + base);
        T v_best;
        if constexpr (IsMax) {
            v_best = hn::GetLane(hn::MaxOfLanes(d, v));
        } else {
            v_best = hn::GetLane(hn::MinOfLanes(d, v));
        }

        bool is_better = IsMax ? (v_best > global_best_val) : (v_best < global_best_val);
        if (BRANCH_UNLIKELY(is_better)) {
            global_best_val = v_best;
            for (size_t i = 0; i < N; ++i) {
                if (arr[base + i] == v_best) {
                    global_best_idx = base + i;
                    break;
                }
            }
        }
    }

    // Pure scalar remainder processing
    for (; base < len; ++base) {
        T val = arr[base];
        bool is_better = IsMax ? (val > global_best_val) : (val < global_best_val);
        if (is_better) {
            global_best_val = val;
            global_best_idx = base;
        }
    }

    return global_best_idx;
}

template <typename T>
int ComputeArgMaxWrapper(const T* HWY_RESTRICT arr, npy_intp len)
{
    if constexpr (std::is_integral<T>::value) {
        return ComputeArgMinMaxInteger<T, true>(arr, len);
    } else {
        return ComputeArgMinMaxFloating<T, true>(arr, len);
    }
}

template <typename T>
int ComputeArgMinWrapper(const T* HWY_RESTRICT arr, npy_intp len)
{
    if constexpr (std::is_integral<T>::value) {
        return ComputeArgMinMaxInteger<T, false>(arr, len);
    } else {
        return ComputeArgMinMaxFloating<T, false>(arr, len);
    }
}

} // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

#define DECLARE_NPY_INT_ARGMINMAX(NPY_TYPE_NAME, NPY_TYPE, STD_TYPE) \
    extern "C" NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(NPY_TYPE_NAME##_argmax)( \
        NPY_TYPE *ip, npy_intp n, npy_intp *mindx, PyArrayObject *NPY_UNUSED(aip)) \
    { \
        *mindx = HWY_STATIC_DISPATCH(ComputeArgMaxWrapper<STD_TYPE>)( \
                    reinterpret_cast<const STD_TYPE*>(ip), n); \
        return 0; \
    } \
    \
    extern "C" NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(NPY_TYPE_NAME##_argmin)( \
        NPY_TYPE *ip, npy_intp n, npy_intp *mindx, PyArrayObject *NPY_UNUSED(aip)) \
    { \
        *mindx = HWY_STATIC_DISPATCH(ComputeArgMinWrapper<STD_TYPE>)( \
                    reinterpret_cast<const STD_TYPE*>(ip), n); \
        return 0; \
    }

// ----------------------------------------------------------------------------
// Unsigned integer dispatch direct binding
// ----------------------------------------------------------------------------
DECLARE_NPY_INT_ARGMINMAX(UBYTE,     npy_ubyte,     uint8_t)
DECLARE_NPY_INT_ARGMINMAX(USHORT,    npy_ushort,    uint16_t)
DECLARE_NPY_INT_ARGMINMAX(UINT,      npy_uint,      uint32_t)
DECLARE_NPY_INT_ARGMINMAX(ULONG,     npy_ulong,     uint64_t)
DECLARE_NPY_INT_ARGMINMAX(ULONGLONG, npy_ulonglong, uint64_t)

// ----------------------------------------------------------------------------
// Signed integer dispatch direct binding
// ----------------------------------------------------------------------------
DECLARE_NPY_INT_ARGMINMAX(BYTE,      npy_byte,      int8_t)
DECLARE_NPY_INT_ARGMINMAX(SHORT,     npy_short,     int16_t)
DECLARE_NPY_INT_ARGMINMAX(INT,       npy_int,       int32_t)
DECLARE_NPY_INT_ARGMINMAX(LONG,      npy_long,      int64_t)
DECLARE_NPY_INT_ARGMINMAX(LONGLONG,  npy_longlong,  int64_t)

// ----------------------------------------------------------------------------
// Floating point type dispatch direct binding
// ----------------------------------------------------------------------------
DECLARE_NPY_INT_ARGMINMAX(FLOAT,  npy_float,  float)
DECLARE_NPY_INT_ARGMINMAX(DOUBLE,  npy_double,  double)

// LONGDOUBLE: No SIMD support, use scalar fallback
extern "C" NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(LONGDOUBLE_argmax)(
    npy_longdouble *ip, npy_intp n, npy_intp *mindx, PyArrayObject *NPY_UNUSED(aip))
{
    npy_longdouble mp = *ip;
    if (npy_isnan(mp)) {
        *mindx = 0;
        return 0;
    }
    *mindx = 0;
    for (npy_intp i = 1; i < n; ++i) {
        npy_longdouble a = ip[i];
        if (!(a <= mp)) {  // negated, for correct nan handling
            mp = a;
            *mindx = i;
            if (npy_isnan(mp)) {
                break;
            }
        }
    }
    return 0;
}

extern "C" NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(LONGDOUBLE_argmin)(
    npy_longdouble *ip, npy_intp n, npy_intp *mindx, PyArrayObject *NPY_UNUSED(aip))
{
    npy_longdouble mp = *ip;
    if (npy_isnan(mp)) {
        *mindx = 0;
        return 0;
    }
    *mindx = 0;
    for (npy_intp i = 1; i < n; ++i) {
        npy_longdouble a = ip[i];
        if (!(a >= mp)) {  // negated, for correct nan handling
            mp = a;
            *mindx = i;
            if (npy_isnan(mp)) {
                break;
            }
        }
    }
    return 0;
}

// Cleanup macro
#undef DECLARE_NPY_INT_ARGMINMAX

extern "C" NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(BOOL_argmax)(
    npy_bool *ip, npy_intp n, npy_intp *mindx, PyArrayObject *NPY_UNUSED(aip))
{
    *mindx = HWY_STATIC_DISPATCH(ComputeArgMaxBool)(reinterpret_cast<const uint8_t*>(ip), n);
    return 0;
}

// BOOL_argmin uses platform-specific implementation:
// - ARM64: Highway SIMD ComputeArgMinBool for better performance
// - x86/other: memchr is highly optimized (often SIMD-internal) and faster
extern "C" NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(BOOL_argmin)(
    npy_bool *ip, npy_intp n, npy_intp *mindx, PyArrayObject *NPY_UNUSED(aip))
{
#if defined(__aarch64__)
    *mindx = HWY_STATIC_DISPATCH(ComputeArgMinBool)(reinterpret_cast<const uint8_t*>(ip), n);
#else
    npy_bool* p = (npy_bool*)memchr(ip, 0, n * sizeof(*ip));
    if (p == NULL) {
        *mindx = 0;
        return 0;
    }
    *mindx = p - ip;
#endif
    return 0;
}