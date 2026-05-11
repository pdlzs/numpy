#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "highway_qsort.hpp"
#include "quicksort.hpp"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <utility>
#include <vector>

#include "common.hpp"

namespace np::highway::qsort_simd {
template <typename T>
void NPY_CPU_DISPATCH_CURFX(QSort)(T *arr, npy_intp size)
{
#if VQSORT_ENABLED
    hwy::HWY_NAMESPACE::VQSortStatic(arr, size, hwy::SortAscending());
#else
    sort::Quick(arr, size);
#endif
}

template <typename T>
void NPY_CPU_DISPATCH_CURFX(QSelect)(T *arr, npy_intp num, npy_intp kth)
{
#if VQSORT_ENABLED
    hwy::HWY_NAMESPACE::VQSelectStatic(arr, num, kth, hwy::SortAscending());
#else
    sort::Quick(arr, num);
#endif
}

template void NPY_CPU_DISPATCH_CURFX(QSort)<int32_t>(int32_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint32_t>(uint32_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<int64_t>(int64_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint64_t>(uint64_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<float>(float*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<double>(double*, npy_intp);

template void NPY_CPU_DISPATCH_CURFX(QSelect)<int32_t>(int32_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<uint32_t>(uint32_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<int64_t>(int64_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<uint64_t>(uint64_t*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<float>(float*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSelect)<double>(double*, npy_intp, npy_intp);

namespace {

/*
 * Threshold for using insertion sort vs Highway SIMD.
 * Determined empirically: insertion sort is faster for small arrays
 * due to SIMD setup overhead. Value based on x86/ARM benchmarks.
 */
constexpr npy_intp kSmallArgSort = 64;

template <typename T>
inline bool ArgLess(T *arr, npy_intp a, npy_intp b)
{
    if constexpr (std::is_floating_point_v<T>) {
        if (arr[a] != arr[a]) return false;  // NaN not less than anything, goes to end
        if (arr[b] != arr[b]) return true;   // Non-NaN less than NaN, goes before NaN
    }
    return arr[a] < arr[b];
}

template <typename T>
void ArgInsertionSort(T *arr, npy_intp *arg, npy_intp size)
{
    for (npy_intp i = 0; i < size; ++i) {
        arg[i] = i;
    }
    for (npy_intp *pi = arg + 1; pi < arg + size; ++pi) {
        npy_intp vi = *pi;
        npy_intp *pj = pi;
        while (pj > arg && ArgLess<T>(arr, vi, *(pj - 1))) {
            *pj = *(pj - 1);
            --pj;
        }
        *pj = vi;
    }
}

/*
 * ArgInsertionSelect: Performs selection (not full sort).
 * Only ensures the kth element is in its correct position,
 * with elements before kth being smaller and after being larger.
 */
template <typename T>
void ArgInsertionSelect(T *arr, npy_intp *arg, npy_intp num, npy_intp kth)
{
    // Initialize indices
    for (npy_intp i = 0; i < num; ++i) {
        arg[i] = i;
    }
    // Selection: find and place the kth smallest element
    // Elements 0..kth-1 will be smaller, kth+1..num-1 larger or equal
    for (npy_intp i = 0; i <= kth; ++i) {
        npy_intp min_idx = i;
        for (npy_intp j = i + 1; j < num; ++j) {
            if (ArgLess<T>(arr, arg[j], arg[min_idx])) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            std::swap(arg[i], arg[min_idx]);
        }
    }
}

/*
 * ToSortableKey: Converts values to unsigned types for SIMD sorting.
 * Ensures proper ordering: negative < positive, NaNs go to end.
 * Returns uint64_t for consistency (32-bit types truncated).
 */
template <typename T>
inline uint64_t ToSortableKey(T val)
{
    if constexpr (std::is_same_v<T, uint32_t>) {
        return static_cast<uint64_t>(val);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        return val;
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        return static_cast<uint64_t>(static_cast<uint32_t>(val) ^ 0x80000000U);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        return static_cast<uint64_t>(val) ^ 0x8000000000000000ULL;
    }
    else if constexpr (std::is_same_v<T, float>) {
        if (std::isnan(val)) {
            return 0xFFFFFFFFFFFFFFFFULL; // NaN at end (use 64-bit max for consistency)
        }
        uint32_t u = BitCast<uint32_t, T>(val);
        if (u & 0x80000000U) {            // Negative (including -0.0)
            return static_cast<uint64_t>(~u);
        } else {                           // Non-negative (+0.0, positive, +inf)
            return static_cast<uint64_t>(u | 0x80000000U);
        }
    }
    else if constexpr (std::is_same_v<T, double>) {
        if (std::isnan(val)) {
            return 0xFFFFFFFFFFFFFFFFULL; // NaN at end
        }
        uint64_t u = BitCast<uint64_t, T>(val);
        if (u & 0x8000000000000000ULL) {   // Negative
            return ~u;
        } else {                            // Non-negative
            return u | 0x8000000000000000ULL;
        }
    }
}

template <typename T>
void ArgQSort_Fallback(T *arr, npy_intp* arg, npy_intp size)
{
    for (npy_intp i = 0; i < size; ++i) {
        arg[i] = i;
    }
    std::sort(arg, arg + size, [arr](npy_intp a, npy_intp b) {
        return ArgLess<T>(arr, a, b);
    });
}

template <typename T>
void ArgQSelect_Fallback(T *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    for (npy_intp i = 0; i < num; ++i) {
        arg[i] = i;
    }
    std::nth_element(arg, arg + kth, arg + num, [arr](npy_intp a, npy_intp b) {
        return ArgLess<T>(arr, a, b);
    });
}

/*
 * CheckSortedReversed: Checks if array is already sorted or reverse sorted.
 * Returns true if special case handled (arg initialized accordingly).
 * Only meaningful for full sort, not for select.
 */
template <typename T>
inline bool CheckSortedReversed(T *arr, npy_intp size, npy_intp *arg)
{
    if (size <= 1) {
        if (size == 1) arg[0] = 0;
        return true;
    }

    // Check for already sorted arrays
    bool is_sorted = true;
    for (npy_intp i = 1; i < size; ++i) {
        if (ArgLess<T>(arr, i, i - 1)) {
            is_sorted = false;
            break;
        }
    }
    if (is_sorted) {
        for (npy_intp i = 0; i < size; ++i) {
            arg[i] = i;
        }
        return true;
    }

    // Check for reverse sorted arrays
    bool is_reversed = true;
    for (npy_intp i = 1; i < size; ++i) {
        if (!ArgLess<T>(arr, i, i - 1)) {
            is_reversed = false;
            break;
        }
    }
    if (is_reversed) {
        for (npy_intp i = 0; i < size; ++i) {
            arg[i] = size - 1 - i;
        }
        return true;
    }

    return false;
}

} // anonymous namespace

template <typename T>
void ArgQSort_Impl(T *arr, npy_intp* arg, npy_intp size)
{
    // Early exit for sorted/reversed arrays (optimization for sort)
    if (CheckSortedReversed<T>(arr, size, arg)) {
        return;
    }

    if (size < kSmallArgSort) {
        ArgInsertionSort(arr, arg, size);
        return;
    }

#if VQSORT_ENABLED
    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float>) {
        // Use std::vector for RAII memory management
        std::vector<hwy::K32V32> pairs(size);
        for (npy_intp i = 0; i < size; ++i) {
            pairs[i].key = static_cast<uint32_t>(ToSortableKey(arr[i]));
            pairs[i].value = static_cast<uint32_t>(i);
        }
        hwy::HWY_NAMESPACE::VQSortStatic(pairs.data(), static_cast<size_t>(size), hwy::SortAscending());
        for (npy_intp i = 0; i < size; ++i) {
            arg[i] = static_cast<npy_intp>(pairs[i].value);
        }
    } else {
        std::vector<hwy::K64V64> pairs(size);
        for (npy_intp i = 0; i < size; ++i) {
            pairs[i].key = ToSortableKey(arr[i]);
            pairs[i].value = static_cast<uint64_t>(i);
        }
        hwy::HWY_NAMESPACE::VQSortStatic(pairs.data(), static_cast<size_t>(size), hwy::SortAscending());
        for (npy_intp i = 0; i < size; ++i) {
            arg[i] = static_cast<npy_intp>(pairs[i].value);
        }
    }
#else
    ArgQSort_Fallback(arr, arg, size);
#endif
}

template <typename T>
void ArgQSelect_Impl(T *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    // Note: For select operation, we cannot use sorted/reversed optimization
    // because we need to find the kth element position, not just initialize indices

    if (num <= 1) {
        if (num == 1) arg[0] = 0;
        return;
    }

    if (num < kSmallArgSort) {
        ArgInsertionSelect(arr, arg, num, kth);
        return;
    }

#if VQSORT_ENABLED
    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float>) {
        std::vector<hwy::K32V32> pairs(num);
        for (npy_intp i = 0; i < num; ++i) {
            pairs[i].key = static_cast<uint32_t>(ToSortableKey(arr[i]));
            pairs[i].value = static_cast<uint32_t>(i);
        }
        hwy::HWY_NAMESPACE::VQSelectStatic(pairs.data(), static_cast<size_t>(num),
                                           static_cast<size_t>(kth), hwy::SortAscending());
        for (npy_intp i = 0; i < num; ++i) {
            arg[i] = static_cast<npy_intp>(pairs[i].value);
        }
    } else {
        std::vector<hwy::K64V64> pairs(num);
        for (npy_intp i = 0; i < num; ++i) {
            pairs[i].key = ToSortableKey(arr[i]);
            pairs[i].value = static_cast<uint64_t>(i);
        }
        hwy::HWY_NAMESPACE::VQSelectStatic(pairs.data(), static_cast<size_t>(num),
                                           static_cast<size_t>(kth), hwy::SortAscending());
        for (npy_intp i = 0; i < num; ++i) {
            arg[i] = static_cast<npy_intp>(pairs[i].value);
        }
    }
#else
    ArgQSelect_Fallback(arr, arg, num, kth);
#endif
}

template <typename T>
void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(T *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    ArgQSelect_Impl(arr, arg, num, kth);
}

template <typename T>
void NPY_CPU_DISPATCH_CURFX(ArgQSort)(T *arr, npy_intp* arg, npy_intp size)
{
    ArgQSort_Impl(arr, arg, size);
}

#define HWAY_DISPATCH_DECLARE(TYPE) \
    template void NPY_CPU_DISPATCH_CURFX(ArgQSort)<TYPE>(TYPE*, npy_intp*, npy_intp); \
    template void NPY_CPU_DISPATCH_CURFX(ArgQSelect)<TYPE>(TYPE*, npy_intp*, npy_intp, npy_intp)

HWAY_DISPATCH_DECLARE(int32_t);
HWAY_DISPATCH_DECLARE(uint32_t);
HWAY_DISPATCH_DECLARE(int64_t);
HWAY_DISPATCH_DECLARE(uint64_t);
HWAY_DISPATCH_DECLARE(float);
HWAY_DISPATCH_DECLARE(double);
#undef HWAY_DISPATCH_DECLARE

} // np::highway::qsort_simd