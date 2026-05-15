#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "highway_qsort.hpp"
#include "quicksort.hpp"

#include <algorithm>
#include <vector>

#include "common.hpp"

namespace np::highway::qsort_simd {
template <typename T>
void NPY_CPU_DISPATCH_CURFX(QSort)(T *arr, npy_intp size)
{
#if VQSORT_ENABLED
    using THwy = std::conditional_t<std::is_same_v<T, Half>, hwy::float16_t, T>;
    hwy::HWY_NAMESPACE::VQSortStatic(reinterpret_cast<THwy*>(arr), size, hwy::SortAscending());
#else
    sort::Quick(arr, size);
#endif
}
#if !HWY_HAVE_FLOAT16
template <>
void NPY_CPU_DISPATCH_CURFX(QSort)<Half>(Half *arr, npy_intp size)
{
    sort::Quick(arr, size);
}
#endif // !HWY_HAVE_FLOAT16

template void NPY_CPU_DISPATCH_CURFX(QSort)<int16_t>(int16_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint16_t>(uint16_t*, npy_intp);
#if HWY_HAVE_FLOAT16
template void NPY_CPU_DISPATCH_CURFX(QSort)<Half>(Half*, npy_intp);
#endif

namespace {

constexpr npy_intp kSmallArgSort16 = 64;

template <typename T>
inline bool ArgLess16(T *arr, npy_intp a, npy_intp b)
{
    if constexpr (std::is_same_v<T, Half>) {
        if (arr[a].IsNaN()) return false;
        if (arr[b].IsNaN()) return true;
    }
    return arr[a] < arr[b];
}

template <typename T>
void ArgInsertionSort16(T *arr, npy_intp *arg, npy_intp size)
{
    for (npy_intp i = 0; i < size; ++i) {
        arg[i] = i;
    }
    for (npy_intp *pi = arg + 1; pi < arg + size; ++pi) {
        npy_intp vi = *pi;
        npy_intp *pj = pi;
        while (pj > arg && ArgLess16<T>(arr, vi, *(pj - 1))) {
            *pj = *(pj - 1);
            --pj;
        }
        *pj = vi;
    }
}

template <typename T>
inline uint32_t ToSortableKey16(T val)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        return static_cast<uint32_t>(val);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        return static_cast<uint32_t>(static_cast<uint16_t>(val) ^ 0x8000U);
    }
    else if constexpr (std::is_same_v<T, Half>) {
        uint16_t u = val.Bits();
        if (val.IsNaN()) {
            return 0xFFFFFFFFU;
        }
        if (u & 0x8000U) {
            return ~static_cast<uint32_t>(u);
        } else {
            return static_cast<uint32_t>(u) | 0x80000000U;
        }
    }
}

template <typename T>
bool CheckSortedReversed16(T *arr, npy_intp size, npy_intp *arg)
{
    if (size <= 1) {
        if (size == 1) arg[0] = 0;
        return true;
    }

    bool is_sorted = true;
    for (npy_intp i = 1; i < size; ++i) {
        if (ArgLess16<T>(arr, i, i - 1)) {
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

    bool is_reversed = true;
    for (npy_intp i = 1; i < size; ++i) {
        if (!ArgLess16<T>(arr, i, i - 1)) {
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
void NPY_CPU_DISPATCH_CURFX(ArgQSort)(T *arr, npy_intp* arg, npy_intp size)
{
    if (CheckSortedReversed16<T>(arr, size, arg)) {
        return;
    }

    if (size < kSmallArgSort16) {
        ArgInsertionSort16(arr, arg, size);
        return;
    }

#if VQSORT_ENABLED
    std::vector<hwy::K32V32> pairs(size);
    for (npy_intp i = 0; i < size; ++i) {
        pairs[i].key = ToSortableKey16(arr[i]);
        pairs[i].value = static_cast<uint32_t>(i);
    }
    hwy::HWY_NAMESPACE::VQSortStatic(pairs.data(), static_cast<size_t>(size), hwy::SortAscending());
    for (npy_intp i = 0; i < size; ++i) {
        arg[i] = static_cast<npy_intp>(pairs[i].value);
    }
#else
    for (npy_intp i = 0; i < size; ++i) {
        arg[i] = i;
    }
    std::sort(arg, arg + size, [arr](npy_intp a, npy_intp b) {
        return ArgLess16<T>(arr, a, b);
    });
#endif
}

template <typename T>
void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(T *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    if (num <= 1) {
        if (num == 1) arg[0] = 0;
        return;
    }

    if (num < kSmallArgSort16) {
        for (npy_intp i = 0; i < num; ++i) {
            arg[i] = i;
        }
        for (npy_intp i = 0; i <= kth; ++i) {
            npy_intp min_idx = i;
            for (npy_intp j = i + 1; j < num; ++j) {
                if (ArgLess16<T>(arr, arg[j], arg[min_idx])) {
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                std::swap(arg[i], arg[min_idx]);
            }
        }
        return;
    }

#if VQSORT_ENABLED
    std::vector<hwy::K32V32> pairs(num);
    for (npy_intp i = 0; i < num; ++i) {
        pairs[i].key = ToSortableKey16(arr[i]);
        pairs[i].value = static_cast<uint32_t>(i);
    }
    hwy::HWY_NAMESPACE::VQSelectStatic(pairs.data(), static_cast<size_t>(num),
                                       static_cast<size_t>(kth), hwy::SortAscending());
    for (npy_intp i = 0; i < num; ++i) {
        arg[i] = static_cast<npy_intp>(pairs[i].value);
    }
#else
    for (npy_intp i = 0; i < num; ++i) {
        arg[i] = i;
    }
    std::nth_element(arg, arg + kth, arg + num, [arr](npy_intp a, npy_intp b) {
        return ArgLess16<T>(arr, a, b);
    });
#endif
}

template void NPY_CPU_DISPATCH_CURFX(ArgQSort)<int16_t>(int16_t*, npy_intp*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(ArgQSelect)<int16_t>(int16_t*, npy_intp*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(ArgQSort)<uint16_t>(uint16_t*, npy_intp*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(ArgQSelect)<uint16_t>(uint16_t*, npy_intp*, npy_intp, npy_intp);
#if HWY_HAVE_FLOAT16
template void NPY_CPU_DISPATCH_CURFX(ArgQSort)<Half>(Half*, npy_intp*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(ArgQSelect)<Half>(Half*, npy_intp*, npy_intp, npy_intp);
#endif

} // np::highway::qsort_simd
