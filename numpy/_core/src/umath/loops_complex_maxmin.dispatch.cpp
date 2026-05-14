/*
 * Copyright (c) 2025, NumPy Developers
 * Distributed under the BSD-3-Clause license
 * See LICENSE.txt for more information
 *
 * Highway SIMD-optimized complex maximum/minimum ufunc loops.
 *
 * Replaces the scalar CGE/CLE-based complex max/min from loops.c.src with
 * vectorized Highway SIMD kernels for four memory access patterns:
 *   - Map:    contiguous Array vs Array -> Array
 *   - Reduce: contiguous Array -> Scalar
 *   - Bcast1: Scalar vs Array -> Array
 *   - Bcast2: Array vs Scalar -> Array
 *
 * Non-standard strides and trailing elements fall back to a branchless
 * scalar loop.  CLONGDOUBLE always uses the scalar path (no SIMD).
 */

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <cmath>
#include <cstddef>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

/*
 * Complex comparison macros: lexicographic order by (real, imag).
 * CGE = complex greater-or-equal, CLE = complex less-or-equal.
 * NaN in either operand of > / < yields false, so NaN propagation
 * is handled by the caller checking std::isnan before these macros.
 */
#define CGE(in1r, in1i, in2r, in2i) \
        ((in1r) > (in2r) || ((in1r) == (in2r) && (in1i) >= (in2i)))
#define CLE(in1r, in1i, in2r, in2i) \
        ((in1r) < (in2r) || ((in1r) == (in2r) && (in1i) <= (in2i)))

namespace {

/*
 * Branchless scalar fallback for non-standard strides or tail elements.
 * Preserves NumPy NaN propagation: if in1 has NaN, keep in1;
 * otherwise compare with CGE/CLE (in2 NaN propagates via failed comparison).
 */
template <typename T, bool IsMax>
static void
scalar_loop(char *ip1, char *ip2, char *op1,
        npy_intp n, npy_intp is1, npy_intp is2, npy_intp os1)
{
    for (npy_intp i = 0; i < n;
            i++, ip1 += is1, ip2 += is2, op1 += os1) {
        T in1r = ((T *)ip1)[0];
        T in1i = ((T *)ip1)[1];
        const T in2r = ((T *)ip2)[0];
        const T in2i = ((T *)ip2)[1];

        bool keep_in1 = std::isnan(in1r) || std::isnan(in1i);
        if (!keep_in1) {
            keep_in1 = IsMax ? CGE(in1r, in1i, in2r, in2i)
                             : CLE(in1r, in1i, in2r, in2i);
        }

        if (!keep_in1) {
            in1r = in2r;
            in1i = in2i;
        }

        ((T *)op1)[0] = in1r;
        ((T *)op1)[1] = in1i;
    }
}

#if NPY_HWY

/*
 * Contiguous Map: both inputs and output are fully contiguous.
 * Uses interleaved load/store for complex real/imag separation.
 */
template <typename T, bool IsMax>
HWY_ATTR static npy_intp
simd_map(const T *src1, const T *src2, T *dst, npy_intp n)
{
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    using M = hn::Mask<decltype(d)>;

    const npy_intp lanes = static_cast<npy_intp>(hn::Lanes(d));
    const npy_intp vec_n = (n / lanes) * lanes;

    for (npy_intp i = 0; i < vec_n; i += lanes) {
        V v1r, v1i, v2r, v2i;
        hn::LoadInterleaved2(d, src1 + 2 * i, v1r, v1i);
        hn::LoadInterleaved2(d, src2 + 2 * i, v2r, v2i);

        M keep_in1;
        if constexpr (IsMax) {
            M r_gt = hn::Gt(v1r, v2r);
            M r_eq = hn::Eq(v1r, v2r);
            M i_ge = hn::Ge(v1i, v2i);
            M cge_mask = hn::Or(r_gt, hn::And(r_eq, i_ge));
            M nan_mask = hn::Or(hn::IsNaN(v1r), hn::IsNaN(v1i));
            keep_in1 = hn::Or(nan_mask, cge_mask);
        } else {
            M r_lt = hn::Lt(v1r, v2r);
            M r_eq = hn::Eq(v1r, v2r);
            M i_le = hn::Le(v1i, v2i);
            M cle_mask = hn::Or(r_lt, hn::And(r_eq, i_le));
            M nan_mask = hn::Or(hn::IsNaN(v1r), hn::IsNaN(v1i));
            keep_in1 = hn::Or(nan_mask, cle_mask);
        }

        V vr = hn::IfThenElse(keep_in1, v1r, v2r);
        V vi = hn::IfThenElse(keep_in1, v1i, v2i);
        hn::StoreInterleaved2(vr, vi, d, dst + 2 * i);
    }

    return vec_n;
}

/*
 * Reduction: output is a single scalar complex value.
 * Accumulates in vector registers, then does a horizontal reduction.
 */
template <typename T, bool IsMax>
HWY_ATTR static npy_intp
simd_reduce(const T *src1, const T *src2, T *dst, npy_intp n)
{
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    using M = hn::Mask<decltype(d)>;
    const npy_intp lanes = static_cast<npy_intp>(hn::Lanes(d));
    const npy_intp vec_n = (n / lanes) * lanes;

    if (vec_n == 0) {
        return 0;
    }

    V acc_r = hn::Set(d, src1[0]);
    V acc_i = hn::Set(d, src1[1]);

    for (npy_intp i = 0; i < vec_n; i += lanes) {
        V v2r, v2i;
        hn::LoadInterleaved2(d, src2 + 2 * i, v2r, v2i);

        M keep_acc;
        if constexpr (IsMax) {
            M r_gt = hn::Gt(acc_r, v2r);
            M r_eq = hn::Eq(acc_r, v2r);
            M i_ge = hn::Ge(acc_i, v2i);
            keep_acc = hn::Or(
                    hn::Or(hn::IsNaN(acc_r), hn::IsNaN(acc_i)),
                    hn::Or(r_gt, hn::And(r_eq, i_ge)));
        } else {
            M r_lt = hn::Lt(acc_r, v2r);
            M r_eq = hn::Eq(acc_r, v2r);
            M i_le = hn::Le(acc_i, v2i);
            keep_acc = hn::Or(
                    hn::Or(hn::IsNaN(acc_r), hn::IsNaN(acc_i)),
                    hn::Or(r_lt, hn::And(r_eq, i_le)));
        }

        acc_r = hn::IfThenElse(keep_acc, acc_r, v2r);
        acc_i = hn::IfThenElse(keep_acc, acc_i, v2i);
    }

    /* Horizontal reduction via stack store (no heap allocation). */
    T temp_r[hn::Lanes(d)];
    T temp_i[hn::Lanes(d)];
    hn::StoreU(acc_r, d, temp_r);
    hn::StoreU(acc_i, d, temp_i);

    T best_r = temp_r[0];
    T best_i = temp_i[0];

    for (npy_intp i = 1; i < lanes; ++i) {
        bool keep_1 = std::isnan(best_r) || std::isnan(best_i);
        if (!keep_1) {
            keep_1 = IsMax
                    ? CGE(best_r, best_i, temp_r[i], temp_i[i])
                    : CLE(best_r, best_i, temp_r[i], temp_i[i]);
        }
        if (!keep_1) {
            best_r = temp_r[i];
            best_i = temp_i[i];
        }
    }

    dst[0] = best_r;
    dst[1] = best_i;

    return vec_n;
}

/*
 * Broadcast 1: left operand is scalar, right operand is array.
 * The scalar NaN state is computed once and reused across all lanes.
 */
template <typename T, bool IsMax>
HWY_ATTR static npy_intp
simd_bcast1(const T *src1, const T *src2, T *dst, npy_intp n)
{
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    using M = hn::Mask<decltype(d)>;
    const npy_intp lanes = static_cast<npy_intp>(hn::Lanes(d));
    const npy_intp vec_n = (n / lanes) * lanes;

    V v1r = hn::Set(d, src1[0]);
    V v1i = hn::Set(d, src1[1]);
    M nan_1 = hn::Or(hn::IsNaN(v1r), hn::IsNaN(v1i));

    for (npy_intp i = 0; i < vec_n; i += lanes) {
        V v2r, v2i;
        hn::LoadInterleaved2(d, src2 + 2 * i, v2r, v2i);

        M keep_in1;
        if constexpr (IsMax) {
            M r_gt = hn::Gt(v1r, v2r);
            M r_eq = hn::Eq(v1r, v2r);
            M i_ge = hn::Ge(v1i, v2i);
            keep_in1 = hn::Or(nan_1,
                    hn::Or(r_gt, hn::And(r_eq, i_ge)));
        } else {
            M r_lt = hn::Lt(v1r, v2r);
            M r_eq = hn::Eq(v1r, v2r);
            M i_le = hn::Le(v1i, v2i);
            keep_in1 = hn::Or(nan_1,
                    hn::Or(r_lt, hn::And(r_eq, i_le)));
        }
        hn::StoreInterleaved2(
                hn::IfThenElse(keep_in1, v1r, v2r),
                hn::IfThenElse(keep_in1, v1i, v2i),
                d, dst + 2 * i);
    }
    return vec_n;
}

/*
 * Broadcast 2: left operand is array, right operand is scalar.
 */
template <typename T, bool IsMax>
HWY_ATTR static npy_intp
simd_bcast2(const T *src1, const T *src2, T *dst, npy_intp n)
{
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    using M = hn::Mask<decltype(d)>;
    const npy_intp lanes = static_cast<npy_intp>(hn::Lanes(d));
    const npy_intp vec_n = (n / lanes) * lanes;

    V v2r = hn::Set(d, src2[0]);
    V v2i = hn::Set(d, src2[1]);

    for (npy_intp i = 0; i < vec_n; i += lanes) {
        V v1r, v1i;
        hn::LoadInterleaved2(d, src1 + 2 * i, v1r, v1i);

        M keep_in1;
        if constexpr (IsMax) {
            M r_gt = hn::Gt(v1r, v2r);
            M r_eq = hn::Eq(v1r, v2r);
            M i_ge = hn::Ge(v1i, v2i);
            keep_in1 = hn::Or(
                    hn::Or(hn::IsNaN(v1r), hn::IsNaN(v1i)),
                    hn::Or(r_gt, hn::And(r_eq, i_ge)));
        } else {
            M r_lt = hn::Lt(v1r, v2r);
            M r_eq = hn::Eq(v1r, v2r);
            M i_le = hn::Le(v1i, v2i);
            keep_in1 = hn::Or(
                    hn::Or(hn::IsNaN(v1r), hn::IsNaN(v1i)),
                    hn::Or(r_lt, hn::And(r_eq, i_le)));
        }
        hn::StoreInterleaved2(
                hn::IfThenElse(keep_in1, v1r, v2r),
                hn::IfThenElse(keep_in1, v1i, v2i),
                d, dst + 2 * i);
    }
    return vec_n;
}

#endif /* NPY_HWY */

} /* anonymous namespace */

/*
 * Extern "C" ufunc loop functions.
 *
 * Each function detects the memory access pattern (Map, Reduce, Bcast1,
 * Bcast2) and dispatches to the appropriate Highway SIMD kernel when
 * available, falling back to a branchless scalar loop for non-standard
 * strides or when Highway is disabled.
 */

template <typename T, bool IsMax>
static void
execute_complex_op(char **args, const npy_intp *dimensions,
        const npy_intp *steps)
{
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
    npy_intp n = dimensions[0];
    const npy_intp csz = sizeof(T) * 2;

    /* Identify standard memory access patterns. */
    bool is_map    = (is1 == csz && is2 == csz && os1 == csz);
    bool is_reduce = (is1 == 0   && is2 == csz && os1 == 0);
    bool is_bcast1 = (is1 == 0   && is2 == csz && os1 == csz);
    bool is_bcast2 = (is1 == csz && is2 == 0   && os1 == csz);

    if (n > 0) {
#if NPY_HWY
        npy_intp processed = 0;

        if (is_map) {
            processed = simd_map<T, IsMax>(
                    (const T *)ip1, (const T *)ip2, (T *)op1, n);
        } else if (is_reduce) {
            processed = simd_reduce<T, IsMax>(
                    (const T *)ip1, (const T *)ip2, (T *)op1, n);
        } else if (is_bcast1) {
            processed = simd_bcast1<T, IsMax>(
                    (const T *)ip1, (const T *)ip2, (T *)op1, n);
        } else if (is_bcast2) {
            processed = simd_bcast2<T, IsMax>(
                    (const T *)ip1, (const T *)ip2, (T *)op1, n);
        }

        ip1 += processed * is1;
        ip2 += processed * is2;
        op1 += processed * os1;
        n -= processed;
#endif /* NPY_HWY */

        if (n > 0) {
            scalar_loop<T, IsMax>(ip1, ip2, op1, n, is1, is2, os1);
        }
    }
}

extern "C" {

NPY_NO_EXPORT void
CFLOAT_maximum(char **args, npy_intp const *dimensions,
        npy_intp const *steps, void *NPY_UNUSED(func))
{
    execute_complex_op<float, true>(args, dimensions, steps);
    npy_clear_floatstatus_barrier((char *)dimensions);
}

NPY_NO_EXPORT void
CFLOAT_minimum(char **args, npy_intp const *dimensions,
        npy_intp const *steps, void *NPY_UNUSED(func))
{
    execute_complex_op<float, false>(args, dimensions, steps);
    npy_clear_floatstatus_barrier((char *)dimensions);
}

NPY_NO_EXPORT void
CDOUBLE_maximum(char **args, npy_intp const *dimensions,
        npy_intp const *steps, void *NPY_UNUSED(func))
{
    execute_complex_op<double, true>(args, dimensions, steps);
    npy_clear_floatstatus_barrier((char *)dimensions);
}

NPY_NO_EXPORT void
CDOUBLE_minimum(char **args, npy_intp const *dimensions,
        npy_intp const *steps, void *NPY_UNUSED(func))
{
    execute_complex_op<double, false>(args, dimensions, steps);
    npy_clear_floatstatus_barrier((char *)dimensions);
}

NPY_NO_EXPORT void
CLONGDOUBLE_maximum(char **args, npy_intp const *dimensions,
        npy_intp const *steps, void *NPY_UNUSED(func))
{
    scalar_loop<long double, true>(
            args[0], args[1], args[2],
            dimensions[0], steps[0], steps[1], steps[2]);
    npy_clear_floatstatus_barrier((char *)dimensions);
}

NPY_NO_EXPORT void
CLONGDOUBLE_minimum(char **args, npy_intp const *dimensions,
        npy_intp const *steps, void *NPY_UNUSED(func))
{
    scalar_loop<long double, false>(
            args[0], args[1], args[2],
            dimensions[0], steps[0], steps[1], steps[2]);
    npy_clear_floatstatus_barrier((char *)dimensions);
}

} /* extern "C" */
