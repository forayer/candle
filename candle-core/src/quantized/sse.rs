//! SSSE3/SSE4.1 fallback for the quantized matmul kernels.
//!
//! Only `vec_dot_q6k_q8k` is implemented here. This is a straight port of
//! `super::avx::vec_dot_q6k_q8k`, splitting every 256-bit (`__m256i`/`__m256`)
//! operation into a pair of 128-bit (`__m128i`/`__m128`) operations acting on
//! the low and high halves of the original register. The element-to-scale
//! pairing is preserved exactly (see the `get_scale_shuffle` table, which
//! replicates each of the 16 scale bytes 8 times - the low 8 words produced by
//! `maddubs_epi16` on a 16-byte half line up with the first scale byte of that
//! half, and the high 8 words line up with the second).
//!
//! This path exists for x86_64 CPUs that have SSSE3 + SSE4.1 but lack AVX2
//! (e.g. pre-Sandy-Bridge-refresh / pre-Haswell-era Xeons), where candle-core
//! upstream has no vectorized fallback and silently drops to a pure-scalar
//! generic implementation. It intentionally does not use FMA (`_mm_fmadd_ps`)
//! since SSE4.1-only CPUs typically lack it too; a plain multiply + add is
//! used instead.
//!
//! No other quant formats (q2k/q3k/q4k/q5k/q4_0/q8_0/...) are ported: see
//! `docs/ops/test-server-runbook.md` in the `tutelary-ai` repo for the
//! empirical analysis (GGUF header tensor-type scan) showing that Q6K is the
//! only quantized weight format present in the model this fork targets.

use super::k_quants::{BlockQ6K, BlockQ8K, QK_K};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
unsafe fn get_scale_shuffle(i: usize) -> __m128i {
    const K_SHUFFLE: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
        13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];
    _mm_loadu_si128((K_SHUFFLE.as_ptr() as *const __m128i).add(i))
}

#[inline(always)]
unsafe fn hsum_float_4(x: __m128) -> f32 {
    let res = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

/// Widens each byte of `v` into the low 8 or high 8 words of a 128-bit
/// register, matching how `_mm256_cvtepi8_epi16` widens a full 16-byte
/// `__m128i` into a 256-bit 16-word register (low 8 bytes -> low 8 words,
/// high 8 bytes -> high 8 words).
#[inline(always)]
unsafe fn cvtepi8_epi16_lo(v: __m128i) -> __m128i {
    _mm_cvtepi8_epi16(v)
}

#[inline(always)]
unsafe fn cvtepi8_epi16_hi(v: __m128i) -> __m128i {
    _mm_cvtepi8_epi16(_mm_srli_si128(v, 8))
}

pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q6k_q8k (sse): {n} is not divisible by {QK_K}"
    );

    unsafe {
        let m4 = _mm_set1_epi8(0xF);
        let m2 = _mm_set1_epi8(3);
        let m32s = _mm_set1_epi8(32);
        let mut acc_lo = _mm_setzero_ps();
        let mut acc_hi = _mm_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let d_bcast = _mm_set1_ps(d);
            let mut q4 = x.ql.as_ptr();
            let mut qh = x.qh.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let mut sumi_lo = _mm_setzero_si128();
            let mut sumi_hi = _mm_setzero_si128();

            for j in 0..QK_K / 128 {
                let is = j * 4;
                let scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is));
                let scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                let scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                let scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));

                // Each "256-bit" load below is replaced by a pair of 128-bit
                // loads (lo = first 16 bytes, hi = next 16 bytes).
                let q4bits1_lo = _mm_loadu_si128(q4 as *const __m128i);
                let q4bits1_hi = _mm_loadu_si128(q4.add(16) as *const __m128i);
                q4 = q4.add(32);
                let q4bits2_lo = _mm_loadu_si128(q4 as *const __m128i);
                let q4bits2_hi = _mm_loadu_si128(q4.add(16) as *const __m128i);
                q4 = q4.add(32);
                let q4bits_h_lo = _mm_loadu_si128(qh as *const __m128i);
                let q4bits_h_hi = _mm_loadu_si128(qh.add(16) as *const __m128i);
                qh = qh.add(32);

                let (q4h_0_lo, q4h_0_hi) = (
                    _mm_slli_epi16(_mm_and_si128(q4bits_h_lo, m2), 4),
                    _mm_slli_epi16(_mm_and_si128(q4bits_h_hi, m2), 4),
                );
                let (q4h_1_lo, q4h_1_hi) = (
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_lo, 2), m2), 4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_hi, 2), m2), 4),
                );
                let (q4h_2_lo, q4h_2_hi) = (
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_lo, 4), m2), 4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_hi, 4), m2), 4),
                );
                let (q4h_3_lo, q4h_3_hi) = (
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_lo, 6), m2), 4),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bits_h_hi, 6), m2), 4),
                );

                let (q4_0_lo, q4_0_hi) = (
                    _mm_or_si128(_mm_and_si128(q4bits1_lo, m4), q4h_0_lo),
                    _mm_or_si128(_mm_and_si128(q4bits1_hi, m4), q4h_0_hi),
                );
                let (q4_1_lo, q4_1_hi) = (
                    _mm_or_si128(_mm_and_si128(q4bits2_lo, m4), q4h_1_lo),
                    _mm_or_si128(_mm_and_si128(q4bits2_hi, m4), q4h_1_hi),
                );
                let (q4_2_lo, q4_2_hi) = (
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_lo, 4), m4), q4h_2_lo),
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_hi, 4), m4), q4h_2_hi),
                );
                let (q4_3_lo, q4_3_hi) = (
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_lo, 4), m4), q4h_3_lo),
                    _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_hi, 4), m4), q4h_3_hi),
                );

                let (q8_0_lo, q8_0_hi) = (
                    _mm_loadu_si128(q8 as *const __m128i),
                    _mm_loadu_si128(q8.add(16) as *const __m128i),
                );
                q8 = q8.add(32);
                let (q8_1_lo, q8_1_hi) = (
                    _mm_loadu_si128(q8 as *const __m128i),
                    _mm_loadu_si128(q8.add(16) as *const __m128i),
                );
                q8 = q8.add(32);
                let (q8_2_lo, q8_2_hi) = (
                    _mm_loadu_si128(q8 as *const __m128i),
                    _mm_loadu_si128(q8.add(16) as *const __m128i),
                );
                q8 = q8.add(32);
                let (q8_3_lo, q8_3_hi) = (
                    _mm_loadu_si128(q8 as *const __m128i),
                    _mm_loadu_si128(q8.add(16) as *const __m128i),
                );
                q8 = q8.add(32);

                let q8s_0_lo = _mm_maddubs_epi16(m32s, q8_0_lo);
                let q8s_0_hi = _mm_maddubs_epi16(m32s, q8_0_hi);
                let q8s_1_lo = _mm_maddubs_epi16(m32s, q8_1_lo);
                let q8s_1_hi = _mm_maddubs_epi16(m32s, q8_1_hi);
                let q8s_2_lo = _mm_maddubs_epi16(m32s, q8_2_lo);
                let q8s_2_hi = _mm_maddubs_epi16(m32s, q8_2_hi);
                let q8s_3_lo = _mm_maddubs_epi16(m32s, q8_3_lo);
                let q8s_3_hi = _mm_maddubs_epi16(m32s, q8_3_hi);

                let p16_0_lo = _mm_maddubs_epi16(q4_0_lo, q8_0_lo);
                let p16_0_hi = _mm_maddubs_epi16(q4_0_hi, q8_0_hi);
                let p16_1_lo = _mm_maddubs_epi16(q4_1_lo, q8_1_lo);
                let p16_1_hi = _mm_maddubs_epi16(q4_1_hi, q8_1_hi);
                let p16_2_lo = _mm_maddubs_epi16(q4_2_lo, q8_2_lo);
                let p16_2_hi = _mm_maddubs_epi16(q4_2_hi, q8_2_hi);
                let p16_3_lo = _mm_maddubs_epi16(q4_3_lo, q8_3_lo);
                let p16_3_hi = _mm_maddubs_epi16(q4_3_hi, q8_3_hi);

                let p16_0_lo = _mm_sub_epi16(p16_0_lo, q8s_0_lo);
                let p16_0_hi = _mm_sub_epi16(p16_0_hi, q8s_0_hi);
                let p16_1_lo = _mm_sub_epi16(p16_1_lo, q8s_1_lo);
                let p16_1_hi = _mm_sub_epi16(p16_1_hi, q8s_1_hi);
                let p16_2_lo = _mm_sub_epi16(p16_2_lo, q8s_2_lo);
                let p16_2_hi = _mm_sub_epi16(p16_2_hi, q8s_2_hi);
                let p16_3_lo = _mm_sub_epi16(p16_3_lo, q8s_3_lo);
                let p16_3_hi = _mm_sub_epi16(p16_3_hi, q8s_3_hi);

                let p16_0_lo = _mm_madd_epi16(cvtepi8_epi16_lo(scale_0), p16_0_lo);
                let p16_0_hi = _mm_madd_epi16(cvtepi8_epi16_hi(scale_0), p16_0_hi);
                let p16_1_lo = _mm_madd_epi16(cvtepi8_epi16_lo(scale_1), p16_1_lo);
                let p16_1_hi = _mm_madd_epi16(cvtepi8_epi16_hi(scale_1), p16_1_hi);
                let p16_2_lo = _mm_madd_epi16(cvtepi8_epi16_lo(scale_2), p16_2_lo);
                let p16_2_hi = _mm_madd_epi16(cvtepi8_epi16_hi(scale_2), p16_2_hi);
                let p16_3_lo = _mm_madd_epi16(cvtepi8_epi16_lo(scale_3), p16_3_lo);
                let p16_3_hi = _mm_madd_epi16(cvtepi8_epi16_hi(scale_3), p16_3_hi);

                sumi_lo = _mm_add_epi32(sumi_lo, _mm_add_epi32(p16_0_lo, p16_1_lo));
                sumi_hi = _mm_add_epi32(sumi_hi, _mm_add_epi32(p16_0_hi, p16_1_hi));
                sumi_lo = _mm_add_epi32(sumi_lo, _mm_add_epi32(p16_2_lo, p16_3_lo));
                sumi_hi = _mm_add_epi32(sumi_hi, _mm_add_epi32(p16_2_hi, p16_3_hi));
            }

            acc_lo = _mm_add_ps(acc_lo, _mm_mul_ps(d_bcast, _mm_cvtepi32_ps(sumi_lo)));
            acc_hi = _mm_add_ps(acc_hi, _mm_mul_ps(d_bcast, _mm_cvtepi32_ps(sumi_hi)));
        }
        hsum_float_4(_mm_add_ps(acc_lo, acc_hi))
    }
}
