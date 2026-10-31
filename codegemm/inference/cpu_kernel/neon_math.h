// Pure arithmetic helpers for the ARM variants, deliberately independent of
// PyTorch. They can be cross-compiled on WSL to inspect the generated AArch64
// instructions, while the actual extension must be built against Pi PyTorch.
#pragma once

#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "These kernels require AArch64 with ARM NEON; no scalar fallback is used."
#endif

#include <arm_neon.h>
#include <cstdint>

namespace codegemm_cpu_neon {

constexpr int kBooks = 2;
constexpr int kCentroids = 256;
constexpr int kVector = 8;
constexpr int kSegmentsPerGroup = 16;
constexpr int kCodebookFloats = kBooks * kCentroids * kVector;
constexpr int kPsumFloats = kSegmentsPerGroup * kBooks * kCentroids;

struct Shape {
    int64_t K;
    int64_t M;
    int64_t P;
};

// Transpose four rows of four floats. Doing this explicitly avoids relying on
// compiler auto-vectorization in cpu-2/cpu-3, which are compiled with -O0.
inline void transpose_four(
    const float* row0, const float* row1,
    const float* row2, const float* row3,
    float* destination, int64_t stride
) {
    const float32x4x2_t ab = vtrnq_f32(vld1q_f32(row0), vld1q_f32(row1));
    const float32x4x2_t cd = vtrnq_f32(vld1q_f32(row2), vld1q_f32(row3));
    vst1q_f32(destination,
        vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0])));
    vst1q_f32(destination + stride,
        vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1])));
    vst1q_f32(destination + 2 * stride,
        vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0])));
    vst1q_f32(destination + 3 * stride,
        vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1])));
}

// Checkpoints keep [book, centroid, feature]. The dot-product loop benefits
// from [book, feature, centroid]: one NEON register then represents four
// independent centroids, so no horizontal reduction is required.
// This 16 KiB temporary is rebuilt per call; learned values and the checkpoint
// layout are unchanged, and load_state_dict cannot leave a stale cached copy.
inline void transpose_codebooks(const float* source, float* transposed) {
    for (int64_t book = 0; book < kBooks; ++book) {
        for (int64_t centroid = 0; centroid < kCentroids; centroid += 4) {
            const float* first = source + (book * kCentroids + centroid) * kVector;
            float* target = transposed + book * kVector * kCentroids + centroid;
            transpose_four(first, first + 8, first + 16, first + 24,
                           target, kCentroids);
            transpose_four(first + 4, first + 12, first + 20, first + 28,
                           target + 4 * kCentroids, kCentroids);
        }
    }
}

// One task computes every centroid dot product for one (segment, book) pair.
// Four independent accumulators handle sixteen centroids at a time, giving
// the CPU independent arithmetic work while reusing each broadcast input.
// These are FP32 FMA operations, not integer dot products: saved bytes are
// centroid indices, rather than numerical int8 weights.
inline void build_psum_task(
    const float* input, const float* transposed, float* psum, int64_t task
) {
    const int64_t segment = task / kBooks;
    const int64_t book = task % kBooks;
    const float* x = input + segment * kVector;
    const float* cb = transposed + book * kVector * kCentroids;
    float* table = psum + task * kCentroids;

    for (int64_t centroid = 0; centroid < kCentroids; centroid += 16) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        for (int64_t feature = 0; feature < kVector; ++feature) {
            const float32x4_t value = vdupq_n_f32(x[feature]);
            const float* values = cb + feature * kCentroids + centroid;
            sum0 = vfmaq_f32(sum0, vld1q_f32(values), value);
            sum1 = vfmaq_f32(sum1, vld1q_f32(values + 4), value);
            sum2 = vfmaq_f32(sum2, vld1q_f32(values + 8), value);
            sum3 = vfmaq_f32(sum3, vld1q_f32(values + 12), value);
        }
        vst1q_f32(table + centroid, sum0);
        vst1q_f32(table + centroid + 4, sum1);
        vst1q_f32(table + centroid + 8, sum2);
        vst1q_f32(table + centroid + 12, sum3);
    }
}

// NEON has no arbitrary FP32 gather instruction. Select four table entries
// with scalar loads, then perform row accumulation and scaling in SIMD.
inline float32x4_t gather_four(const float* table, uint32x4_t indices) {
    const float values[4] = {
        table[vgetq_lane_u32(indices, 0)],
        table[vgetq_lane_u32(indices, 1)],
        table[vgetq_lane_u32(indices, 2)],
        table[vgetq_lane_u32(indices, 3)],
    };
    return vld1q_f32(values);
}

inline uint32_t decode_scalar(
    const int32_t* codes, const Shape& shape,
    int64_t book, int64_t segment, int64_t row
) {
    const int64_t offset = (book * shape.P + segment / 4) * shape.M + row;
    const uint32_t word = static_cast<uint32_t>(codes[offset]);
    return (word >> (8 * (segment % 4))) & 255U;
}

// Each worker owns disjoint output rows. No atomics or cross-thread floating
// reductions are needed, so changing worker count preserves accumulation order
// within a row. Unaligned NEON loads are valid for partial worker boundaries.
inline void lookup_rows(
    const int32_t* codes, const float* scales, const float* psum,
    const Shape& shape, int64_t group, float* output,
    int64_t row_begin, int64_t row_end
) {
    int64_t row = row_begin;
    const uint32x4_t mask = vdupq_n_u32(255U);
    for (; row + 4 <= row_end; row += 4) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (int64_t packed = 0; packed < 4; ++packed) {
            const int64_t position = group * 4 + packed;
            const uint32x4_t word0 = vreinterpretq_u32_s32(
                vld1q_s32(codes + position * shape.M + row));
            const uint32x4_t word1 = vreinterpretq_u32_s32(
                vld1q_s32(codes + (shape.P + position) * shape.M + row));
            for (int byte = 0; byte < 4; ++byte) {
                const int64_t segment = packed * 4 + byte;
                const int32x4_t shift = vdupq_n_s32(-8 * byte);
                const uint32x4_t index0 = vandq_u32(vshlq_u32(word0, shift), mask);
                const uint32x4_t index1 = vandq_u32(vshlq_u32(word1, shift), mask);
                sum = vaddq_f32(sum, gather_four(psum + segment * 2 * 256, index0));
                sum = vaddq_f32(sum, gather_four(psum + (segment * 2 + 1) * 256, index1));
            }
        }
        const float32x4_t scaled = vmulq_f32(sum,
            vld1q_f32(scales + group * shape.M + row));
        vst1q_f32(output + row, vaddq_f32(vld1q_f32(output + row), scaled));
    }
    // Output widths need not be multiples of four. Handle the final 1-3 rows
    // without reading beyond the checkpoint or output allocation.
    for (; row < row_end; ++row) {
        float sum = 0.0f;
        for (int64_t segment = 0; segment < 16; ++segment) {
            for (int64_t book = 0; book < 2; ++book) {
                const uint32_t index = decode_scalar(
                    codes, shape, book, group * 16 + segment, row);
                sum += psum[(segment * 2 + book) * 256 + index];
            }
        }
        output[row] += sum * scales[group * shape.M + row];
    }
}

// Reconstruct complete eight-feature segments with two NEON loads/stores.
// Input width is divisible by 128, so there is no feature tail to handle.
inline void reconstruct_rows(
    const int32_t* codes, const float* scales, const float* codebooks,
    const Shape& shape, int64_t block_begin, float* weight,
    int64_t row_begin, int64_t row_end
) {
    for (int64_t row = row_begin; row < row_end; ++row) {
        float* destination = weight + (row - block_begin) * shape.K;
        for (int64_t segment = 0; segment < shape.K / 8; ++segment) {
            const uint32_t index0 = decode_scalar(codes, shape, 0, segment, row);
            const uint32_t index1 = decode_scalar(codes, shape, 1, segment, row);
            const float* book0 = codebooks + index0 * 8;
            const float* book1 = codebooks + (256 + index1) * 8;
            const float32x4_t scale = vdupq_n_f32(scales[(segment / 16) * shape.M + row]);
            vst1q_f32(destination + segment * 8,
                vmulq_f32(vaddq_f32(vld1q_f32(book0), vld1q_f32(book1)), scale));
            vst1q_f32(destination + segment * 8 + 4,
                vmulq_f32(vaddq_f32(vld1q_f32(book0 + 4), vld1q_f32(book1 + 4)), scale));
        }
    }
}

} // namespace codegemm_cpu_neon
