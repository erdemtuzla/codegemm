// Scalar cpu-1 arithmetic with the same work partitioning as cpu-3.
// No NEON intrinsics, automatic vectorization, FMA contraction, or compiler
// optimization are enabled. This gives the NEON implementation a threaded
// comparison baseline while leaving the original cpu-1 source unchanged.
#define CODEGEMM_CPU_SCALAR_MT 1

#include <torch/extension.h>
#include <array>
#include <cstdint>

#include "build_info.h"
#include "cpu_parallel.h"

namespace {

using torch::Tensor;
using codegemm_cpu::for_chunks;

struct Shape {
    int64_t K; // Input width.
    int64_t M; // Output rows.
    int64_t P; // Four 8-bit indices per packed word, covering 32 input features.
};

// Every raw-pointer access below assumes CPU, FP32, contiguous storage.
void check_float(const Tensor& value, const char* name) {
    TORCH_CHECK(value.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32, name, " must have dtype float32");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

Shape validate(const Tensor& codes, const Tensor& scales,
               const Tensor& books, int64_t group_size) {
    TORCH_CHECK(group_size == 128, "CPU backend requires group_size=128");
    TORCH_CHECK(codes.device().is_cpu(), "codes must be on CPU");
    TORCH_CHECK(codes.scalar_type() == torch::kInt32, "codes must have dtype int32");
    TORCH_CHECK(codes.is_contiguous(), "codes must be contiguous");
    TORCH_CHECK(codes.dim() == 3 && codes.size(0) == 2,
                "codes must have shape [2, K/32, M]");
    check_float(scales, "scales");
    check_float(books, "codebooks");
    TORCH_CHECK(books.dim() == 3 && books.size(0) == 2 &&
                books.size(1) == 256 && books.size(2) == 8,
                "CPU backend requires codebooks with shape [2, 256, 8]");
    const int64_t P = codes.size(1);
    const int64_t M = codes.size(2);
    const int64_t K = P * 32;
    TORCH_CHECK(P > 0 && M > 0, "feature dimensions must be positive");
    TORCH_CHECK(K % 128 == 0, "input features must be divisible by 128");
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == K / 128 &&
                scales.size(1) == M, "scales must have shape [K/128, M]");
    return {K, M, P};
}

// Storage and index order exactly match cpu-1, including indices 128-255.
// Convert the stored signed word to unsigned before shifting its bit pattern.
uint32_t decode(const int32_t* codes, const Shape& shape,
                int64_t book, int64_t segment, int64_t row) {
    const int64_t offset = (book * shape.P + segment / 4) * shape.M + row;
    const uint32_t word = static_cast<uint32_t>(codes[offset]);
    return (word >> (8 * (segment % 4))) & 255U;
}

Tensor gemv(const Tensor& x, const Tensor& codes, const Tensor& scales,
            const Tensor& books, int64_t group_size) {
    const Shape shape = validate(codes, scales, books, group_size);
    check_float(x, "input");
    TORCH_CHECK(x.dim() == 1 && x.numel() == shape.K,
                "gemv input must have shape [K]");
    const float* input = x.data_ptr<float>();
    const int32_t* packed = codes.data_ptr<int32_t>();
    const float* scale = scales.data_ptr<float>();
    const float* codebooks = books.data_ptr<float>();
    Tensor output = torch::zeros({shape.M}, x.options());
    float* result = output.data_ptr<float>();

    // Same 32 KiB Psumbook as cpu-1. It is shared between workers, but every
    // build task owns a different [segment, book, :] range of this array.
    alignas(64) std::array<float, 16 * 2 * 256> psum;
    for (int64_t group = 0; group < shape.K / 128; ++group) {
        // Match cpu-3's 32 (segment, book) tasks and join them before lookup.
        // Each dot product keeps cpu-1's feature order and scalar arithmetic.
        for_chunks(32, [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
                const int64_t segment = task / 2;
                const int64_t book = task % 2;
                const int64_t input_start = group * 128 + segment * 8;
                for (int64_t centroid = 0; centroid < 256; ++centroid) {
                    float dot = 0.0f;
                    for (int64_t feature = 0; feature < 8; ++feature) {
                        dot += input[input_start + feature]
                            * codebooks[(book * 256 + centroid) * 8 + feature];
                    }
                    psum[task * 256 + centroid] = dot;
                }
            }
        });

        // Rows are independent. Give every row one worker, retaining the
        // segment/book/group accumulation order and requiring no atomics.
        for_chunks(shape.M, [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                float group_sum = 0.0f;
                for (int64_t segment = 0; segment < 16; ++segment) {
                    for (int64_t book = 0; book < 2; ++book) {
                        const uint32_t index = decode(
                            packed, shape, book, group * 16 + segment, row);
                        group_sum += psum[(segment * 2 + book) * 256 + index];
                    }
                }
                result[row] += group_sum * scale[group * shape.M + row];
            }
        });
        // for_chunks joins here before the next group reuses the same buffer.
    }
    return output;
}

Tensor dequantize_block(const Tensor& codes, const Tensor& scales,
                        const Tensor& books, int64_t group_size,
                        int64_t row_begin, int64_t row_end) {
    const Shape shape = validate(codes, scales, books, group_size);
    TORCH_CHECK(row_begin >= 0 && row_begin < row_end && row_end <= shape.M,
                "invalid output-row block");
    const int32_t* packed = codes.data_ptr<int32_t>();
    const float* scale = scales.data_ptr<float>();
    const float* codebooks = books.data_ptr<float>();
    Tensor weight = torch::empty({row_end - row_begin, shape.K}, scales.options());
    float* destination = weight.data_ptr<float>();

    // Match cpu-3's row partitions for prompt processing. Only the requested
    // block is expanded; the Python caller still limits it to 128 output rows.
    for_chunks(row_end - row_begin, [&](int64_t begin, int64_t end) {
        for (int64_t row = row_begin + begin; row < row_begin + end; ++row) {
            for (int64_t segment = 0; segment < shape.K / 8; ++segment) {
                const uint32_t index0 = decode(packed, shape, 0, segment, row);
                const uint32_t index1 = decode(packed, shape, 1, segment, row);
                const float factor = scale[(segment / 16) * shape.M + row];
                for (int64_t feature = 0; feature < 8; ++feature) {
                    const float sum = codebooks[index0 * 8 + feature]
                        + codebooks[(256 + index1) * 8 + feature];
                    destination[(row - row_begin) * shape.K + segment * 8 + feature]
                        = sum * factor;
                }
            }
        }
    });
    return weight;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("build_info", &codegemm_cpu::build_info);
    // The GIL is released for both calls. Worker scheduling uses the same
    // helper as the NEON backends, rather than Python threads or processes.
    module.def("gemv", &gemv, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("dequantize_block", &dequantize_block,
               pybind11::call_guard<pybind11::gil_scoped_release>());
}
