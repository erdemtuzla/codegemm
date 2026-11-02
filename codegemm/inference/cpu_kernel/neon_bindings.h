// Shared implementation for cpu-2, cpu-3, and cpu-4. Separate tiny translation
// units select separate module/object names; only the threading macro and
// compiler flags distinguish successive experiments.
#pragma once

#include <torch/extension.h>
#include <array>
#include <cstdint>

#include "build_info.h"
#include "cpu_parallel.h"
#include "neon_math.h"

#if CODEGEMM_CPU_VARIANT < 2 || CODEGEMM_CPU_VARIANT > 4
#error "The NEON implementation is used only by cpu-2/cpu-3/cpu-4."
#endif

namespace {

using torch::Tensor;
using codegemm_cpu_neon::Shape;
using codegemm_cpu::for_chunks;

// Reject device, dtype, and storage mismatches before taking raw pointers.
// These checks are deliberately inside the C++ boundary as Python callers can
// invoke an extension function directly without the normal layer dispatcher.
void check_float(const Tensor& value, const char* name) {
    TORCH_CHECK(value.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32, name, " must have dtype float32");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

Shape validate(
    const Tensor& codes, const Tensor& scales,
    const Tensor& codebooks, int64_t group_size
) {
    TORCH_CHECK(group_size == 128, "CPU backend requires group_size=128");
    TORCH_CHECK(codes.device().is_cpu(), "codes must be on CPU");
    TORCH_CHECK(codes.scalar_type() == torch::kInt32, "codes must have dtype int32");
    TORCH_CHECK(codes.is_contiguous(), "codes must be contiguous");
    TORCH_CHECK(codes.dim() == 3 && codes.size(0) == 2,
                "codes must have shape [2, K/32, M]");
    check_float(scales, "scales");
    check_float(codebooks, "codebooks");
    TORCH_CHECK(codebooks.dim() == 3 && codebooks.size(0) == 2 &&
                codebooks.size(1) == 256 && codebooks.size(2) == 8,
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

Tensor gemv(
    const Tensor& x, const Tensor& codes,
    const Tensor& scales, const Tensor& codebooks, int64_t group_size
) {
    const Shape shape = validate(codes, scales, codebooks, group_size);
    check_float(x, "input");
    TORCH_CHECK(x.dim() == 1 && x.numel() == shape.K,
                "gemv input must have shape [K]");
    const float* input = x.data_ptr<float>();
    const int32_t* packed = codes.data_ptr<int32_t>();
    const float* scale = scales.data_ptr<float>();
    const float* books = codebooks.data_ptr<float>();
    Tensor output = torch::zeros({shape.M}, x.options());
    float* result = output.data_ptr<float>();

    // Fixed temporary storage: 16 KiB transposed books + 32 KiB Psumbooks.
    // Tables are shared read-only during lookup; no full dense matrix exists.
    // Cache-line alignment helps the SIMD data layout but is not a load-safety
    // requirement: all NEON helpers also accept unaligned tensor pointers.
    alignas(64) std::array<float, codegemm_cpu_neon::kCodebookFloats> transposed;
    alignas(64) std::array<float, codegemm_cpu_neon::kPsumFloats> psum;
    codegemm_cpu_neon::transpose_codebooks(books, transposed.data());

    for (int64_t group = 0; group < shape.K / 128; ++group) {
        // Each task writes its own [segment, book, :] table. parallel_for
        // completes before returning, providing the required stage barrier.
        for_chunks(32, [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
                codegemm_cpu_neon::build_psum_task(
                    input + group * 128, transposed.data(), psum.data(), task);
            }
        });

        // A row belongs to exactly one worker. Finish all row updates before
        // the next group reuses the Psumbook buffer. This retains per-row
        // accumulation order for reproducibility across worker counts.
        for_chunks(shape.M, [&](int64_t begin, int64_t end) {
            codegemm_cpu_neon::lookup_rows(
                packed, scale, psum.data(), shape, group, result, begin, end);
        });
    }
    return output;
}

Tensor dequantize_block(
    const Tensor& codes, const Tensor& scales, const Tensor& codebooks,
    int64_t group_size, int64_t row_begin, int64_t row_end
) {
    const Shape shape = validate(codes, scales, codebooks, group_size);
    TORCH_CHECK(row_begin >= 0 && row_begin < row_end && row_end <= shape.M,
                "invalid output-row block");
    const int32_t* packed = codes.data_ptr<int32_t>();
    const float* scale = scales.data_ptr<float>();
    const float* books = codebooks.data_ptr<float>();
    Tensor weight = torch::empty({row_end - row_begin, shape.K}, scales.options());
    float* destination = weight.data_ptr<float>();

    // Only the caller-requested rows are expanded. Each worker owns separate
    // destination rows; no dense weight cache or checkpoint mutation is used.
    for_chunks(row_end - row_begin, [&](int64_t begin, int64_t end) {
        codegemm_cpu_neon::reconstruct_rows(packed, scale, books, shape,
            row_begin, destination, row_begin + begin, row_begin + end);
    });
    return weight;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("build_info", &codegemm_cpu::build_info);
    // Releasing the Python GIL permits other Python work; actual C++ worker
    // parallelism is selected by for_chunks in variants cpu-3/cpu-4 only.
    module.def("gemv", &gemv, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("dequantize_block", &dequantize_block,
               pybind11::call_guard<pybind11::gil_scoped_release>());
}
