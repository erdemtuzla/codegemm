#include <torch/extension.h>

#include <array>
#include <cstdint>

using torch::Tensor;

struct Shape {
    int64_t K; // Number of input features
    int64_t M; // Number of output features
    int64_t P; // Number of packed words along the input dimension
};

static void check_float(const Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32,
        name, " must have dtype float32"
    );
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Reject unsuported shapes and dtypes before accessing tensor memory.
static Shape validate(const Tensor& codes, const Tensor& scales,
                      const Tensor& codebooks, int64_t group_size
) {
    TORCH_CHECK(group_size == 128, "CPU backend requires group_size=128");

    TORCH_CHECK(codes.device().is_cpu(), "codes must be on CPU");
    TORCH_CHECK(
        codes.scalar_type() == torch::kInt32,
        "codes must have dtype int32"
    );
    TORCH_CHECK(codes.is_contiguous(), "codes must be contiguous");
    TORCH_CHECK(
        codes.dim() == 3 && codes.size(0) == 2,
        "codes must have shape [2, K/32, M]"
    );

    check_float(scales, "scales");
    check_float(codebooks, "codebooks");

    TORCH_CHECK(
        codebooks.dim() == 3 &&
        codebooks.size(0) == 2 &&
        codebooks.size(1) == 256 &&
        codebooks.size(2) == 8,
        "CPU backend requires codebooks with shape [2, 256, 8]"
    );

    const int64_t P = codes.size(1);
    const int64_t M = codes.size(2);
    const int64_t K = P * 32;

    TORCH_CHECK(P > 0 && M > 0, "feature dimensions must be positive");
    TORCH_CHECK(K % 128 == 0, "input features must be divisible by 128");

    TORCH_CHECK(
        scales.dim() == 2 &&
        scales.size(0) == K / 128 &&
        scales.size(1) == M,
        "scales must have shape [K/128, M]"
    );

    return {K, M, P};
}

// Reads one of the four indices inside and int32
static uint32_t decode(
    const int32_t* codes,
    const Shape& shape,
    int64_t book,
    int64_t segment,
    int64_t row
) {
    const int64_t offset = (book * shape.P + segment / 4) * shape.M + row;

    // Converting to uint32_t preserves the stored 32-bit pattern.
    // This avouds signed-shift problems for bytes 128 through 255.
    const uint32_t word = static_cast<uint32_t>(codes[offset]);
    const int shift = static_cast<int>(8 * (segment % 4));

    return (word >> shift) & 0xFFU;
}

// Calculates centroid/input dot products once, then reuses them through lookup
Tensor gemv(
    const Tensor& x,
    const Tensor& codes,
    const Tensor& scales,
    const Tensor& codebooks,
    int64_t group_size
) {
    const Shape shape = validate(codes, scales, codebooks, group_size);

    check_float(x, "input");
    TORCH_CHECK(
        x.dim() == 1 && x.numel() == shape.K,
        "gemv input must have shape [K]"
    );

    const auto* xp = x.data_ptr<float>();
    const auto* qp = codes.data_ptr<int32_t>();
    const auto* sp = scales.data_ptr<float>();
    const auto* cp = codebooks.data_ptr<float>();

    auto output = torch::zeros({shape.M}, x.options());
    auto* yp = output.data_ptr<float>();

    // One scale group contains 128 / 8 = 16 input segments.
    // Temporary memory: 16 * 2 * 256 floats = 32 KiB.
    std::array<float, 16 * 2 * 256> psum;

    for (int64_t group = 0; group < shape.K / 128; ++group) {
        // Build the Psumbooks for this input group.
        for (int64_t segment = 0; segment < 16; ++segment) {
            const int64_t input_start = group * 128 + segment * 8;

            for (int64_t book = 0; book < 2; ++book) {
                for (int64_t centroid = 0; centroid < 256; ++centroid) {
                    float dot = 0.0f;

                    for (int64_t j = 0; j < 8; ++j) {
                        const int64_t cb_offset =
                            (book * 256 + centroid) * 8 + j;

                        dot += xp[input_start + j] * cp[cb_offset];
                    }

                    psum[(segment * 2 + book) * 256 + centroid] = dot;
                }
            }
        }

        // Decode each row's indices and accumulate selected entries.
        for (int64_t row = 0; row < shape.M; ++row) {
            float group_sum = 0.0f;

            for (int64_t segment = 0; segment < 16; ++segment) {
                const int64_t global_segment = group * 16 + segment;

                for (int64_t book = 0; book < 2; ++book) {
                    const uint32_t index =
                        decode(qp, shape, book, global_segment, row);

                    group_sum +=
                        psum[(segment * 2 + book) * 256 + index];
                }
            }

            yp[row] += group_sum * sp[group * shape.M + row];
        }
    }

    return output;
}

// Reconstructs only the requested output rows for prompt processing
Tensor dequantize_block(
    const Tensor& codes,
    const Tensor& scales,
    const Tensor& codebooks,
    int64_t group_size,
    int64_t row_begin,
    int64_t row_end
) {
    const Shape shape = validate(codes, scales, codebooks, group_size);

    TORCH_CHECK(
        row_begin >= 0 &&
        row_begin < row_end &&
        row_end <= shape.M,
        "invalid output-row block"
    );

    const auto* qp = codes.data_ptr<int32_t>();
    const auto* sp = scales.data_ptr<float>();
    const auto* cp = codebooks.data_ptr<float>();

    auto weight = torch::empty(
        {row_end - row_begin, shape.K},
        scales.options()
    );
    auto* wp = weight.data_ptr<float>();

    for (int64_t row = row_begin; row < row_end; ++row) {
        for (int64_t segment = 0; segment < shape.K / 8; ++segment) {
            const uint32_t index0 = decode(qp, shape, 0, segment, row);
            const uint32_t index1 = decode(qp, shape, 1, segment, row);

            const float scale = sp[(segment / 16) * shape.M + row];

            for (int64_t j = 0; j < 8; ++j) {
                const float centroid_sum =
                    cp[index0 * 8 + j] +
                    cp[(256 + index1) * 8 + j];

                wp[
                    (row - row_begin) * shape.K + segment * 8 + j
                ] = centroid_sum * scale;
            }
        }
    }

    return weight;
}

// Makes the two C++ functions callable from Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "gemv",
        &gemv,
        pybind11::call_guard<pybind11::gil_scoped_release>()
    );

    module.def(
        "dequantize_block",
        &dequantize_block,
        pybind11::call_guard<pybind11::gil_scoped_release>()
    );
}