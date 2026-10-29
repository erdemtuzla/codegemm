import io
import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from codegemm.inference.codegemm_linear import CodeGEMMLinear


def make_case(bias=False, out_features=7):
    generator = torch.Generator(device="cpu").manual_seed(19)
    in_features = 256

    layer = CodeGEMMLinear(
        in_features=in_features,
        out_features=out_features,
        vec_len=8,
        group_size=128,
        num_codebooks=2,
        nbits_per_codebook=8,
        bias=bias,
        backend="cpu",
    )

    indices = torch.randint(
        0,
        256,
        (2, in_features // 8, out_features),
        generator=generator,
    )

    # Ensure every test exercises these boundary indices.
    indices[:, :4, :] = torch.tensor(
        [0, 127, 128, 255]
    ).reshape(1, 4, 1)

    parts = indices.reshape(2, in_features // 32, 4, out_features)
    packed = sum(parts[:, :, j, :] << (8 * j) for j in range(4))

    layer.codes.copy_(packed.to(torch.int32))
    layer.codebooks.copy_(
        torch.randn(
            layer.codebooks.shape,
            generator=generator,
        ) * 0.1
    )

    # Distinct scales across both input groups and output rows.
    layer.scales.copy_(
        torch.stack(
            [
                torch.linspace(0.1, 0.5, out_features),
                torch.linspace(1.0, 2.0, out_features),
            ]
        )
    )

    if bias:
        layer.bias.copy_(
            torch.randn(out_features, generator=generator) * 0.1
        )

    return layer, indices


def reference_weight(layer, indices):
    # Independent reference: use the ORIGINAL unpacked indices.
    # It does not call the extension's decoder or reconstruction.
    weight = torch.empty(
        layer.out_features,
        layer.in_features,
        dtype=torch.float64,
    )

    for row in range(layer.out_features):
        for segment in range(layer.in_features // 8):
            index0 = int(indices[0, segment, row])
            index1 = int(indices[1, segment, row])

            centroid = (
                layer.codebooks[0, index0].double()
                + layer.codebooks[1, index1].double()
            )

            weight[row, segment * 8:(segment + 1) * 8] = (
                centroid
                * layer.scales[segment // 16, row].double()
            )

    return weight


def expected_output(layer, indices, x):
    output = x.double() @ reference_weight(layer, indices).t()

    if layer.bias is not None:
        output = output + layer.bias.double()

    return output.float()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (256,),
        (1, 256),
        (1, 1, 256),
        (2, 1, 256),
        (1, 3, 256),
        (2, 3, 256),
    ],
)
def test_lookup_and_prefill(shape, bias):
    layer, indices = make_case(bias=bias)
    generator = torch.Generator(device="cpu").manual_seed(29)
    x = torch.randn(shape, generator=generator)

    actual = layer(x)
    expected = expected_output(layer, indices, x)

    assert actual.shape == shape[:-1] + (layer.out_features,)
    assert actual.dtype == torch.float32

    torch.testing.assert_close(
        actual, expected, rtol=2e-5, atol=2e-5
    )


def test_noncontiguous_input_and_multiple_output_blocks():
    layer, indices = make_case(bias=True, out_features=131)
    x = torch.arange(
        2 * 256 * 3,
        dtype=torch.float32,
    ).reshape(2, 256, 3).transpose(1, 2) / 1000

    assert not x.is_contiguous()

    torch.testing.assert_close(
        layer(x),
        expected_output(layer, indices, x),
        rtol=2e-5,
        atol=2e-5,
    )


def test_repeated_calls_do_not_reuse_accumulated_output():
    layer, indices = make_case(bias=True)
    first_input = torch.ones(1, 1, 256)
    second_input = torch.full((1, 1, 256), 0.25)

    first = layer(first_input)
    second = layer(second_input)

    torch.testing.assert_close(
        first,
        expected_output(layer, indices, first_input),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        second,
        expected_output(layer, indices, second_input),
        rtol=2e-5,
        atol=2e-5,
    )


def test_checkpoint_keys_and_saved_values():
    layer, _ = make_case(bias=True)

    assert set(layer.state_dict()) == {
        "codes", "codebooks", "scales", "bias"
    }

    saved = {
        name: tensor.clone()
        for name, tensor in layer.state_dict().items()
    }

    # Exercise the floating dtypes found in the real checkpoints.
    saved["codebooks"] = saved["codebooks"].bfloat16()
    saved["scales"] = saved["scales"].bfloat16()
    saved["bias"] = saved["bias"].half()

    stream = io.BytesIO()
    torch.save(saved, stream)
    stream.seek(0)

    restored, _ = make_case(bias=True)
    restored.load_state_dict(
        torch.load(stream, weights_only=True),
        strict=True,
    )

    for name, tensor in restored.state_dict().items():
        expected = saved[name]
        if expected.is_floating_point():
            expected = expected.float()
        assert torch.equal(tensor, expected)


@pytest.mark.parametrize(
    "change",
    [
        {"num_codebooks": 1},
        {"nbits_per_codebook": 7},
        {"vec_len": 4},
        {"group_size": 64},
        {"in_features": 160},
        {"out_features": 0},
        {"backend": "unknown"},
    ],
)
def test_unsupported_configuration(change):
    arguments = {
        "in_features": 256,
        "out_features": 7,
        "vec_len": 8,
        "group_size": 128,
        "num_codebooks": 2,
        "nbits_per_codebook": 8,
        "backend": "cpu",
    }
    arguments.update(change)

    with pytest.raises((ValueError, NotImplementedError)):
        CodeGEMMLinear(**arguments)


def test_invalid_scale_shape_is_rejected():
    layer, _ = make_case()
    layer.scales = layer.scales[:1].contiguous()

    with pytest.raises(RuntimeError, match="scales"):
        layer(torch.ones(1, 1, 256))


def test_cpu_import_and_execution_without_cuda_extension():
    program = """
import sys
sys.modules["codegemm_kernel"] = None

import torch

def forbidden(*args, **kwargs):
    raise AssertionError("CPU execution attempted a CUDA operation")

torch.cuda._lazy_init = forbidden
torch.cuda.empty_cache = forbidden

from codegemm.inference.codegemm_causallm import CodeGEMMForCausalLM
from codegemm.inference.codegemm_linear import CodeGEMMLinear

layer = CodeGEMMLinear(
    128, 3, 8, 128, 2, 8, backend="cpu"
)
layer.codes.zero_()
layer.codebooks.fill_(1)
layer.scales.fill_(1)

x = torch.ones(1, 1, 128)
expected = torch.full((1, 1, 3), 256.0)

torch.testing.assert_close(layer(x), expected)
torch.testing.assert_close(layer(x), expected)
"""

    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    not os.environ.get("CODEGEMM_CPU_CHECKPOINT"),
    reason="Set CODEGEMM_CPU_CHECKPOINT to test a real checkpoint",
)
def test_real_checkpoint_prefill_and_generation():
    from codegemm.inference.codegemm_causallm import (
        CodeGEMMForCausalLM,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU model attempted a CUDA operation")

    with (
        patch.object(torch.cuda, "_lazy_init", forbidden),
        patch.object(torch.cuda, "empty_cache", forbidden),
    ):
        model = CodeGEMMForCausalLM.from_quantized(
            os.environ["CODEGEMM_CPU_CHECKPOINT"],
            backend="cpu",
        ).eval()

        for tensor in list(model.parameters()) + list(model.buffers()):
            assert tensor.device.type == "cpu"
            if tensor.is_floating_point():
                assert tensor.dtype == torch.float32

        seen_vector_counts = []
        compressed = next(
            module
            for module in model.modules()
            if isinstance(module, CodeGEMMLinear)
        )

        def record_vectors(module, inputs, output):
            seen_vector_counts.append(
                inputs[0].numel() // module.in_features
            )

        handle = compressed.register_forward_hook(record_vectors)

        try:
            input_ids = torch.tensor([[1, 42, 43]])

            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                min_new_tokens=2,
                max_new_tokens=2,
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
            )
        finally:
            handle.remove()

        assert output.shape == (1, 5)
        assert any(count > 1 for count in seen_vector_counts)
        assert 1 in seen_vector_counts