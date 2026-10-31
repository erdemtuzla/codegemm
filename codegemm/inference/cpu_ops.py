import importlib

import torch

from codegemm.inference.cpu_backends import (
    canonical_backend,
    cpu_extension_name,
    validate_backend_platform,
)


# Each backend has its own binary and compiler flags. Cache by canonical name
# so selecting cpu-4 can never accidentally reuse the cpu-1 module.
_CPU_EXTENSIONS = {}
_OUTPUT_ROWS_PER_BLOCK = 128


def validate_config(
    in_features,
    out_features,
    vec_len,
    group_size,
    num_codebooks,
    nbits_per_codebook,
):
    metadata = (
        num_codebooks,
        nbits_per_codebook,
        vec_len,
        group_size,
    )

    if metadata != (2, 8, 8, 128):
        raise NotImplementedError(
            "CPU inference initially supports only "
            "m=2, b=8, v=8, g=128."
        )

    if in_features <= 0 or in_features % 128 != 0:
        raise ValueError(
            "CPU input features must be positive and divisible by 128."
        )

    if out_features <= 0:
        raise ValueError("CPU output features must be positive.")


def _cpu_extension(backend="cpu"):
    backend = canonical_backend(backend)
    validate_backend_platform(backend)
    if backend not in _CPU_EXTENSIONS:
        name = cpu_extension_name(backend)
        try:
            extension = importlib.import_module(name)
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f"{backend} requires {name}. Build the requested variant in "
                "codegemm/inference/cpu_kernel first."
            ) from exc
        # An older installation used -O2 for the baseline. Refuse an unlabelled
        # binary rather than report it as the new, deliberately unoptimized one.
        if not hasattr(extension, "build_info"):
            raise RuntimeError(
                f"{name} is an older build without variant metadata. Rebuild "
                "codegemm/inference/cpu_kernel before running CPU variants."
            )
        info = extension.build_info()
        if info.get("backend") != backend:
            raise RuntimeError(f"Requested {backend}, but {name} reports {info!r}; rebuild the extension")
        _CPU_EXTENSIONS[backend] = extension
    return _CPU_EXTENSIONS[backend]


def cpu_backend_info(backend="cpu"):
    """Expose the actual compiled configuration to benchmark reports."""
    return dict(_cpu_extension(backend).build_info())


@torch.no_grad()
def linear(x, codes, scales, codebooks, group_size, bias=None, *, backend="cpu"):
    if x.device.type != "cpu" or x.dtype != torch.float32:
        raise ValueError("CPU activations must be CPU float32 tensors.")

    if x.ndim < 1 or x.numel() == 0:
        raise ValueError("CPU input must contain at least one vector.")

    if codes.ndim != 3:
        raise ValueError("codes must have shape [2, K/32, M].")

    in_features = codes.shape[1] * 32
    out_features = codes.shape[2]

    if x.shape[-1] != in_features:
        raise ValueError(
            f"Expected {in_features} input features, "
            f"received {x.shape[-1]}."
        )

    if bias is not None:
        if (
            bias.device.type != "cpu"
            or bias.dtype != torch.float32
            or bias.shape != (out_features,)
        ):
            raise ValueError(
                "CPU bias must be a float32 tensor with shape [M]."
            )

    # Keep the existing tensor layouts and algorithm dispatch; only the chosen
    # C++ implementation changes between the four experiments.
    extension = _cpu_extension(backend)

    # These calls do not copy already-contiguous tensors.
    codes = codes.contiguous()
    scales = scales.contiguous()
    codebooks = codebooks.contiguous()

    flat = x.reshape(-1, in_features).contiguous()

    if flat.shape[0] == 1:
        output = extension.gemv(
            flat[0],
            codes,
            scales,
            codebooks,
            group_size,
        )
    else:
        output = torch.empty(
            (flat.shape[0], out_features),
            dtype=torch.float32,
            device="cpu",
        )

        for begin in range(0, out_features, _OUTPUT_ROWS_PER_BLOCK):
            end = min(begin + _OUTPUT_ROWS_PER_BLOCK, out_features)

            weight = extension.dequantize_block(
                codes,
                scales,
                codebooks,
                group_size,
                begin,
                end,
            )

            output[:, begin:end].copy_(flat @ weight.t())
            del weight

    output = output.reshape(x.shape[:-1] + (out_features,))

    if bias is not None:
        output.add_(bias)

    return output
