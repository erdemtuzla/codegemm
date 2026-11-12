# Originally from https://github.com/SNU-ARC/any-precision-llm/blob/main/any_precision/modules/AnyPrecisionForCausalLM.py

import gc
import torch
import os
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)
from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

from codegemm.inference.codegemm_linear import CodeGEMMLinear
from codegemm.inference.cpu_backends import (
    canonical_backend,
    is_cpu_backend,
    validate_backend_platform,
)
from codegemm.utils.analyzer import get_analyzer


def replace_module_by_name(layer, module_name, new_module):
    levels = module_name.split('.')
    module = layer
    for level in levels[:-1]:
        module = getattr(module, level) if not level.isdigit() else module[int(level)]
    setattr(module, levels[-1], new_module)


def get_num_layers(config) -> int:
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2":
            return config.num_hidden_layers
        case unknown_type:
            raise NotImplementedError(f"Can't get number of layers for {unknown_type}")


def get_layers_prefix(config) -> str:
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2":
            return "model.layers"
        case unknown_type:
            raise NotImplementedError(f"Can't get layers prefix for {unknown_type}")


def swap_quant_model(
    model, 
    config, 
    in_path,
    metadata: dict = None,
    layers_to_replace: list = None,
):

    def _get_submodules(model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)

    with torch.no_grad():
        from transformers.pytorch_utils import Conv1D
        from safetensors.torch import load_file
        quantized_ckpt = load_file(os.path.join(in_path, "model.safetensors"))
        for name, layer in model.named_modules():
            if (isinstance(layer, nn.Linear) and 
                'lm_head' not in name and 
                'project' not in name
            ):
                print(f'Swap {name} linear with CodeGEMMLinear')
                parent, target, target_name = _get_submodules(model, name)
                
                quantized_layer_scales = quantized_ckpt[name+'.scales']
                quantized_layer_codes = quantized_ckpt[name+'.codes']
                quantized_layer_codebooks = quantized_ckpt[name+'.codebooks']
                in_features = quantized_layer_codebooks.shape[3] * quantized_layer_codes.shape[1] * 4
                out_features = quantized_layer_codebooks.shape[2] * quantized_layer_codes.shape[2]
                new_module = CodeGEMMLinear(
                    in_features=in_features,
                    out_features=out_features,
                    in_group_size=quantized_layer_codebooks.shape[3],
                    scale_group_size=128,
                    out_group_size=quantized_layer_codebooks.shape[2],
                    num_codebooks=quantized_layer_codebooks.shape[0],
                    nbits_per_codebook=int(torch.log2(torch.tensor(quantized_layer_codebooks.shape[1])).item()),
                    device="cuda",
                    dtype=model.dtype,
                )
                new_module.scales = torch.nn.Parameter(quantized_layer_scales, requires_grad=False)
                new_module.codes = torch.nn.Parameter(quantized_layer_codes, requires_grad=False)
                new_module.codebooks = torch.nn.Parameter(quantized_layer_codebooks, requires_grad=False)
                _replace_module(parent, target_name, new_module, target)
                torch.cuda.empty_cache()

def _avoid_onednn_for_single_token(linear):
    # The FP32 lm_head stays unquantized and uses PyTorch matmul. On AArch64,
    # oneDNN is ~5x slower than the plain path for one vector times a large
    # matrix, but faster for multi-token prompts, so only bypass it per token.
    original_forward = linear.forward

    def forward(x):
        if x.numel() == linear.in_features:
            with torch.backends.mkldnn.flags(enabled=False):
                return original_forward(x)
        return original_forward(x)

    linear.forward = forward


class CodeGEMMForCausalLM(nn.Module):
    def __init__(
            self,
            model_path,
            config,
            torch_dtype=torch.float16,
            fuse_layers=False,
            trust_remote_code=True,
            new_vocab_size=None,
            backend="cuda",
    ):
        super().__init__()

        # All CPU variants share loading, dtype, and framework operations; only
        # compressed linear computation differs. Leave global thread pools alone
        # here so loading one model cannot silently reconfigure another model.
        backend = canonical_backend(backend)
        validate_backend_platform(backend)

        if is_cpu_backend(backend) and fuse_layers:
            raise NotImplementedError(
                "CPU inference does not support layer fusion."
            )

        self.backend = backend

        if is_cpu_backend(backend):
            torch_dtype = torch.float32

        self.inference_dtype = torch_dtype

        self.config = config
        self.group_size = self.config.codegemm['group_size']
        self.vec_len = self.config.codegemm['vec_len']
        self.num_codebooks = self.config.codegemm['num_codebooks']
        self.nbits_per_codebook = self.config.codegemm['nbits_per_codebook']

        if is_cpu_backend(self.backend):
            metadata = (
                self.num_codebooks,
                self.nbits_per_codebook,
                self.vec_len,
                self.group_size,
            )

            if metadata != (2, 8, 8, 128):
                raise NotImplementedError(
                    "CPU inference initially supports only "
                    "m=2, b=8, v=8, g=128."
                )
            
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }

        if is_cpu_backend(self.backend):
            model_kwargs["attn_implementation"] = "eager"

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                config=config,
                **model_kwargs,
            )

        self.analyzer = get_analyzer(self.model)

        self._load_quantized_modules()

        self.tie_weights()

        if is_cpu_backend(self.backend):
            device_map = {"": "cpu"}
        else:
            device_map = {
                key: "cpu"
                for key in self.model.state_dict().keys()
            }

        self.model.resize_token_embeddings(new_vocab_size)


        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            self.model,
            checkpoint=model_path,
            device_map=device_map,
            dtype=torch_dtype,
        )

        if is_cpu_backend(self.backend):
            _avoid_onednn_for_single_token(self.model.lm_head)

        # Dispath to devices
        if fuse_layers:
            self.fuse_layers()

        # self.prune_precisions()

    def forward(self, *args, **kwargs):
        results = self.model.forward(*args, **kwargs)
        return results

    def generate(self, *args, **kwargs):
        with torch.inference_mode():
            results = self.model.generate(*args, **kwargs)
        return results

    @staticmethod
    def _load_config(
            model_path,
            trust_remote_code=True,
    ):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return config

    @classmethod
    def from_quantized(
            cls,
            quant_model_path,
            trust_remote_code=True,
            fuse_layers=False,
            precisions=None,
            new_vocab_size=None,
            backend="cuda",
    ):
        # Reject an unavailable ISA before reading or allocating a checkpoint.
        backend = canonical_backend(backend)
        validate_backend_platform(backend)
        config = cls._load_config(quant_model_path, trust_remote_code)

        codegemm_model = cls(
            model_path=quant_model_path,
            config=config,
            fuse_layers=fuse_layers,
            trust_remote_code=trust_remote_code,
            new_vocab_size=new_vocab_size,
            backend=backend,
        )

        return codegemm_model

    def _load_quantized_modules(self):
        # Get blocks of model
        layers = self.analyzer.get_layers()

        for layer in tqdm(layers, desc="Loading CodeGEMM Layers"):
            # Get every linear layer in a block
            named_linears = self.analyzer.get_modules(layer)

            for name, module in named_linears.items():
                wqlinear = CodeGEMMLinear(
                    module.in_features, module.out_features,
                    self.vec_len,
                    self.group_size,
                    self.num_codebooks,
                    self.nbits_per_codebook,
                    bias=module.bias is not None,
                    dtype=self.inference_dtype,
                    backend=self.backend,
                )
                replace_module_by_name(layer, name, wqlinear)

            if self.backend == "cuda":
                torch.cuda.empty_cache()
                
            gc.collect()

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def get_model_layers(self):
        module = self.model
        for attrib_name in self.config.codegemm['arch_config']['model_name'].split('.'):
            module = getattr(module, attrib_name)
        return getattr(module, self.config.codegemm['arch_config']['layers_name'])

    def fuse_layers(self):
        if 'fuse_target_layers' not in self.model_config:
            raise NotImplementedError("This model does not support layer fusion")
        pass

    @property
    def layer_type(self):
        for layer in self.get_model_layers():
            layer_class_name = layer.__class__.__name__
            if layer_class_name.endswith("DecoderLayer"):
                return layer_class_name
        return None

    @property
    def device(self):
        return self.model.device
