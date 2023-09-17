import deepspeed
import torch

from ds_inference_util import inject_ds_inference_module
def inject_ds_inference_module(model, enable_quantization, max_out_tokens=256):
    if enable_quantization:
        weight_quantization = {
            "quant":{
                "enabled": True,
                "qkv": {
                    "enabled": True,
                    "num_bits": 4
                },
                "attn_out": {
                    "enabled": True,
                    "num_bits": 4
                },
                "mlp1": {
                    "enabled": True,
                    "num_bits": 4
                },
            }
        }
    else:
        weight_quantization = {
            "quant":{
                "enabled": False
            }
        }
    deepspeed.init_inference(model, 
                        replace_with_kernel_inject=True, 
                        dtype=torch.int8 if enable_quantization else torch.half, 
                        return_tuple=False, 
                        mp_size=torch.distributed.get_world_size(),
                        max_out_tokens=max_out_tokens,
                        **weight_quantization,
                        )