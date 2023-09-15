import torch
from collections import OrderedDict
import gc
def merge_checkpints(ckpt_sd_list, dim=8192, vocab_size=32000):
    keys = list(ckpt_sd_list[0].keys())
    new_sd = OrderedDict()
    for key in keys:
        data = [sd[key] for sd in ckpt_sd_list]
        if data[0].shape[0] != dim and data[0].shape[0] != vocab_size:
            data = torch.cat(data, dim=0)
        elif len(data[0].shape) > 1:
            data = torch.cat(data, dim=-1)
        else:
            data = data[0]
        new_sd[key] = data.cpu()
    for sd in ckpt_sd_list:
        del sd
    ckpt_sd_list = None
    gc.collect()
    return new_sd
