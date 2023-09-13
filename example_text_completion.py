# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
import time 
from llama import Llama
from typing import List

import deepspeed

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    test_for_performance: bool = False,
    use_deepspeed_inference: bool = False,
    checkpoint_device: str = 'cpu'
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        use_deepspeed_inference=use_deepspeed_inference,
        checkpoint_device=f'cuda:{torch.distributed.get_rank()}' if checkpoint_device == 'cuda' else 'cpu'
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    if test_for_performance:
        # warmup
        for _ in range(3):
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
        torch.cuda.synchronize()
        t1 = time.time()
        baseline_time = t1 - t0
        
        deepspeed.init_inference(generator.model, 
                                replace_with_kernel_inject=True, 
                                dtype=torch.half, 
                                return_tuple=False, 
                                mp_size=torch.distributed.get_world_size(),
                                )
        # warmup
        for _ in range(3):
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
        torch.cuda.synchronize()
        t1 = time.time()
        ds_time = t1 - t0
        print(f'baseline: generation time {baseline_time / 10:.3f} sec for generating x tokens.')
        print(f'ds-inference: generation time {ds_time / 10:.3f} sec for generating x tokens.')
        print(f'speeup: {baseline_time / ds_time:.3f}x')
    else:
        for _ in range(1):
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
        for prompt, result in zip(prompts, results):
            print(prompt)
            print(f"> {result['generation']}")
            print("\n==================================\n")
    exit()

if __name__ == "__main__":
    fire.Fire(main)
