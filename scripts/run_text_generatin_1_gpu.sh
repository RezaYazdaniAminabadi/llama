
NUM_GPUS=$1
MODEL_PATH=$2
TOKENIZER_PATH=$3

deepspeed --num_nodes 1 --num_gpus 1 example_text_completion.py \
    --ckpt_dir=$MODEL_PATH \
    --tokenizer_path=$TOKENIZER_PATH \
    --checkpoint_device='cpu' \
    --use_cpu_initialization=true \
    --use_deepspeed_inference=true \
    --enable-quantization=true
