
NUM_GPUS=$1
MODEL_PATH=$2
TOKENIZER_PATH=$3

deepspeed --num_nodes 1 --num_gpus $NUM_GPUS example_text_completion.py \
    --ckpt_dir=$MODEL_PATH \
    --tokenizer_path=$TOKENIZER_PATH \
    --test_for_performance=true \
    #--use_deepspeed_inference=true