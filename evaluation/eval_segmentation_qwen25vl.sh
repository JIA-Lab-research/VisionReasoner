#!/bin/bash
set -e

MODEL_TYPE="qwen25vl"  # Model type: qwen2vl or vision_reasoner or qwen25vl
TEST_DATA_PATH=${1:-"Ricky06662/ReasonSeg_val"}
MODEL_PATH=${2:-"Qwen/Qwen2.5-VL-7B-Instruct"}

# Extract model name and test dataset name for output directory
TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="./detection_eval_results/${MODEL_TYPE}/${TEST_NAME}"

# Customize GPU array here - specify which GPUs to use
GPU_ARRAY=(0 1 2 3 4 5 6 7)  # Example: using GPUs 0, 1, 2, 3
NUM_PARTS=${#GPU_ARRAY[@]}

# Create output directory
mkdir -p $OUTPUT_PATH

# Run processes in parallel
for i in $(seq 0 $((NUM_PARTS-1))); do
    gpu_id=${GPU_ARRAY[$i]}
    process_idx=$i  # 0-based indexing for process
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    (
        python evaluation/evaluation_segmentation.py \
            --model $MODEL_TYPE \
            --model_path $MODEL_PATH \
            --output_path $OUTPUT_PATH \
            --test_data_path $TEST_DATA_PATH \
            --idx $process_idx \
            --num_parts $NUM_PARTS \
            --batch_size 16 || { echo "1" > /tmp/process_status.$$; kill -TERM -$$; }
    ) &
done

# Wait for all processes to complete
wait

python evaluation/calculate_iou_with_bbox.py --output_dir $OUTPUT_PATH