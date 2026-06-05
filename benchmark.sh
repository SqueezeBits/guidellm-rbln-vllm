#! /bin/bash
# Default values
host="127.0.0.1"
port=30001
MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
TP_SIZE=4
MAX_SEQ_LEN=4096
BLOCK_SIZE=4096
LENGTH=2048
MAX_NUM_SEQS=8
PLATFORM="torch_compile"
DURATION=1800 # 30 minutes

CHUNK_SIZE=128
# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -p, --platform <name>      Platform to use: 'torch_compile' or 'optimum' (default: $PLATFORM)"
    echo "  -m, --model-id <name>      Model ID (default: $MODEL_ID)"
    echo "  -t, --tp-size <num>        Tensor Parallel size (default: $TP_SIZE)"
    echo "  -s, --max-seq-len <num>    Max sequence length (default: $MAX_SEQ_LEN)"
    echo "  -b, --block-size <num>     Block size (default: $BLOCK_SIZE)"
    echo "  -n, --max-num-seqs <num>   Max number of sequences (default: $MAX_NUM_SEQS)"
    echo "  -H, --host <ip>            Host IP (default: $host)"
    echo "  -P, --port <num>           Port number (default: $port)"
    echo "  -l, --length <num>         Length of the prompt and output tokens (default: $LENGTH)"
    echo "  -d, --duration <num>       Duration of the single benchmark in seconds (default: $DURATION)"
    echo "  -h, --help                 Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -m|--model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        -t|--tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        -s|--max-seq-len)
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        -b|--block-size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        -n|--max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        -H|--host)
            host="$2"
            shift 2
            ;;
        -P|--port)
            port="$2"
            shift 2
            ;;
        -l|--length)
            LENGTH="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

MODEL_NAME=${MODEL_ID##*/}

# if platform is not in ['torch_compile', 'optimum'], exit
if [ "$PLATFORM" != "torch_compile" ] && [ "$PLATFORM" != "optimum" ]; then
    echo "Invalid platform: $PLATFORM, use 'torch_compile' or 'optimum' as platform"
    exit 1
fi

# Function to check if server is ready
check_server_health() {
    local max_attempts=180  # 30 minutes timeout
    local attempt=1
    
    echo "Checking server health..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://${host}:${port}/v1/models > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting..."
        sleep 10
        ((attempt++))
    done
    
    echo "Server failed to start within timeout"
    return 1
}

# Function to cleanup server
cleanup_server() {
    echo "Shutting down server..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 
    fi
    # Also kill any remaining vllm processes
    pkill -f "vllm serve"

    echo "Waiting for server to shutdown..."
    if [ ! -z "$SERVER_PID" ]; then
        wait $SERVER_PID
    fi
    sleep 10
    echo "Server shutdown completed"
}


output_dir="results/greedy_sample/${MODEL_NAME}/${PLATFORM}-tp_${TP_SIZE}-mb_${MAX_NUM_SEQS}-seq_len_${MAX_SEQ_LEN}-block_${BLOCK_SIZE}-io_len_${LENGTH}"
mkdir -p ${output_dir}


if [ "$PLATFORM" == "optimum" ]; then
    if [ $MAX_SEQ_LEN -eq $BLOCK_SIZE ]; then
        attn_impl="eager"
    else
        attn_impl="flash_attn"
    fi
    USE_VLLM_MODEL=0
    OPTIMUM_RBLN_MODEL_PATH="model_optimum-${MODEL_NAME}-tp_${TP_SIZE}-mb_${MAX_NUM_SEQS}-attn_${attn_impl}-seq_len_${MAX_SEQ_LEN}-block_${BLOCK_SIZE}"
    echo "Compiling optimum model ${MODEL_ID} with the following arguments: ${OPTIMUM_RBLN_MODEL_PATH}"
    python3 compile_optimum_rbln.py \
        $MODEL_ID \
        --output-dir $OPTIMUM_RBLN_MODEL_PATH \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --max-model-len ${MAX_SEQ_LEN} \
        --block-size ${BLOCK_SIZE} \
        --enable-chunked-prefill \
        --max-num-batched-tokens ${CHUNK_SIZE} \
        --tensor-parallel-size ${TP_SIZE} \
        --attn-impl ${attn_impl} >& ${output_dir}/compile_optimum_rbln.log
    echo "Compiled model saved to ${OPTIMUM_RBLN_MODEL_PATH}"
    MODEL_PATH=$OPTIMUM_RBLN_MODEL_PATH
else
    USE_VLLM_MODEL=1
    MODEL_PATH=$MODEL_ID
fi



echo "Starting server with max_num_sequences: ${MAX_NUM_SEQS} "
RBLN_USE_CUSTOM_KERNEL=1 VLLM_RBLN_SAMPLER=0 VLLM_RBLN_TP_SIZE=${TP_SIZE} VLLM_RBLN_USE_VLLM_MODEL=${USE_VLLM_MODEL} VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_V1=1 \
    vllm serve $MODEL_PATH \
    --tokenizer $MODEL_ID \
    --host ${host} \
    --port ${port} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-model-len ${MAX_SEQ_LEN} \
    --block-size ${BLOCK_SIZE} \
    --enable-chunked-prefill \
    --max-num-batched-tokens ${CHUNK_SIZE} \
    >& ${output_dir}/vllm_server.log &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for server to be ready
if ! check_server_health; then
    echo "Failed to start server, skipping this configuration"
    cleanup_server
    continue
fi

echo "Starting benchmark with duration: ${DURATION} seconds"
guidellm benchmark \
    --request-type  text_completions \
    --profile concurrent  --rate ${MAX_NUM_SEQS} \
    --backend-args "{\"extras\":{\"body\":{\"temperature\":0.0}}, \"timeout\": ${DURATION}}" \
    --data "prompt_tokens=${LENGTH},output_tokens=${LENGTH},prompt_tokens_min=${LENGTH},prompt_tokens_max=${LENGTH},output_tokens_min=${LENGTH},output_tokens_max=${LENGTH}" \
    --model $MODEL_PATH \
    --target http://${host}:${port} \
    --max-seconds ${DURATION} \
    --warmup 1  --cooldown 1 \
    --output-dir ${output_dir} \
    >& ${output_dir}/guidellm_benchmark.log

if [ "$PLATFORM" == "optimum" ]; then
    rm -rf $OPTIMUM_RBLN_MODEL_PATH
    echo "Removed optimum model ${OPTIMUM_RBLN_MODEL_PATH}"
fi

cleanup_server
SERVER_PID=""
echo "Waiting a bit before next configuration..."
