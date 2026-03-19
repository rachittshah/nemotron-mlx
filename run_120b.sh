#!/usr/bin/env bash
# Run Nemotron-3 Super 120B-A12B on M4 Pro 48GB via llama.cpp
#
# Uses: expert offloading, KV-cache quantization, flash attention,
# macOS GPU memory override. See FITTING_120B.md for details.
#
# Prerequisites:
#   brew install llama.cpp   (or build from source)
#   Download GGUF from bartowski or unsloth on HuggingFace

set -euo pipefail

# --- Configuration ---
MODEL_DIR="${HOME}/.cache/nemotron-120b"
# Pick your quant (uncomment one):
QUANT="IQ2_XXS"  # 50.16 GB — best quality that's close to fitting
# QUANT="IQ1_S"  # 46.38 GB — only one under 48GB, terrible quality
# QUANT="IQ2_XS" # 52.04 GB — slightly better than IQ2_XXS

BARTOWSKI_REPO="bartowski/nvidia_Nemotron-3-Super-120B-A12B-GGUF"
CONTEXT_LENGTH=8192
PORT=8080

# --- Functions ---
check_deps() {
    if ! command -v llama-server &>/dev/null; then
        echo "llama-server not found. Install:"
        echo "  brew install llama.cpp"
        echo "  # or build from source: https://github.com/ggml-org/llama.cpp"
        exit 1
    fi
    if ! command -v huggingface-cli &>/dev/null; then
        echo "huggingface-cli not found. Install:"
        echo "  pip install huggingface-hub"
        exit 1
    fi
}

download_model() {
    local model_file
    model_file=$(find "$MODEL_DIR" -name "*${QUANT}*" -type f 2>/dev/null | head -1)

    if [ -n "$model_file" ]; then
        echo "Model found: $model_file"
        echo "$model_file"
        return
    fi

    echo "Downloading ${BARTOWSKI_REPO} (${QUANT})..."
    echo "This may take a while (~50 GB)..."
    mkdir -p "$MODEL_DIR"

    huggingface-cli download "$BARTOWSKI_REPO" \
        --include "*${QUANT}*" \
        --local-dir "$MODEL_DIR"

    model_file=$(find "$MODEL_DIR" -name "*${QUANT}*" -type f | head -1)
    echo "$model_file"
}

set_gpu_memory() {
    local current
    current=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")

    if [ "$current" -lt 44000 ]; then
        echo "Setting GPU memory limit to 44 GB (requires sudo)..."
        echo "  Current: ${current} MB"
        sudo sysctl iogpu.wired_limit_mb=45056
        echo "  New: 45056 MB (44 GB)"
    else
        echo "GPU memory already set to ${current} MB"
    fi
}

print_memory_budget() {
    local model_size_gb=$1
    echo ""
    echo "========================================"
    echo " Memory Budget: Nemotron-3 Super 120B"
    echo "========================================"
    echo " System RAM:           48.0 GB"
    echo " GPU allocation:       44.0 GB (after override)"
    echo " Model (${QUANT}):     ${model_size_gb} GB"
    echo " KV-cache (${CONTEXT_LENGTH}, q8/q4):  ~1.5 GB"
    echo " Compute buffers:      ~2.0 GB"
    echo "----------------------------------------"
    echo " Strategy: Expert offload to CPU"
    echo "   Metal GPU: attention + shared (~12 GB)"
    echo "   CPU/mmap:  expert weights (~${model_size_gb} GB)"
    echo "   Hot experts stay in RAM"
    echo "   Cold experts paged to NVMe (~7.4 GB/s)"
    echo "========================================"
    echo ""
}

run_server() {
    local model_file=$1

    echo "Starting llama-server..."
    echo "  Model:   $(basename "$model_file")"
    echo "  Context: ${CONTEXT_LENGTH}"
    echo "  Port:    ${PORT}"
    echo "  Expert offload: CPU"
    echo "  KV-cache: k=q8_0, v=q4_0"
    echo ""
    echo "API endpoint: http://localhost:${PORT}/v1"
    echo "Press Ctrl+C to stop"
    echo ""

    llama-server \
        -m "$model_file" \
        -ngl 99 \
        --cpu-moe \
        --cache-type-k q8_0 \
        --cache-type-v q4_0 \
        -fa on \
        --single-turn \
        -c "$CONTEXT_LENGTH" \
        -b 4096 -ub 4096 \
        --host 0.0.0.0 \
        --port "$PORT"
}

run_cli() {
    local model_file=$1
    local prompt="${2:-Explain mixture of experts architecture in 3 sentences.}"

    echo "Running single inference..."
    echo "Prompt: ${prompt}"
    echo ""

    llama-cli \
        -m "$model_file" \
        -ngl 99 \
        --cpu-moe \
        --cache-type-k q8_0 \
        --cache-type-v q4_0 \
        -fa on \
        --single-turn \
        -c "$CONTEXT_LENGTH" \
        -n 256 \
        -p "$prompt"
}

# --- Main ---
main() {
    local mode="${1:-server}"
    local prompt="${2:-}"

    check_deps

    echo "Nemotron-3 Super 120B-A12B on M4 Pro 48GB"
    echo "Quant: ${QUANT}"
    echo ""

    set_gpu_memory

    local model_file
    model_file=$(download_model)

    if [ -z "$model_file" ]; then
        echo "ERROR: Could not find or download model file"
        exit 1
    fi

    # Get file size
    local size_bytes
    size_bytes=$(stat -f%z "$model_file" 2>/dev/null || stat -c%s "$model_file" 2>/dev/null || echo "0")
    local size_gb
    size_gb=$(echo "scale=1; $size_bytes / 1073741824" | bc 2>/dev/null || echo "?")

    print_memory_budget "$size_gb"

    case "$mode" in
        server)
            run_server "$model_file"
            ;;
        cli)
            run_cli "$model_file" "$prompt"
            ;;
        *)
            echo "Usage: $0 [server|cli] [prompt]"
            exit 1
            ;;
    esac
}

main "$@"
