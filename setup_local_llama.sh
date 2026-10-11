#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CACHE_ROOT=${LUCY_CACHE_DIR:-"$HOME/.cache/lucy"}
MODEL_DIR="$CACHE_ROOT/models"
VENV_DIR="$CACHE_ROOT/venv"
MODEL_NAME="Qwen3-0.6B-Q8_0.gguf"
MODEL_PATH=${LUCY_MODEL_PATH:-"$MODEL_DIR/$MODEL_NAME"}
MODEL_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/$MODEL_NAME?download=true"
EXPECTED_MODEL_BYTES=639446688
EXPECTED_MODEL_SHA256="9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_PATH" ]; then
    if [ -n "${LUCY_MODEL_PATH:-}" ]; then
        printf 'Configured model file does not exist: %s\n' "$MODEL_PATH" >&2
        exit 1
    fi

    temporary_model=$(mktemp "$MODEL_PATH.XXXXXX")
    trap 'rm -f "$temporary_model"' EXIT HUP INT TERM
    printf 'Downloading the 0.6B local model to %s\n' "$MODEL_PATH"
    curl --fail --location --retry 3 --output "$temporary_model" "$MODEL_URL"
    actual_model_bytes=$(wc -c < "$temporary_model" | tr -d '[:space:]')
    if [ "$actual_model_bytes" -ne "$EXPECTED_MODEL_BYTES" ]; then
        printf 'Downloaded model has unexpected size: %s bytes\n' "$actual_model_bytes" >&2
        exit 1
    fi
    mv "$temporary_model" "$MODEL_PATH"
    trap - EXIT HUP INT TERM
fi

if [ -z "${LUCY_MODEL_PATH:-}" ]; then
    actual_model_bytes=$(wc -c < "$MODEL_PATH" | tr -d '[:space:]')
    actual_model_sha256=$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')
    if [ "$actual_model_bytes" -ne "$EXPECTED_MODEL_BYTES" ] \
        || [ "$actual_model_sha256" != "$EXPECTED_MODEL_SHA256" ]; then
        printf 'Cached model failed its size or SHA-256 integrity check: %s\n' "$MODEL_PATH" >&2
        exit 1
    fi
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    mkdir -p "$CACHE_ROOT"
    python3 -m venv "$VENV_DIR"
fi

if ! command -v cmake >/dev/null 2>&1; then
    "$VENV_DIR/bin/python" -m pip install cmake
    PATH="$VENV_DIR/bin:$PATH"
    export PATH
fi

if ! "$VENV_DIR/bin/python" -c 'import llama_cpp' >/dev/null 2>&1; then
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        CMAKE_ARGS="-DGGML_METAL=on" \
            "$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements-llama-cpp.txt"
    else
        "$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements-llama-cpp.txt"
    fi
fi

cd "$PROJECT_ROOT"
export LUCY_MODEL_PATH="$MODEL_PATH"
export LUCY_MODEL_BACKEND=llama-cpp
unset LUCY_API_BASE_URL LUCY_API_KEY LUCY_API_MODEL LUCY_CODING_API_MODEL
exec "$VENV_DIR/bin/python" "$PROJECT_ROOT/main.py" "$@"
