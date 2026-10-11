# Lucy Local-Model Wiki: llama.cpp

This wiki describes Lucy's direct, local GGUF inference path. It explains what
gets installed, how model selection works, what data stays local, and how to
diagnose common setup and conversation issues.

## At a glance

Lucy has three ways to answer:

1. **Knowledge-only:** no text-generation model is loaded. Lucy can handle
   built-in commands, simple deterministic questions, and matching local
   references; this mode does not generate an open-ended conversation.
2. **Direct local inference:** Lucy loads a `.gguf` model into the Python
   process using `llama-cpp-python` or GPT4All. No Ollama server is involved.
3. **OpenAI-compatible API:** Lucy sends chat requests to the explicitly
   configured API. This can be a local Ollama server or a remote service.

For macOS, [`setup_local_llama.sh`](../setup_local_llama.sh) installs the
llama.cpp Python binding in a cache-only virtual environment, downloads a
compact Qwen3 GGUF model into the same cache, and starts Lucy with the local
backend selected. The model weights and installed packages are not committed
to the repository.

## What the setup script installs

By default, the script uses:

| Item | Location / value |
| --- | --- |
| Model | `~/.cache/lucy/models/Qwen3-0.6B-Q8_0.gguf` |
| Python environment | `~/.cache/lucy/venv` |
| Runtime | `llama-cpp-python==0.3.16` |
| Model source | [Qwen/Qwen3-0.6B-GGUF on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) |
| Model size | 639,446,688 bytes (about 639 MB) |

The selected quantization is Q8_0 and fits under Lucy's 900,000,000-byte
per-model limit. The script checks the downloaded file's expected size and
SHA-256 before moving it into the cache. It downloads only if the default file
is absent. On Apple Silicon, the runtime is built with Metal enabled; on Intel
Macs it uses CPU inference.

The first run needs internet access for the model and Python package download,
plus sufficient build tools for `llama-cpp-python`. Once both are cached,
starting Lucy uses the local files and does not need Ollama or an API service.
The direct model path does not transmit prompts to Hugging Face; Hugging Face
is used only to fetch model files during setup.

## Setup and normal use

From the Lucy project directory:

```sh
./setup_local_llama.sh
```

The script creates the isolated environment and downloads the model on its
first run, then starts the interactive Lucy CLI. On later runs, use the same
command to start Lucy again. Exit with `/quit` or `Ctrl+C`.

The script deliberately sets `LUCY_MODEL_BACKEND=llama-cpp` and clears API
endpoint/model variables for that process. This prevents an old
`LUCY_API_BASE_URL=http://localhost:11434/v1` in the terminal from sending
messages to Ollama instead. It does not change the parent shell's environment.

To use a different existing GGUF file, pass its path to the script:

```sh
LUCY_MODEL_PATH="$HOME/Models/my-chat-model.gguf" ./setup_local_llama.sh
```

The custom file must already exist and remain at or below 900,000,000 bytes.
The script does not download a replacement when an explicit path is invalid.
The custom model is loaded using the same isolated llama.cpp installation.

To use a different cache root, set `LUCY_CACHE_DIR`; by default, the standard
user cache directory is used:

```sh
LUCY_CACHE_DIR="$HOME/Library/Caches/lucy" ./setup_local_llama.sh
```

`LUCY_MODEL_CONTEXT` controls the llama.cpp context size, from 512 through
32,768 tokens. It defaults to 4,096. Larger contexts allow more conversation
history but use more memory; lower values reduce that cost. The model itself
must also fit in available system memory.

## How model selection works

`Lucy.from_environment()` chooses the backend in this order:

1. `LUCY_MODEL_PATH`, when explicitly set, identifies a direct local model.
2. With no API configuration or explicit model path, Lucy uses the default
   cached Qwen GGUF if it exists.
3. If no local GGUF is configured, Lucy may detect supported small models in a
   running Ollama service as a backward-compatible optional path.
4. If none is available, Lucy starts in knowledge-only mode.

An explicit `LUCY_API_BASE_URL` selects the API route rather than local
inference. Use the setup script when the intent is specifically to run the
local llama.cpp model; it clears API settings for the launched process.

With `LUCY_MODEL_BACKEND=auto`, macOS prefers llama.cpp when installed, then
GPT4All. Other platforms prefer GPT4All, then llama.cpp. The setup script
selects `llama-cpp` explicitly. Lucy rejects missing model files, unsupported
backend selections, and model files above the 900 MB limit with a startup
error instead of silently changing backends.

## What happens during a chat turn

1. The CLI reads the user's message and calls `Lucy.respond()`.
2. Lucy handles commands and deterministic cases directly, then searches its
   local references for relevant context.
3. If a local chat-capable model is loaded, Lucy sends a prompt and recent
   conversation context to llama.cpp's in-process chat-completion API.
   Qwen-style `<think>` reasoning blocks are removed before the answer is
   displayed.
4. Lucy displays the generated answer and records the conversation in its
   local memory file.

The history sent to a local model remains on the machine. The generation
runtime and model execute inside Lucy's Python process; there is no local HTTP
server and no Ollama dependency. Explicit web-search/read commands still need
internet access, and using those tools intentionally contacts the requested
public site.

Lucy constrains model tool use to the functions it defines, such as arithmetic,
local-reference lookup, Wikipedia summaries, and saved-memory recall. It does
not give the model arbitrary shell execution. llama.cpp function calling
depends on the GGUF model's chat template; ordinary chat does not. Slash
commands remain available regardless of function-calling support. Per turn,
Lucy limits tool activity to five rounds and six calls, and blocks repeated
identical calls.

## Memory and privacy

Conversation history and explicitly saved facts are stored in Lucy's local
memory JSON file (by default, `data/lucy_memory.json`). A name mentioned in
conversation is session context unless the user explicitly asks Lucy to save
it. `/clear` clears conversation history and unsaved corrections but retains
saved facts.

Direct llama.cpp inference does not send prompts or saved memory to a remote
chat API. Model downloads contact the configured Hugging Face URL. Explicit
remote API configuration behaves differently: the prompt and relevant context
are sent to that provider. See the main [README](../README.md) before choosing
a remote endpoint.

## Troubleshooting

### Lucy says it is in knowledge-only mode

The GGUF model or llama.cpp runtime has not been configured for the Python
process that launched Lucy. Run `./setup_local_llama.sh`; do not expect plain
`python3 main.py` to find a model in a separate virtual environment. The setup
script launches Lucy with its own isolated environment.

### `Could not load the llama-cpp model`

Check that the GGUF file is complete, readable, compatible with llama.cpp, and
under the 900 MB limit. For the standard setup, rerun the script after
removing or renaming a corrupt cached model file; the script will download the
standard file if it is absent. It does not replace an existing file silently.

### `llama-cpp-python` installation fails

The package includes native C++ components. Confirm that Python 3.9+ and a
working compiler are available. On macOS, install the Xcode Command Line Tools
if the compiler is missing. Apple Silicon builds enable Metal; Intel Mac builds
use CPU. If a Python package or build tool is missing, the setup script reports
the installation error rather than launching in a pretend generated-chat mode.

### Replies are slow or low quality

The default Qwen model is intentionally small to fit the local model-size cap.
Small models can be less capable than larger hosted models. Intel Macs use CPU
inference and may respond more slowly. Keep the context size modest, close
memory-intensive applications, or point Lucy at another compatible GGUF model
that fits the 900 MB cap.

### Messages still go to Ollama or another API

Start Lucy with `./setup_local_llama.sh`. The script forces the llama.cpp
backend and unsets API endpoint/model variables for its child process. For
manual launch, remove `LUCY_API_BASE_URL` and set both `LUCY_MODEL_PATH` and
`LUCY_MODEL_BACKEND=llama-cpp`.

### Need to confirm the selected route

Run `/status` inside Lucy for a summary of active capabilities and
configuration. To make a deterministic local-only launch, use the setup script
rather than relying on automatic Ollama discovery.

## Related files

- [`core/ai_engine.py`](../core/ai_engine.py) — runtime discovery, model
  loading, prompt assembly, tool limits, and response routing.
- [`main.py`](../main.py) — CLI startup, error reporting, and input loop.
- [`requirements-llama-cpp.txt`](../requirements-llama-cpp.txt) — pinned
  llama.cpp Python package.
- [`setup_local_llama.sh`](../setup_local_llama.sh) — local download, runtime
  setup, and launcher.
