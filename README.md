# Lucy

Lucy is a text-based assistant with persistent conversation context, local
knowledge lookup, safe arithmetic, current date and time answers, explicit
Wikipedia lookups, and optional speech. Common identity, date, and time
questions are answered directly by Lucy rather than delegated to the language
model. For generated conversations, it supports either a local GPT4All model or
an OpenAI-compatible chat API (including a local Ollama server).

## Run Lucy

Python 3.9 or newer is required. From the project directory, run:

```sh
python3 main.py
```

### macOS setup

From Terminal, create an isolated environment and run Lucy:

```sh
cd /path/to/Lucy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python main.py
```

The text-only app uses Python's standard library, so you do not need to install
the full `requirements.txt` to run it. macOS spoken replies use the built-in
`say` command; enable them in the same Terminal session with:

```sh
LUCY_VOICE=1 python main.py
```

Lucy runs in knowledge-only mode without extra packages or a model. In that
mode, it answers simple built-in questions and questions covered by the local
knowledge base; it does not pretend to generate answers to other questions.
Lucy can run a GGUF chat model directly on the computer through GPT4All or
llama.cpp; an Ollama server is optional. Lucy's local-model size limit is
900 MB per model, including GPT4All files.

### Run Lucy with llama.cpp (no Ollama)

On macOS, use the setup-and-run script to create an isolated Python environment,
install llama.cpp (with Metal acceleration on Apple Silicon), and download the
Qwen3 0.6B Q8 GGUF model (about 639 MB) to `~/.cache/lucy`. The model weights are
not stored in the repository. The selected model is under Lucy's 900 MB limit.

```sh
cd /path/to/Lucy
./setup_local_llama.sh
```

The script caches model weights and its virtual environment outside the
repository. Run it again to start Lucy later. It uses llama.cpp directly and
does not start or connect to Ollama. It ignores exported API settings so an
old Ollama URL cannot override the local backend. If a local model is already downloaded,
provide its path with `LUCY_MODEL_PATH=/path/to/model.gguf` when running the
script; otherwise the default cached model is used automatically when launching
Lucy from an environment with llama.cpp installed.
`llama-cpp-python` loads GGUF files directly. Set `LUCY_MODEL_CONTEXT` to choose
a context size from 512 to 32,768 tokens (default 4,096); a larger context
needs more memory.

For Intel Macs, the script installs CPU inference without Metal. To install
llama.cpp manually, use the optional requirements file in a virtual
environment, adding `CMAKE_ARGS="-DGGML_METAL=on"` on Apple Silicon. Set
`LUCY_MODEL_BACKEND=auto` to let Lucy choose an installed local runtime, or
`gpt4all` to select GPT4All explicitly.
Lucy can send its restricted tool definitions to llama.cpp models that support
function calling in their GGUF chat template. GPT4All still supports normal
local chat without structured model tool calls; Lucy's explicit slash commands
remain available either way.
See the [local llama.cpp wiki](docs/LOCAL-LLAMA-WIKI.md) for architecture,
configuration, privacy boundaries, and troubleshooting.

## Generated conversations with Ollama

Ollama is an optional local-server backend. Install and launch the Ollama macOS app from
[ollama.com/download](https://ollama.com/download). Lucy can use two small,
specialized local models: Qwen3 0.6B (about 523 MB) for general conversation and
Qwen2.5-Coder 0.5B (about 397 MB) for coding explanations. Both are below the
900 MB per-model limit. When both are installed, Lucy routes coding questions to the coder
model if no project workspace is enabled. When `LUCY_CODE_DIR` is enabled, Lucy
keeps the tool-capable Qwen3 model for project work. If the workspace is not
enabled, Lucy can explain or draft code but cannot inspect or edit project files.
Lucy responds to greetings and follow-ups conversationally and carries recent
turns into model requests. To prevent runaway tool loops, each turn is limited
to five tool rounds and six tool calls total; a repeated identical tool request
stops that turn before Lucy repeats the action.

```sh
ollama pull qwen3:0.6b
ollama pull qwen2.5-coder:0.5b
LUCY_API_BASE_URL=http://localhost:11434/v1 \
LUCY_API_MODEL=qwen3:0.6b \
LUCY_CODING_API_MODEL=qwen2.5-coder:0.5b \
python main.py
```

For quick factual and conversational replies, Lucy disables Qwen3's separate
reasoning output. On longer or more complex prompts, including coding work in a
configured workspace, she enables it. Reasoning may take longer on a CPU-only
computer. Common identity and date/time questions are answered directly by
Lucy, not guessed by a model.

When this backend is selected, Ollama must be running while Lucy is in use. With the auto-detected model,
conversation requests stay on the local machine. The model can choose from
Lucy's restricted tools:
calculator, local knowledge search, Wikipedia summaries, and saved-memory
recall. It cannot execute shell commands or arbitrary code. Lucy only saves a
new fact when you explicitly ask it to remember or save it. `/clear` clears
conversation history but keeps saved facts.

You can omit `ollama pull` after the models have been downloaded. Set the
`LUCY_API_*` variables again in each new terminal session where you run Lucy.
If `LUCY_API_BASE_URL` points to local Ollama and `LUCY_CODING_API_MODEL` is
omitted, Lucy detects the installed Qwen2.5-Coder model automatically.

Other OpenAI-compatible services can be configured the same way:

```sh
export LUCY_API_BASE_URL="https://your-provider.example/v1"
export LUCY_API_MODEL="your-model-name"
export LUCY_API_KEY="your-api-key"
python main.py
```

Only configure a provider you trust. When using a remote API, your conversation
and relevant saved context are sent to that provider. Lucy uses the API key only
for the request and does not write it to its memory file.

API configuration is checked at startup; malformed URLs and unsupported timeout
values fail with a clear error. Lucy's `/status` reports the configured model,
request timeout, and that API connectivity is checked when you send a request
(it does not claim a remote service is reachable before contacting it). The
default request timeout is 120 seconds; set `LUCY_API_TIMEOUT` to a value from
1 through 600 to change it. Responses are limited to 4 MB. Transient network
failures are retried once only for requests without tools, so tool-enabled
requests cannot accidentally repeat an action. If a request still fails, Lucy
reports the error and keeps the conversation open so you can retry.

GPT4All is another direct local runtime and supports macOS as well as Windows
and Linux; install it in the virtual environment:

```sh
python -m pip install gpt4all
```

Download a compatible `.gguf` model separately, then point Lucy at its file.
Local model files larger than 900 MB are rejected.
On macOS, use its full path in the environment variable:

```sh
export LUCY_MODEL_PATH="$HOME/Models/model.gguf"
python main.py
```

Use `LUCY_MODEL_BACKEND=gpt4all` to select it explicitly. For generated chat
and voice together, run `LUCY_VOICE=1 python main.py` after setting
`LUCY_MODEL_PATH`. On Windows PowerShell, set the path with
`$env:LUCY_MODEL_PATH = "C:\path\to\model.gguf"` before running `python main.py`.
Lucy loads only the specified local model and will not download one
automatically.

## Commands

- `/help` — list commands and capabilities
- `/remember <fact>` — save a fact for later sessions (`remember: <fact>` also works)
- `/recall [topic]` — list saved facts or find facts matching a topic
- `/search <topic>` — request a short Wikipedia summary (requires internet access)
- `/web <query>` — search the public web; Lucy can also search while chatting with a configured model
- `/read-url <url>` — read the text from a public webpage
- `/open-url <url>` — open a webpage in your browser after you approve
- `/open-app <name>` — open Calculator, Calendar, Finder, Notes, Safari, or System Settings after you approve
- `/files [folder]` — list supported files inside Lucy's selected learning folder
- `/learn <relative-path>` — explicitly copy one supported text file into Lucy's local learned library
- `/learned [query]` — search/read learned excerpts; use `/forget-file <relative-path>` to remove one
- `/feedback <correction>` — teach Lucy for this conversation; she asks before saving it for later chats
- `/reflect` — review recent conversation turns for possible reusable lessons; Lucy asks before saving
- `/code-files [folder]` — list source files in Lucy's coding workspace
- `/code-read <path>` — ask Lucy to inspect a project file
- `/code-check <python-path>` — check Python syntax without executing the file
- `/code-delete <path>` and `/code-restore <path>` — remove to recoverable trash or restore, after approval
- `/coding [topic]` — browse Lucy's built-in software engineering and language reference
- `/calculate <expression>` or `=<expression>` — calculate using numbers and
  `+`, `-`, `*`, `/`, `//`, `%`, and `**`
- `/status` — show the active mode and local data counts
- `/clear` — clear saved conversation history without deleting facts
- `/quit` — exit

Conversation history and saved facts are stored in `data/lucy_memory.json`.
Set `LUCY_MEMORY_PATH` or `LUCY_KNOWLEDGE_PATH` to use different files.
Memory and learned-file JSON are written to a temporary file and atomically
replaced to protect existing data from interrupted writes. If a save fails,
Lucy reports it instead of silently claiming the conversation or new memories
were saved; corrupt memory files are left untouched and reported at startup.
Use `/feedback <correction>` to teach Lucy during the current conversation.
Lucy will ask before saving that correction as a preference for future chats;
without approval it stays session-only and is cleared by `/clear`. When a
language model is configured, Lucy is instructed to reconsider and correct her
answer when you point out a mistake. Without a language model, she can retain
the correction but cannot generate revised open-ended answers.
With a language model configured, `/reflect` reviews recent conversation turns
and suggests only clearly stated, reusable response preferences or corrections.
Lucy asks separately before saving each suggestion; reflection does not run
arbitrary tools or save suggestions when approval is unavailable. The model may
choose relevant read-only tools to fulfill your explicit request, while code
changes, deletion, and computer actions remain approval-gated.
Do not put private information in `/search` queries; those queries are sent to
Wikipedia only when you explicitly use that command.

### Coding reference

Lucy includes a concise built-in programming reference in
`data/coding_reference.json`, covering software engineering fundamentals,
debugging, tests, common data structures, Python, JavaScript/TypeScript, Swift,
Java, Go, Rust, C/C++, SQL, web development, security, and Git. Browse it with
`/coding` or look up a topic with `/coding Python`. Coding questions search this
reference even without a language model, alongside any matching entries in the
main knowledge base. The reference is practical guidance, not a complete
language manual; generated, project-specific coding help still requires a
configured language model and an explicitly selected workspace.

### Learning from selected files

Lucy does not scan your computer. To let it access files, choose one folder and
set `LUCY_LEARN_DIR` before starting Lucy:

```sh
export LUCY_LEARN_DIR="$HOME/Documents/LucyShared"
python main.py
```

Use `/files` to see supported files in that folder, then `/learn notes.md` to
explicitly learn one. Access is limited to that folder, non-hidden UTF-8 text
files of supported types, and a 128 KB per-file limit. Symlinks, hidden files,
common credential-named files, and non-text formats are blocked. Lucy does not
execute or modify learned files. Learned copies are stored in
`data/lucy_learned.json`; `/forget-file notes.md` removes Lucy's saved copy.

### Supplied research reference

Lucy also loads `data/supplied_research_knowledge.json` as a local reference.
It includes the supplied emotion vocabulary and concise descriptions of the
image-triplet dataset and fMRI checkpoint. Ask Lucy a normal question about an
emotion or research artifact to retrieve related material; `/search` is
reserved for Wikipedia. This improves retrieval; it does not fine-tune the
language model. The image-triplet records are visual
similarity choices rather than emotion labels, and the fMRI checkpoint is a
specialized research model rather than a chat model. Participant-level records
and identifiers are not included in Lucy's reference.

### Optional fMRI response model

The supplied `best.ckpt` is a TRIBE v2 fMRI response encoder, not a chat model.
Lucy can load it as an optional, separate model for predicting responses to a
text, audio, or video stimulus. It does not decode thoughts, diagnose emotions,
or replace Lucy's conversational model. Loading and inference require the
matching `config.yaml`, the TRIBE v2 runtime, and its modality feature
encoders. The checkpoint is large; allow several gigabytes of memory and
additional storage for downloaded feature encoders.

TRIBE v2 requires Python 3.11 or newer. In a separate virtual environment,
install the optional runtime (licensed CC BY-NC 4.0; check those terms before
use):

```sh
python3.11 -m venv .venv-brain
source .venv-brain/bin/activate
python -m pip install -r requirements-brain.txt
LUCY_BRAIN_CHECKPOINT="$HOME/Downloads/best.ckpt" python main.py
```

By default Lucy looks for `config.yaml` beside the checkpoint and runs the
model on CPU. Set `LUCY_BRAIN_CONFIG` to use a different config,
`LUCY_BRAIN_CACHE` to choose the feature cache directory, or
`LUCY_BRAIN_DEVICE` to choose another PyTorch device. Use `/brain` to confirm
the model is loaded and `/brain-predict video /path/to/clip.mp4` (or `audio`
or `text`) to request a prediction. Text prediction may use external speech
services through the TRIBE runtime. Review that runtime's data handling and
license before use. Lucy safely normalizes the supplied YAML config before
passing it to the model loader; PyTorch checkpoint loading uses the
weights-only mode.

### Coding and improving Lucy

To enable project coding, point Lucy at exactly one source folder. For Lucy's
own repository:

```sh
export LUCY_CODE_DIR="$HOME/GitHub/Lucy"
python main.py
```

Lucy can inspect supported files and propose complete file changes in Python,
JavaScript/TypeScript, Swift, Java, C/C++, Go, Rust, Ruby, PHP, shell, SQL,
JSON, YAML, HTML/CSS, and other listed text formats. You will see the diff and
must approve every write. Removed files go to `.lucy-trash` inside the selected
workspace and can be restored with `/code-restore <path>`, also with approval.
Avoid pointing the workspace at your whole home folder. Keep important work
under version control and review changes before using them.

For Lucy to generate changes instead of just using the direct coding commands,
also configure a model with tool-calling support using the Ollama or GPT4All
setup above. Coding work is only available inside `LUCY_CODE_DIR`.

Lucy can check Python syntax without running the code. It does not execute
generated code, run arbitrary terminal commands, or run test suites on its own:
those actions could modify files, access credentials, or contact services.
Review and test approved code yourself.

For a natural-language request such as “analyze main.py,” Lucy reads only the
explicitly selected file. For Python files she also reports a static,
non-executing outline of imports, top-level definitions, and a direct script
entry point. A configured model may add interpretation, but the outline remains
available if the model returns a generic or empty answer.

Use `/feedback <correction or preference>` or `/remember <fact>` to help Lucy
adapt future responses. These notes persist locally; this is explicit memory,
not self-training or human-like learning.

### Public internet and computer actions

With an internet connection, `/web <query>` searches the public web and
`/read-url <url>` reads a public HTTP(S) text or HTML page. Lucy's configured
chat model can use these tools when relevant. Page reads are limited to 1 MB
and extracted text to 12,000 characters. Local/private-network addresses,
non-HTTP(S) schemes, and nonstandard ports are blocked to avoid exposing
services on your computer. `/open-url` and model-requested app/site launches
ask you before opening anything. App launching is limited to the named macOS
allowlist. Lucy cannot click around, type, run terminal commands, install
software, upload files, or post to websites.

Search queries are sent to Bing and pages are fetched from the sites you
request. If you use a remote chat API, relevant conversation, learned-file
excerpts, and web results may also be sent to that provider. Use local Ollama
if you want model inference to stay on your Mac.

Spoken replies are optional. On macOS they use the built-in `say` command; on
other systems, install `pyttsx3` and set `LUCY_VOICE=1` to use the platform's
local text-to-speech engine. Without that setting, Lucy uses text only.

## Development checks

Run the standard-library test suite with:

```sh
python3 -m unittest discover -s tests
```
