# Lucy

Lucy is a text-based assistant with persistent conversation context, local
knowledge lookup, safe arithmetic, explicit Wikipedia lookups, and optional
speech. For generated conversations, it supports either a local GPT4All model
or an OpenAI-compatible chat API (including a local Ollama server).

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

## Generated conversations with Ollama

For local model inference on macOS, install and launch Ollama, then download a
model that supports tool calling (for example `qwen3:4b`):

```sh
ollama pull qwen3:4b
LUCY_API_BASE_URL=http://localhost:11434/v1 \
LUCY_API_MODEL=qwen3:4b \
python main.py
```

Ollama must be running while Lucy is in use. In this mode, conversation requests
stay on the local machine. The model can choose from Lucy's restricted tools:
calculator, local knowledge search, Wikipedia summaries, and saved-memory
recall. It cannot execute shell commands or arbitrary code. Lucy only saves a
new fact when you explicitly ask it to remember or save it. `/clear` clears
conversation history but keeps saved facts.

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

To enable generated conversation, install GPT4All in the virtual environment:

```sh
python -m pip install gpt4all
```

Download a compatible `.gguf` model separately, then point Lucy at its file.
On macOS, use its full path in the environment variable:

```sh
export LUCY_MODEL_PATH="$HOME/Models/model.gguf"
python main.py
```

For generated chat and voice together, run `LUCY_VOICE=1 python main.py` after
setting `LUCY_MODEL_PATH`. On Windows PowerShell, set the path with
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
- `/calculate <expression>` or `=<expression>` — calculate using numbers and
  `+`, `-`, `*`, `/`, `//`, `%`, and `**`
- `/status` — show the active mode and local data counts
- `/clear` — clear saved conversation history without deleting facts
- `/quit` — exit

Conversation history and saved facts are stored in `data/lucy_memory.json`.
Set `LUCY_MEMORY_PATH` or `LUCY_KNOWLEDGE_PATH` to use different files.
Do not put private information in `/search` queries; those queries are sent to
Wikipedia only when you explicitly use that command.

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
