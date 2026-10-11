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

Install and launch the Ollama macOS app from
[ollama.com/download](https://ollama.com/download). For this Intel Mac, Lucy is
configured to use the faster Qwen3 0.6B model; it needs about 522 MB of model
storage. In a short conversational test on this Mac, Lucy replied in about 10
seconds with Qwen3 0.6B versus 24 seconds with Qwen3 1.7B. The smaller model is
less capable; use Qwen3 1.7B instead if you prefer its answer quality.

```sh
ollama pull qwen3:0.6b
LUCY_API_BASE_URL=http://localhost:11434/v1 \
LUCY_API_MODEL=qwen3:0.6b \
python main.py
```

For faster replies, Lucy disables Qwen3's separate reasoning output when using
Ollama on this Mac. To use the larger model, change both `qwen3:0.6b` values
above to `qwen3:1.7b` (download it first with `ollama pull qwen3:1.7b`).

Ollama must be running while Lucy is in use. In this mode, conversation requests
stay on the local machine. The model can choose from Lucy's restricted tools:
calculator, local knowledge search, Wikipedia summaries, and saved-memory
recall. It cannot execute shell commands or arbitrary code. Lucy only saves a
new fact when you explicitly ask it to remember or save it. `/clear` clears
conversation history but keeps saved facts.

You can omit `ollama pull` after the model has been downloaded. Set the two
`LUCY_API_*` variables again in each new terminal session where you run Lucy.

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
