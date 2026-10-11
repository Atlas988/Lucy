# Lucy

Lucy is a local, text-based assistant. It can use a local GPT4All model for
generated replies, search the project's `knowledge_base.json`, remember facts
between runs, perform basic arithmetic, and optionally look up Wikipedia
summaries.

## Run Lucy

Python 3.9 or newer is required. From the project directory, run:

```sh
python3 main.py
```

Lucy runs in knowledge-only mode without extra packages or a model. In that
mode, it answers simple built-in questions and questions covered by the local
knowledge base; it does not pretend to generate answers to other questions.

To enable generated conversation, install GPT4All:

```sh
python3 -m pip install gpt4all
```

Download a compatible `.gguf` model separately, then point Lucy at its file:

```sh
export LUCY_MODEL_PATH="/path/to/model.gguf"
python3 main.py
```

On Windows PowerShell, set the path with
`$env:LUCY_MODEL_PATH = "C:\path\to\model.gguf"` before running `python main.py`.
Lucy loads only the specified local model and will not download one
automatically.

## Commands

- `/help` — list commands and capabilities
- `/remember <fact>` — save a fact for later sessions (`remember: <fact>` also works)
- `/recall [topic]` — list saved facts or find facts matching a topic
- `/search <topic>` — request a short Wikipedia summary (requires internet access)
- `/calculate <expression>` or `=<expression>` — calculate using numbers and
  `+`, `-`, `*`, `/`, `//`, `%`, and `**`
- `/status` — show the active mode and local data counts
- `/quit` — exit

Conversation history and saved facts are stored in `data/lucy_memory.json`.
Set `LUCY_MEMORY_PATH` or `LUCY_KNOWLEDGE_PATH` to use different files.
Do not put private information in `/search` queries; those queries are sent to
Wikipedia only when you explicitly use that command.

Spoken replies are optional. Install `pyttsx3` and set `LUCY_VOICE=1` to enable
the operating system's local text-to-speech engine. Without that setting,
Lucy uses text only.

## Development checks

Run the standard-library test suite with:

```sh
python3 -m unittest discover -s tests
```
