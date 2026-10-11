import os
import sys

from core.audio import VoiceOutput
from core.ai_engine import Lucy


def speak_if_enabled(voice, text):
    if voice is None:
        return None
    try:
        voice.speak(text)
    except RuntimeError as error:
        print(f"Lucy voice output failed: {error}", file=sys.stderr)
        return None
    return voice


def main():
    try:
        lucy = Lucy.from_environment()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Lucy could not start: {error}", file=sys.stderr)
        return 1

    voice = None
    if os.environ.get("LUCY_VOICE", "").lower() in ("1", "true", "yes", "on"):
        try:
            voice = VoiceOutput()
        except RuntimeError as error:
            print(f"Lucy voice output is unavailable: {error}", file=sys.stderr)

    if lucy.model is None:
        print(
            "Lucy is running in local knowledge mode. Set LUCY_MODEL_PATH "
            "to a GPT4All .gguf model for generated conversations."
        )

    greeting = "Hello! Type /help to see what I can do, or /quit to exit."
    print(f"Lucy: {greeting}")
    voice = speak_if_enabled(voice, greeting)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLucy: Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit", "bye", "goodbye"):
            print("Lucy: Goodbye.")
            break

        try:
            response = lucy.respond(user_input)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Lucy encountered an error: {error}", file=sys.stderr)
            return 1
        print(f"Lucy: {response}")
        voice = speak_if_enabled(voice, response)

    speak_if_enabled(voice, "Goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
