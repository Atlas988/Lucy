import os
import sys

from core.audio import VoiceOutput
from core.ai_engine import Lucy


def confirm_computer_action(action):
    print(f"\nLucy wants permission to {action}")
    try:
        answer = input("Allow this one action? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nLucy: Action cancelled.")
        return False
    return answer in ("y", "yes")


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
        lucy = Lucy.from_environment(approval_callback=confirm_computer_action)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Lucy could not start: {error}", file=sys.stderr)
        return 1

    voice = None
    if os.environ.get("LUCY_VOICE", "").lower() in ("1", "true", "yes", "on"):
        try:
            voice = VoiceOutput()
        except RuntimeError as error:
            print(f"Lucy voice output is unavailable: {error}", file=sys.stderr)

    if lucy.model is None and not lucy.api_base_url:
        print(
            "Lucy is running in knowledge-only mode. Configure GPT4All with "
            "LUCY_MODEL_PATH or an OpenAI-compatible model with LUCY_API_BASE_URL "
            "for generated conversations. A running Ollama installation with "
            "Qwen3 0.6B or 1.7B is detected automatically."
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
