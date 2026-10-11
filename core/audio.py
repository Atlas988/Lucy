import platform
import shutil
import subprocess


class VoiceOutput:
    def __init__(self):
        self._macos_say = None
        if platform.system() == "Darwin":
            self._macos_say = shutil.which("say")
            if self._macos_say is None:
                raise RuntimeError("macOS text-to-speech command 'say' was not found.")
            return

        try:
            import pyttsx3
        except ImportError as error:
            raise RuntimeError(
                "Install the optional speech package with: "
                "python -m pip install pyttsx3"
            ) from error

        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 175)
        except Exception as error:
            raise RuntimeError(f"Could not initialize text-to-speech: {error}") from error

    def speak(self, text):
        if self._macos_say:
            try:
                subprocess.run(
                    [self._macos_say, text],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                raise RuntimeError(f"macOS text-to-speech failed: {error}") from error
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as error:
            raise RuntimeError(f"Text-to-speech failed: {error}") from error
