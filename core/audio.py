class VoiceOutput:
    def __init__(self):
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
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as error:
            raise RuntimeError(f"Text-to-speech failed: {error}") from error
