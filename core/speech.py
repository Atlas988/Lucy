try:
    import pyttsx3
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None

# Initialize the text-to-speech engine
engine = None
if pyttsx3 is not None:
    try:
        engine = pyttsx3.init()
        # Optional settings
        engine.setProperty('rate', 175)     # Speed (default ~200)
        engine.setProperty('volume', 1.0)   # Volume (0.0 to 1.0)

        # Pick a voice (male/female depending on OS)
        voices = engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
    except Exception:
        engine = None


def speak(text):
    if engine is None:
        print(f"Speech disabled: {text}")
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Speech error:", e)
