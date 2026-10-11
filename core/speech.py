import pyttsx3

# Initialize the text-to-speech engine
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

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Speech error:", e)
