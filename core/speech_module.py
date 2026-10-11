import subprocess
import sys


def _safe_optional_import(module_name):
    try:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        return __import__(module_name)
    except Exception:
        return None


pyttsx3 = _safe_optional_import("pyttsx3")
GTTSModule = _safe_optional_import("gtts")
gTTS = getattr(GTTSModule, "gTTS", None) if GTTSModule is not None else None
pygame = _safe_optional_import("pygame")

import tempfile
import os
import socket

engine = None
if pyttsx3 is not None:
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 1.0)
    except Exception:
        engine = None

def is_connected(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

def speak_offline(text):
    if engine is None:
        print(f"Offline TTS unavailable: {text}")
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Offline TTS error:", e)

def speak_online(text):
    if gTTS is None or pygame is None:
        print(f"Online TTS unavailable: {text}")
        return
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(tf.name)

        pygame.mixer.init()
        pygame.mixer.music.load(tf.name)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()

        os.remove(tf.name)

    except Exception as e:
        print("Online TTS error:", e)
        speak_offline(text)

def speak(text):
    if is_connected():
        speak_online(text)
    else:
        speak_offline(text)
