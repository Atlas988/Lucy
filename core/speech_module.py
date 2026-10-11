import pyttsx3
from gtts import gTTS
import tempfile
import pygame
import os
import socket

engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('volume', 1.0)

def is_connected(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

def speak_offline(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Offline TTS error:", e)

def speak_online(text):
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
