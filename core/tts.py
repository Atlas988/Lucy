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


edge_tts = _safe_optional_import("edge_tts")
import asyncio
pygame = _safe_optional_import("pygame")

import threading
import uuid
import os
import queue
try:
    import tkinter as tk
except Exception:  # pragma: no cover - optional dependency
    tk = None
import time

# Thread-safe queue for status updates to GUI
status_queue = queue.Queue()

def set_status(text, bg="green", fg="white"):
    status_queue.put((text, bg, fg))

async def generate_speech(text, filename):
    if edge_tts is None:
        raise RuntimeError("edge_tts is not installed")
    set_status("Generating speech...", "blue", "white")
    communicator = edge_tts.Communicate(text, voice="en-US-MichelleNeural")
    await communicator.save(filename)
    set_status("Playing audio...", "yellow", "black")

def safe_delete(filename, retries=5, delay=0.2):
    """Tries to safely delete a file with retries in case it's locked."""
    for _ in range(retries):
        try:
            if os.path.exists(filename):
                os.remove(filename)
                return
        except PermissionError:
            time.sleep(delay)

def play_audio_nonblocking(filename):
    if pygame is None:
        return

    def _play():
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            set_status("Speaking...", "yellow", "black")

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.music.stop()
            safe_delete(filename)
            set_status("Idle", "green", "white")

        except Exception as e:
            set_status(f"Audio error: {e}", "red", "white")
            print(f"Audio playback error: {e}")

    threading.Thread(target=_play, daemon=True).start()

def speak(text):
    filename = f"tts_{uuid.uuid4().hex}.mp3"

    def run_tts():
        asyncio.run(generate_speech(text, filename))
        play_audio_nonblocking(filename)

    threading.Thread(target=run_tts, daemon=True).start()

def gui_thread():
    if tk is None:
        return
    root = tk.Tk()
    root.title("Lucy Status")
    root.geometry("200x50")
    root.resizable(False, False)

    status_label = tk.Label(root, text="Idle", font=("Arial", 14), fg="white", bg="green")
    status_label.pack(fill=tk.BOTH, expand=True)

    def check_queue():
        try:
            while True:
                text, bg, fg = status_queue.get_nowait()
                status_label.config(text=text, bg=bg, fg=fg)
        except queue.Empty:
            pass
        root.after(100, check_queue)

    root.after(100, check_queue)
    root.mainloop()


# Example usage: Speak after 2 seconds delay
if __name__ == "__main__":
    time.sleep(2)
    speak("Hello! I am Lucy, your AI assistant.")
