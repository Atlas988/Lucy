import edge_tts
import asyncio
import pygame
import threading
import uuid
import os
import queue
import tkinter as tk
import time

pygame.mixer.init()

# Thread-safe queue for status updates to GUI
status_queue = queue.Queue()

def set_status(text, bg="green", fg="white"):
    status_queue.put((text, bg, fg))

async def generate_speech(text, filename):
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

# Start GUI thread
threading.Thread(target=gui_thread, daemon=True).start()

# Example usage: Speak after 2 seconds delay
if __name__ == "__main__":
    time.sleep(2)
    speak("Hello! I am Lucy, your AI assistant.")
