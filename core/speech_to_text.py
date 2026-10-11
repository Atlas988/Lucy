import queue
import sounddevice as sd
import vosk
import json
import sys
import threading

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

class VoskRecognizer:
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        print("Loading Vosk model...")
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, 16000)
        self.running = False

    def start_listening(self, on_result_callback):
        self.running = True
        def listen():
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                   channels=1, callback=audio_callback):
                print("Listening for speech... Press Ctrl+C to stop.")
                while self.running:
                    data = q.get()
                    if self.rec.AcceptWaveform(data):
                        result = json.loads(self.rec.Result())
                        text = result.get('text', '')
                        if text:
                            on_result_callback(text)
                    else:
                        # partial = json.loads(self.rec.PartialResult())
                        # You can process partial results here if you want
                        pass
        self.listener_thread = threading.Thread(target=listen, daemon=True)
        self.listener_thread.start()

    def stop_listening(self):
        self.running = False
        if self.listener_thread.is_alive():
            self.listener_thread.join()
