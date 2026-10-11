import queue
import json
import sys
import threading

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None

try:
    import vosk
except Exception:  # pragma: no cover - optional dependency
    vosk = None

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

class VoskRecognizer:
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        self.model = None
        self.rec = None
        self.running = False
        self.listener_thread = None
        if vosk is None or sd is None:
            print("Vosk and sounddevice are not installed; speech recognition is unavailable.")
            return
        print("Loading Vosk model...")
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, 16000)

    def start_listening(self, on_result_callback):
        if self.rec is None or sd is None:
            return
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
        if self.listener_thread is not None and self.listener_thread.is_alive():
            self.listener_thread.join()
