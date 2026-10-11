import asyncio
import threading
import uuid
import csv
import os
import time
import json
import subprocess
import sys
from pathlib import Path
import difflib
from statistics import mean


def _safe_optional_import(module_name, symbol_name=None):
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
        module = __import__(module_name, fromlist=[symbol_name] if symbol_name else [])
    except Exception:
        return None
    if symbol_name:
        return getattr(module, symbol_name, None)
    return module


pygame = _safe_optional_import("pygame")
edge_tts = _safe_optional_import("edge_tts")
GPT4All = _safe_optional_import("gpt4all", "GPT4All")

# --- Reasoning Engine ---
class ReasoningEngine:
    def __init__(self, memory_file="scenario_memory.json", ethics_dir="ethics", gpt_model_path="E:/lucy_ai/models/qwen2-1_5b-instruct-q4_0.gguf"):
        self.memory_file = memory_file
        self.knowledge = []
        self.load()
        if ethics_dir:
            self.load_ethics_dataset(ethics_dir)
        self.model = None
        if GPT4All is not None:
            try:
                self.model = GPT4All(model_name="qwen2-1_5b-instruct-q4_0.gguf", model_path="E:/lucy_ai/models")
            except Exception:
                self.model = None

    def load(self):
        if Path(self.memory_file).exists():
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.knowledge = json.load(f)
        else:
            self.knowledge = []

    def save(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.knowledge, f, indent=2)

    def add_scenario(self, situation, reasoning, decision, moral_weight=0, urgency=0, legal_filter="Unknown", consequence=0):
        self.knowledge.append({
            "situation": situation.strip(),
            "reasoning": reasoning.strip(),
            "decision": decision.strip(),
            "moral_weight": moral_weight,
            "urgency": urgency,
            "legal_filter": legal_filter,
            "consequence": consequence
        })
        self.save()

    def load_ethics_dataset(self, root_dir):
        count = 0
        print(f"Scanning: {root_dir}")
        for filename in os.listdir(root_dir):
            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(root_dir, filename)
            print(f"Loading {file_path}")

            try:
                full_path = os.path.join(root_dir, filename)

                if filename.startswith("util"):
                    with open(full_path, newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) != 3:
                                continue
                            situation, reasoning, decision = row
                            self.knowledge.append({
                                "situation": situation.strip(),
                                "reasoning": reasoning.strip(),
                                "decision": decision.strip(),
                                "moral_weight": 5,
                                "urgency": 5,
                                "legal_filter": "Unknown",
                                "consequence": 5,
                                "source": "util"
                            })
                            count += 1

                else:
                    with open(full_path, newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            situation = ""
                            reasoning = "No reasoning provided."
                            decision = ""

                            if filename.startswith("cm_"):
                                situation = row.get("input", "")
                                decision = row.get("label", "")

                            elif filename.startswith("deontology"):
                                situation = row.get("scenario", "")
                                reasoning = row.get("excuse", "")
                                decision = row.get("label", "")

                            elif filename.startswith("justice"):
                                situation = row.get("scenario", "")
                                decision = row.get("label", "")

                            elif filename.startswith("virtue"):
                                situation = row.get("observation", "")
                                reasoning = row.get("rationale", "")
                                decision = row.get("answer", "")

                            if situation and decision:
                                self.knowledge.append({
                                    "situation": situation.strip(),
                                    "reasoning": reasoning.strip(),
                                    "decision": decision.strip(),
                                    "moral_weight": 5,
                                    "urgency": 5,
                                    "legal_filter": "Unknown",
                                    "consequence": 5,
                                    "source": filename.split("_")[0] if "_" in filename else filename
                                })
                                count += 1

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Loaded {count} moral scenarios from ETHICS dataset.")

    def analyze(self, new_situation):
        if not self.knowledge:
            return "I don't have enough examples yet to reason about that.", None

        best_match = max(self.knowledge, key=lambda x: difflib.SequenceMatcher(None, new_situation, x["situation"]).ratio())
        similarity = difflib.SequenceMatcher(None, new_situation, best_match["situation"]).ratio()

        if similarity < 0.4:
            return "I don't know enough about this situation yet.", None

        # Compose prompt for GPT4All
        prompt = f"""
User Input: {new_situation}

Matched Scenario: {best_match['situation']}
Ethical Decision: {best_match['decision']}
Reasoning Provided: {best_match['reasoning']}
Metadata:
- Moral Weight: {best_match.get('moral_weight', 0)}
- Urgency: {best_match.get('urgency', 0)}
- Consequence Impact: {best_match.get('consequence', 0)}
- Legal Status: {best_match.get('legal_filter', 'Unknown')}

Explain step-by-step why this ethical decision is justified, referencing ethical theories such as utilitarianism, deontology, and virtue ethics.
"""

        try:
            if self.model is None:
                generated = "Local model not available. Returning a best-effort ethical rationale without the model." \
                    "\n\nA responsible decision should weigh the harm caused, rights involved, and the specific circumstances."
            else:
                generated = self.model.generate(prompt, n_predict=256, temperature=0.7)
        except Exception as e:
            generated = f"Error generating explanation: {e}"

        output = (
            f"=== MORAL REASONING REPORT ===\n"
            f"Situation Match (Similarity: {similarity:.2f}):\n'{best_match['situation']}'\n"
            f"Decision: {best_match['decision']}\n\n"
            f"GPT4All Reasoning Explanation:\n{generated.strip()}\n"
        )

        return output, best_match["decision"]


# --- Audio Setup ---
if pygame is not None:
    os.environ.setdefault("SDL_AUDIODRIVER", "directsound")

audio_lock = threading.Lock()

# --- TTS ---
async def generate_speech(text, filename):
    if edge_tts is None:
        raise RuntimeError("edge_tts is not installed")
    communicator = edge_tts.Communicate(text, voice="en-US-MichelleNeural")
    await communicator.save(filename)

def play_audio_blocking(filename):
    if pygame is None:
        return
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()
        if hasattr(pygame.mixer.music, 'unload'):
            pygame.mixer.music.unload()
        time.sleep(0.1)
    except Exception as e:
        print(f"Audio playback error: {e}")

def play_audio_nonblocking(text):
    if pygame is None or edge_tts is None:
        return
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    def _play():
        try:
            asyncio.run(generate_speech(text, filename))
            with audio_lock:
                play_audio_blocking(filename)
        except Exception as e:
            print(f"Audio generation error: {e}")
        try:
            os.remove(filename)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()

# --- Main Async Loop ---
async def main():
    engine = ReasoningEngine()

    greeting = "Hello, I'm Lucy. How can I help you today?"
    print("Lucy:", greeting)
    play_audio_nonblocking(greeting)

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if user_input.lower() in ["exit", "quit"]:
            farewell = "Goodbye. Stay thoughtful."
            print("Lucy:", farewell)
            play_audio_nonblocking(farewell)
            await asyncio.sleep(2)
            break

        # Optional: allow user to add memory with 'remember:' prefix
        if user_input.lower().startswith("remember:"):
            try:
                parts = user_input[9:].split("|")
                if len(parts) >= 3:
                    situation, reasoning, decision = map(str.strip, parts[:3])
                    moral_weight = float(parts[3].strip()) if len(parts) > 3 else 0
                    urgency = float(parts[4].strip()) if len(parts) > 4 else 0
                    legal_filter = parts[5].strip() if len(parts) > 5 else "Unknown"
                    engine.add_scenario(situation, reasoning, decision, moral_weight, urgency, legal_filter)
                    response = "I've memorized that scenario."
                else:
                    response = "Invalid format. Use: remember: situation | reasoning | decision | [moral_weight] | [urgency] | [legal_filter]"
            except Exception as e:
                response = f"Error processing memory: {e}"
        else:
            response, _ = engine.analyze(user_input)
            if not response:
                response = "Sorry, I don't have an answer for that."

        print("Lucy:", response)
        play_audio_nonblocking(response)

if __name__ == "__main__":
    asyncio.run(main())
