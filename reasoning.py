import json
import difflib
import csv
import os
from pathlib import Path
from statistics import mean

try:
    from gpt4all import GPT4All
except Exception:  # pragma: no cover - optional dependency
    GPT4All = None

class ReasoningEngine:
    def __init__(self, memory_file="scenario_memory.json", ethics_dir="ethics", model_path="models/mistral-7b-instruct-v0.1.Q4_0.gguf"):
        self.memory_file = memory_file
        self.knowledge = []
        self.model_path = Path(model_path)
        self.model = None
        if GPT4All is not None:
            try:
                self.model = self.load_model(self.model_path)
            except (FileNotFoundError, Exception):
                self.model = None
        self.load()
        if ethics_dir:
            self.load_ethics_dataset(ethics_dir)

    def load_model(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        return GPT4All(path)

    def load(self):
        if Path(self.memory_file).exists():
            with open(self.memory_file, "r") as f:
                self.knowledge = json.load(f)

    def save(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.knowledge, f, indent=2)

    def load_ethics_dataset(self, ethics_dir):
        ethics_dir = Path(ethics_dir)
        if not ethics_dir.exists():
            return
        for file in ethics_dir.glob("*.csv"):
            with open(file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.knowledge.append(row)

    def similarity(self, a, b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def query_knowledge(self, question, threshold=0.6):
        scored = []
        for entry in self.knowledge:
            context = entry.get("scenario", "") or entry.get("question", "")
            score = self.similarity(context, question)
            if score >= threshold:
                scored.append((score, entry))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [entry for _, entry in scored[:5]]

    def reason(self, question):
        context = self.query_knowledge(question)
        prompt = "Given the following known scenarios:\n"
        for i, entry in enumerate(context):
            prompt += f"{i+1}. {entry.get('scenario') or entry.get('question')} -> {entry.get('answer', entry.get('response'))}\n"
        prompt += f"\nAnswer the following question with reasoning:\n{question}\n"

        print("\n--- Reasoning Prompt ---")
        print(prompt)
        print("------------------------\n")

        if self.model is None:
            return "Local reasoning model is unavailable. Based on the available knowledge, the best answer is to reason carefully from the relevant context and avoid assumptions."

        response = self.model.generate(prompt, max_tokens=300)
        return response.strip()

    def add_knowledge(self, scenario, answer):
        self.knowledge.append({"scenario": scenario, "answer": answer})
        self.save()
