import json
import os

MEMORY_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'learned_qa.json')

def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return {'history': [], 'qa': []}  # ensure 'qa' is initialized
    with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
        memory = json.load(f)

    # Ensure both keys exist even if the file is malformed
    if "qa" not in memory:
        memory["qa"] = []
    if "history" not in memory:
        memory["history"] = []

    return memory


def save_memory(memory):
    with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2)
