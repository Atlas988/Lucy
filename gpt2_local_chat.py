# gpt2_local_chat.py

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None
import os

# === CONFIGURATION ===
MODEL_NAME = "gpt2"                  # You can change to "gpt2-medium", "gpt2-large", etc.
LOCAL_DIR = "./lucy_gpt2_model"     # Where to store/load the model
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7


# === STEP 1: Download and Save Locally (Run once) ===
def download_and_save_model():
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is not installed")
    print("Downloading and saving GPT-2 to local directory...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    tokenizer.save_pretrained(LOCAL_DIR)
    model.save_pretrained(LOCAL_DIR)
    print(f"Model saved to {LOCAL_DIR}")


# === STEP 2: Load Model from Local Directory ===
def load_local_model():
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is not installed")
    print("Loading GPT-2 from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR)
    model.eval()
    return tokenizer, model


# === STEP 3: Get Response from GPT-2 ===
def get_response(prompt, tokenizer, model):
    if tokenizer is None or model is None or torch is None:
        return f"[offline fallback] {prompt.strip() or 'How can I help?'}"
    input_text = f"User: {prompt}\nAI:"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the part after "AI:"
    if "AI:" in response:
        return response.split("AI:")[-1].strip()
    return response.strip()


# === STEP 4: Main Chat Loop ===
def chat_loop(tokenizer, model):
    print("Hi! I'm Lucy (powered by GPT-2). Ask me anything. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Lucy: Goodbye!")
                break
            response = get_response(user_input, tokenizer, model)
            print(f"Lucy: {response}")
        except KeyboardInterrupt:
            print("\nLucy: Exiting chat. Goodbye!")
            break


# === ENTRY POINT ===
if __name__ == "__main__":
    if not os.path.exists(LOCAL_DIR):
        download_and_save_model()

    tokenizer, model = load_local_model()
    chat_loop(tokenizer, model)
