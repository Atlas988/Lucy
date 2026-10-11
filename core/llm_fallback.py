from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model (once)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()  # inference mode

def generate_response(prompt, max_new_tokens=100, temperature=0.7):
    # Encode input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate tokens
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,        # sampling for diversity
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    user_prompt = "Hello, how are you?"
    response = generate_response(user_prompt)
    print("AI:", response)
