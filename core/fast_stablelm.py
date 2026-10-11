from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_name = "stabilityai/stablelm-tuned-alpha-1b"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # float16 reduces memory, still works on CPU with newer PyTorch
        low_cpu_mem_usage=True      # reduces CPU RAM usage while loading
    )

    device = torch.device("cpu")
    model.to(device)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break

        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Trim input prompt from response
        if user_input in response:
            response = response[len(user_input):].strip()

        print("Lucy:", response)

if __name__ == "__main__":
    main()
