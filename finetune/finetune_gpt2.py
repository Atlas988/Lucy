from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

def main():
    # Load and preprocess data
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    dataset = load_dataset("text", data_files={"train": "train.txt"})


    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./finetune_output",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),  # Enable if using GPU
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    print(f"Training samples: {len(tokenized_dataset['train'])}")
    trainer.train()
    trainer.save_model("./finetune_output/final_model")
    trainer.save_model("finetuned-gpt2-small")
    tokenizer.save_pretrained("finetuned-gpt2-small")


if __name__ == "__main__":
    main()
