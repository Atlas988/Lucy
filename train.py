import os
import pandas as pd
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# === Configuration ===
parquet_file = "train.parquet"         # Replace with your .parquet file path
text_column = "text"                   # Replace with the column name containing text
model_name = "gpt2"
output_dir = "./lucy_model_output"     # Changed to avoid corrupted folder issue

# === Load text from Parquet ===
def load_parquet_text(file_path, text_column="text"):
    df = pd.read_parquet(file_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the Parquet file.")
    return df[text_column].dropna().tolist()

# === Tokenize the text ===
def tokenize_data(texts, tokenizer, block_size=128):
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=block_size)
    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(tokenize, batched=True)

# === Load and prepare ===
print("Loading tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Important for GPT2 (no pad token by default)
model = GPT2LMHeadModel.from_pretrained(model_name)

print(f"Loading Parquet file: {parquet_file}")
texts = load_parquet_text(parquet_file, text_column)
print(f"Loaded {len(texts)} samples.")

print("Tokenizing data...")
tokenized_dataset = tokenize_data(texts, tokenizer)

# === Setup training arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
    fp16=False,  # Set to False if not using a GPU with mixed precision
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# === Trainer ===
print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
print(f"Training completed. Model saved to: {output_dir}")
