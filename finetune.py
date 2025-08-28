# qlora_finetune_pubmedqa.py

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# ----------- CONFIG -----------
MODEL_NAME = "mistralai/Mistral-7B-v0.3"  # Base model
DATASET_NAME = "qiaojin/PubMedQA"
OUTPUT_DIR = "./qlora_pubmedqa"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
EPOCHS = 7

# ----------- 1. DOWNLOAD DATASET -----------
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, "pqa_labeled")

# ----------- 2. TOKENIZER -----------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def format_example(example):
    instruction = (
        "### Instruction:\n"
        "Answer the following medical question concisely.\n\n"
        f"Question: {example['question']}\n\n### Response:\n"
    )
    return {
        "input_ids": tokenizer(
            instruction, truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH
        )["input_ids"],
        "labels": tokenizer(
            example["long_answer"], truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH
        )["input_ids"]
    }

print("Tokenizing dataset...")
tokenized = dataset["train"].map(format_example)

# ----------- 3. DOWNLOAD & LOAD MODEL -----------
print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto"
)

# ----------- 4. APPLY QLoRA -----------
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # important for attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ----------- 5. DATA COLLATOR -----------
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ----------- 6. TRAINING ARGS -----------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=100,
    num_train_epochs=EPOCHS,
    fp16=True,
    save_total_limit=2,
    logging_steps=10,
    report_to="none"  # disable WandB by default
)

# ----------- 7. TRAINER -----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# ----------- 8. TRAIN -----------
print("Starting fine-tuning...")
trainer.train()

print("Fine-tuning completed. Model saved at:", OUTPUT_DIR)
