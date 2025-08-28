# evaluate_accuracy.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ----------- CONFIG -----------
MODEL_NAME = "mistralai/Mistral-7B-v0.3"      # Base model
ADAPTER_PATH = "./qlora_pubmedqa/checkpoint-441/"           # Fine-tuned LoRA adapter directory
DATASET_NAME = "qiaojin/PubMedQA"
MAX_SEQ_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------- 1. LOAD DATASET -----------
print("Loading evaluation dataset...")
dataset = load_dataset(DATASET_NAME, "pqa_labeled")

# Use a small subset for quick testing
eval_data = dataset["train"].select(range(100))

# ----------- 2. LOAD MODEL WITH ADAPTER -----------
print("Loading base model and LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# ----------- 3. EVALUATION LOOP -----------
correct = 0
total = len(eval_data)

print(f"Evaluating on {total} samples...")

for example in eval_data:
    question = example["question"]
    expected_answer = example["final_decision"].strip().lower()  # yes/no

    prompt = f"Answer the following medical question concisely with yes or no:\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

    # Check if "yes" or "no" appears in the prediction
    if "yes" in prediction and expected_answer == "yes":
        correct += 1
    elif "no" in prediction and expected_answer == "no":
        correct += 1

accuracy = correct / total * 100
print(f"\nAccuracy: {accuracy:.2f}%")
