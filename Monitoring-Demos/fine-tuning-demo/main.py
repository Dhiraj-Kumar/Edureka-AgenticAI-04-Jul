import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

# Step 1. Model Configuration

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Training on: {device}")

# Step 2. Load the training dataset
with open("domain_dataset.jsonl", "r", encoding="utf-8") as file:
    rows = [json.loads(line) for line in file]

texts = [
    f"Question: {row['prompt']}\nAnswer: {row['completion']}"
    for row in rows
]

dataset = Dataset.from_dict(
    {
        "text": texts
    }
)
print(dataset)
print(f"Training examples: {len(dataset)}")

# Step 3. Load tokenizer

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

tokenizer.pad_token = tokenizer.eos_token

# Step 4. Tokenize the dataset
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )


print("Tokenising dataset...")

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

# Step 5. Load the base model
print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID
)

base_model = base_model.to(device)

# Step 6. Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Step 7. Attach LoRA Adapters
peft_model = get_peft_model(
    base_model,
    lora_config,
)

print("\nLoRA parameter summary:")

peft_model.print_trainable_parameters()

# Step 8. Training configuration
training_args = TrainingArguments(
    output_dir="./lora_output",

    # CPU-friendly demo configuration
    per_device_train_batch_size=1,

    max_steps=5,

    learning_rate=2e-4,

    # Print the loss after every step.
    logging_steps=1,

    # We save the final adapter manually below.
    save_strategy="no",

    report_to="none",

    dataloader_num_workers=0,
)

# Step 9. Data collator

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Step 10. Create trainer

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Step 11. Start LoRA training
print("STARTING LoRA TRAINING")
trainer.train()

# Step 12.
ADAPTER_PATH = "./nimbuscloud-lora-adapter"

peft_model.save_pretrained(
    ADAPTER_PATH
)

tokenizer.save_pretrained(
    ADAPTER_PATH
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"LoRA adapter saved to {ADAPTER_PATH}"
)
