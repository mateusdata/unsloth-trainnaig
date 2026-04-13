import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import json

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj","v_proj"])

records = [json.loads(l) for l in open("treino.jsonl") if l.strip()]
for r in records: r["text"] = tokenizer.apply_chat_template(r["messages"], tokenize=False)

dataset = Dataset.from_list(records)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        max_length=512,
        dataset_text_field="text",
    ),
)
trainer.train()

model.save_pretrained_gguf("outputs/gguf", tokenizer, quantization_method="q4_k_m")