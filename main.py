import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import json
from pathlib import Path

# se tiver uma gpu fuleira vc pode usar essa aqui unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit para seu pc nao passar mal 
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj","v_proj"])

role_map = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


def to_messages(record):
    conversations = record.get("messages")
    if conversations is None:
        conversations = record.get("conversations")
    if conversations is None:
        raise KeyError("record is missing 'messages' or 'conversations'")

    messages = []
    for message in conversations:
        role = role_map.get(message.get("from"))
        if role is None:
            raise ValueError(f"Unsupported role: {message.get('from')!r}")
        content = message.get("content", message.get("value", ""))
        messages.append({"role": role, "content": content})
    return messages


records = []
with Path("treino.jsonl").open(encoding="utf-8") as dataset_file:
    for line in dataset_file:
        if not line.strip():
            continue
        record = json.loads(line)
        messages = to_messages(record)
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        records.append({"text": text})

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
        num_train_epochs=25, #aumentei pra 20 pq o modelo é pequeno, mas se quiser pode deixar 3 ou 5 que já vai dar uma melhorada boa (dependendo do dataset)
        

    ),
)
trainer.train()

model.save_pretrained_gguf("outputs/gguf", tokenizer, quantization_method="q4_k_m")