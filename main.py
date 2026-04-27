import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
# 1. IMPORTAÇÃO NOVA AQUI
from transformers import EarlyStoppingCallback 
import json
from pathlib import Path

max_seq_length = 2048 

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model, 
    r=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

role_map = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}

def to_messages(record):
    # ... (seu código to_messages fica igualzinho) ...
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

# 2. DIVIDIR O DATASET AQUI
dataset_completo = Dataset.from_list(records)
# Tira 5% dos dados aleatoriamente para servir de "prova surpresa"
dataset_dividido = dataset_completo.train_test_split(test_size=0.05, seed=42)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset_dividido["train"], # Treina aqui
    eval_dataset=dataset_dividido["test"],   # Faz a prova surpresa aqui
    
    # 3. ATIVA O FREIO AUTOMÁTICO AQUI
    # patience=3 significa: se o loss de validação não melhorar por 3 testes seguidos, ele corta o treino na hora.
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    
    args=SFTConfig(
        output_dir="./outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True, 
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        
        # Pode botar 5, 10 ou até as 25 épocas agora, porque se ficar burro, o callback mata o processo antes!
        num_train_epochs=5,  # numero máximo de épocas, mas o callback pode cortar antes se não tiver melhoras no eval_loss
        
        # --- CONFIGURAÇÕES DO TESTE SURPRESA ---
        eval_strategy="steps",  
        eval_steps=50,           # A cada 50 passos ele testa com os 5% separados
        save_strategy="steps",   
        save_steps=50,           # Salva um checkpoint junto com o teste
        load_best_model_at_end=True, # Quando o treino acabar (ou for abortado), carrega os pesos de quando ele estava mais inteligente!
        metric_for_best_model="eval_loss",
    ),
)
trainer.train()

# Quando o código chegar aqui, ele já descartou as épocas ruins (se teve) e vai exportar só o filé:
model.save_pretrained_gguf("outputs/gguf", tokenizer, quantization_method="q4_k_m")