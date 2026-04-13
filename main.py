import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. CARREGANDO O MODELO "PENA" (0.5B)
# ==========================================
# Reduzimos o modelo para meio bilhão de parâmetros para não dar erro 137
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print("🔥 Carregando o Qwen2.5 (0.5B) na CPU... Esse é bem mais leve!")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu", 
    torch_dtype=torch.float32,
)

# ==========================================
# 2. CONFIGURANDO LORA E SALVANDO MEMÓRIA
# ==========================================
peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)
model = get_peft_model(model, peft_config)

# O SEGREDO MÁGICO: Apaga cálculos inúteis da RAM durante o treino
model.gradient_checkpointing_enable()

# ==========================================
# 3. PREPARANDO SEU treino.jsonl
# ==========================================
def preprocess_function(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    
    # max_length=512 consome MUITO menos memória que 1024
    result = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("📚 Processando JSONL...")
dataset = load_dataset("json", data_files={"train": "treino.jsonl"})

tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ==========================================
# 4. TREINAMENTO À PROVA DE BALAS
# ==========================================
print("🚀 Iniciando o treinamento... Adeus Erro 137!")

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=30, 
        learning_rate=2e-4,
        logging_steps=1,
        use_cpu=True, 
        remove_unused_columns=False,
        gradient_checkpointing=True, # Avisa o Trainer para economizar RAM
    ),
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# ==========================================
# 5. SALVANDO O RESULTADO
# ==========================================
print("💾 Salvando os pesos treinados...")
trainer.model.save_pretrained("modelo_treinado_lora")
tokenizer.save_pretrained("modelo_treinado_lora")
print("✅ ACABOU A SOFRÊNCIA! Modelo treinado com sucesso.")