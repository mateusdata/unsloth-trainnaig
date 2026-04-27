import os
# Dá um fôlego pro gerenciamento de memória da GPU não engasgar
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import EarlyStoppingCallback 
import json
from pathlib import Path

# Vou deixar 2048 pra não estourar a VRAM usando batch 4
max_seq_length = 2048 

# 1. Puxando o Qwen 2.5 da massa
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# 2. Ativando todos os módulos alvo pra deixar o modelo inteligente de verdade
model = FastLanguageModel.get_peft_model(
    model, 
    r=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 3. Organizando meu dataset
role_map = {"system": "system", "human": "user", "gpt": "assistant"}

def to_messages(record):
    # Pega 'messages' ou 'conversations', o que vier no meu jsonl
    conversations = record.get("messages") or record.get("conversations")
    if conversations is None: 
        raise KeyError("Ih, esqueci de colocar 'messages' ou 'conversations' no registro")
    
    messages = []
    for message in conversations:
        role = role_map.get(message.get("from"))
        if role is None: continue
        content = message.get("content", message.get("value", ""))
        messages.append({"role": role, "content": content})
    return messages

records = []
dataset_path = Path("treino.jsonl")

if not dataset_path.exists():
    raise FileNotFoundError("Cadê o treino.jsonl? Esqueci de colocar na pasta!")

with dataset_path.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        try:
            record = json.loads(line)
            messages = to_messages(record)
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            records.append({"text": text})
        except Exception as e:
            print(f"Deu ruim nessa linha: {e}")

# Separo 5% dos meus dados pra fazer aquela 'prova surpresa' e ver se ele aprendeu mesmo
dataset_completo = Dataset.from_list(records)
dataset_dividido = dataset_completo.train_test_split(test_size=0.05, seed=42)

# 4. Configurando o treino pra exigir o máximo da minha 3060
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset_dividido["train"],
    eval_dataset=dataset_dividido["test"],
    # Se o modelo parar de evoluir no teste surpresa 3 vezes seguidas, eu mando parar tudo
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    args=SFTConfig(
        output_dir="./outputs",
        
        # --- CONFIG TURBO DA MINHA GPU ---
        per_device_train_batch_size=4,      # Mando 4 exemplos de uma vez pra GPU trabalhar
        gradient_accumulation_steps=1,      # Quero que ele aprenda a cada passo, sem enrolação
        learning_rate=2e-4,                 # Taxa de aprendizado pra ele pegar o jeito rápido
        bf16=True,                          # Minha placa aguenta BF16, então vou usar que é mais rápido
        fp16=False,
        # ---------------------------------
        
        gradient_checkpointing=True,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        num_train_epochs=3, # 3 épocas pra ser jogo rápido e ele não ficar 'papagaio'
        
        # Como eu quero acompanhar o progresso e salvar o melhor modelo
        eval_strategy="steps",  
        eval_steps=20,                      
        save_strategy="steps",   
        save_steps=20,
        save_total_limit=3,                 # Pra não entupir meu SSD de checkpoint
        load_best_model_at_end=True,        # No final, pega a versão mais inteligente pra salvar
        metric_for_best_model="eval_loss",
        logging_steps=1,                    # Quero ver o log mudando a cada passo na tela
    ),
)

# 5. Manda brasa no treino!
trainer.train()

# 6. Salva logo em GGUF pra eu testar no LM Studio ou Ollama
model.save_pretrained_gguf("outputs/gguf", tokenizer,  quantization_method="q4_k_m")