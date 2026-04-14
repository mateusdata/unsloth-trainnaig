# unsloth-trainnaig

## Como rodar

### 1. Instalar dependências
```bash
uv sync
```

### 2. Treinar e exportar o modelo
```bash
uv run main.py
```

### 3. Registrar no Ollama
```bash
#sed -i "s|FROM Qwen2.5-0.5B-Instruct.Q4_K_M.gguf|FROM $(pwd)/outputs/gguf_gguf/Qwen2.5-0.5B-Instruct.Q4_K_M.gguf|" outputs/gguf_gguf/Modelfile

#FROM /media/data/hd-externo-1tb/projects-hd/dev-container/projects/unsloth-trainnaig/outputs/gguf_gguf/Qwen2.5-7B-Instruct.Q4_K_M.gguf

sed -i "s|FROM Qwen2.5-7B-Instruct.Q4_K_M.gguf|FROM $(pwd)/outputs/gguf_gguf/Qwen2.5-7B-Instruct.Q4_K_M.gguf|" outputs/gguf_gguf/Modelfile
ollama create meu-modelo -f outputs/gguf_gguf/Modelfile
ollama run meu-modelo
```