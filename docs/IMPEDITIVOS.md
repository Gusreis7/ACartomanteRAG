# Impeditivos e retomada do ambiente Raspberry Pi

Registro: 15/10/2026. Situação: execução do RAG e deploy local suspensos até a manutenção do Raspberry Pi.

## Impedimento observado

O job do Gitea falhou em `Pull images` ao extrair uma camada da imagem oficial do Ollama:

```text
Extracting 2.668GB
failed to register layer: write /usr/lib/ollama/cuda_jetpack6/libcublasLt.so.12.6.1.4: no space left on device
```

A falha registrada é falta de espaço no filesystem usado pelo Docker durante a extração da imagem. O caminho de bibliotecas CUDA faz parte da imagem; ele não demonstra uso de GPU no Raspberry. O erro ocorre antes do preparo do modelo. Trocar apenas o modelo não resolve o espaço exigido para extrair essa imagem.

Os pesos de `qwen3.5:4b`, o cache da aplicação, as imagens e os dados persistentes ainda exigem espaço adicional. A RAM disponível, a concorrência e a latência do RAG no hardware real também precisam de medição; não foram a causa comprovada deste erro.

## Decisão temporária

Preservar o Raspberry para o Gitea e o fluxo do repositório que espelha no GitHub. Pausar a carga de execução local do ACartomanteRAG até revisar armazenamento e capacidade da máquina.
