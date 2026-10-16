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

## Acesso ao MEC pelo crawler no Glue

O job Glue recebeu HTTP 403 nas sete páginas consultadas do portal Domínio Público. A mesma consulta via Python/Requests no notebook retornou HTTP 200, com `server: cloudflare`. O motivo exato da diferença não foi identificado; não é um erro de permissão S3 ou de credenciais do GitHub demonstrado por esses logs.

Por decisão operacional, executar somente o crawler no notebook uma vez, enviando os PDFs e o manifest diretamente ao S3 Raw. A coleta publica um recibo de conclusão após verificar páginas, downloads e integridade dos objetos. Bronze, Silver, embeddings e enrichment permanecem no Glue, usando o modo `run-from-raw` e sem voltar ao portal. Não se migram artefatos processados locais.

O procedimento, os checkpoints e as limitações estão em [AWS_RAW_CRAWL.md](AWS_RAW_CRAWL.md). Não há retomada do runtime no Raspberry nem coleta automática recorrente. Se o notebook também passar a receber 403, a coleta permanece bloqueada até resolver o acesso à fonte; um Raw parcial não será promovido como completo.
