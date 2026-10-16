# Impeditivos e decisões operacionais

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

## Indexação em CPU e permanência no plano FREE

Registro: 16/10/2026. Situação: processamento mantido em CPU, com ampliação da janela de execução.

A Silver produziu 15.402 chunks. O log registrou 9.728 indexados (63%) em 1h06m41s, com cerca de 39 minutos restantes. O ritmo observado projeta aproximadamente 1h45 para o índice textual; o enrichment e seu índice ainda vêm depois. O limite anterior de 120 minutos no Glue era insuficiente para o processamento completo.

A decisão foi preservar o plano FREE e os US$140 de créditos informados, evitando a exposição a cobranças excedentes do plano Paid. A GPU temporária foi adiada: aceleraria a indexação, mas exigiria outro executor, acesso e quota compatíveis. O enrichment continuaria no Bedrock. A API permanece em CPU; não se altera a instância para executar esse ETL.

| Controle | Configuração adotada |
|---|---|
| Glue | Dois workers `G.1X`, limite de 600 minutos, uma execução simultânea e nenhuma repetição automática |
| GitHub Actions | Um início do Glue e cinco janelas sequenciais de acompanhamento, de até 135 minutos |
| Credenciais | Nova sessão OIDC em cada job, preservando a role atual de três horas |
| Publicação | Confirmação de `SUCCEEDED` antes do encerramento do workflow e da recarga da API |

Cada janela acompanha o mesmo identificador `jr_`; não inicia outro processamento. O bloqueio `aws-production` permanece durante todo o workflow. Essa divisão respeita o limite de seis horas por job do GitHub e evita depender de uma única sessão OIDC durante o ETL inteiro.

Como referência de planejamento, duas DPUs a US$0,44 por hora representam US$0,88/h e aproximadamente US$8,80 de computação em uma execução de dez horas. Bedrock, armazenamento e demais recursos têm consumo adicional. O orçamento preliminar de US$5–15 para a carga inicial não é um teto; depende do tempo, dos tokens e das retomadas. O budget do projeto envia alerta, sem desligamento automático configurado.

O novo prazo exige um plano Terraform atualizado e vale para execuções seguintes. Para retomar o processamento existente, manter Raw, `run_id` e SHA da revisão ETL. As etapas concluídas são reutilizadas; o índice parcial ainda recalcula os lotes e faz upsert dos mesmos IDs. Uma interrupção forçada pode deixar o lock do ETL no S3; sua remoção exige confirmar que não existe job ativo. O procedimento está em [AWS_AUTOMATION.md](AWS_AUTOMATION.md).

Referências: [preços do Glue](https://aws.amazon.com/glue/pricing/), [planos e créditos AWS](https://aws.amazon.com/free/free-tier-faqs/) e [limites do GitHub Actions](https://docs.github.com/en/actions/reference/limits).
