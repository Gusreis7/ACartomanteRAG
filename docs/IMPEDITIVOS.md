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

## Qwen pela rota Mantle

Em 16/10/2026, o enrichment parou por throttling no Bedrock Runtime; as quotas consultadas para Qwen estavam zeradas. O mesmo Qwen3 32B respondeu pela API Chat Completions do Mantle no notebook. Adotamos essa rota com assinatura IAM das roles, mantendo o plano FREE e o processamento no Glue. O Mantle tem quotas próprias; o teste curto não garante capacidade para todo o corpus.

O preço Standard em `us-east-1` permanece US$0,15 por milhão de tokens de entrada e US$0,60 por milhão de saída, além do Glue. Smoke e ETL fazem uma inferência curta antes dos embeddings. A migração explícita do checkpoint preserva Bronze, Silver e o índice textual; o enrichment passa a ter identidade própria da nova rota. Procedimento: [AWS_AUTOMATION.md](AWS_AUTOMATION.md#qwen-no-mantle).

Fontes: [endpoints e quotas](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html), [permissões Mantle](https://docs.aws.amazon.com/service-authorization/latest/reference/list_bedrock-mantle.html) e [preços regionais](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json).

## Respostas incompletas no enrichment

A run `jr_52b202ab4666100693d0e4ba81cdf4f6e65db8392655bdaaf135f6ac25f62e2d` terminou com um erro que agrupava resposta vazia e truncamento. O log disponível não preservou o motivo original; a causa permanece sem confirmação. O Mantle também pode cortar a geração ao atingir a quota de saída, conforme a [documentação de quotas](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-mantle.html).

A decisão foi registrar motivo, tokens e request ID, com até três retries de resposta (10/20/40 segundos), preservando o orçamento de tokens por chamada. Após esgotá-los, a obra fica no manifest e o lote continua; quota, permissão ou indisponibilidade de serviço pausam o lote. Pendências bloqueiam o índice auxiliar e a promoção. A retomada reutiliza obras e unidades concluídas; os novos contadores começam em zero nos manifests antigos.

Na execução seguinte, `jr_299f37f11d5012c96cd41e6ebbd3db6d15cf1e0ab6d8c284485a950b374a9323`, o lote terminou com 253 sucessos e quatro erros. O diagnóstico fornecido mostrou `length` exatamente no teto de 1.024 tokens. Repetir esse teto não resolveu o caso. A decisão foi permitir 2.048 e até 4.096 tokens apenas nos retries após esse corte, mantendo o máximo de três retries. O novo teto é registrado como política de recuperação; os resultados concluídos são preservados. As quatro obras serão retomadas no mesmo dataset. Código 2 continua impedindo índice auxiliar e promoção enquanto houver erro; o log mostrado não comprova falta de quota.

Na retomada, `Memórias Póstumas de Brás Cubas` concluiu, mas apareceram respostas curtas com `stop` durante `O almada`. O rótulo `invalid_json` também cobria schema incompatível e estrutura vazia; sem a saída rejeitada, a causa exata não pôde ser confirmada. A decisão foi separar os motivos e guardar a última resposta rejeitada no S3 de enrichment, com schema e contexto da unidade. Os logs mantêm somente metadados e o caminho do objeto. A instrumentação não aceita respostas vazias nem altera prompts, regras ou resultados concluídos; ela fornece evidência para a correção seguinte.
