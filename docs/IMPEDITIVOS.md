# Restrições operacionais

## Raspberry Pi: runtime suspenso

O Docker falhou ao extrair a imagem do Ollama por falta de espaço (`no space left on device`). A falha antecedeu o download do modelo; escolher um modelo menor não resolve o espaço necessário à extração da imagem.

O Raspberry permanece dedicado ao Gitea e ao espelhamento do GitHub. Os workflows locais fazem checagens estáticas; seu sucesso não comprova inicialização ou deploy do RAG. Notebook e publicação AWS seguem independentes.

Para retomar o runtime:

1. Revisar o filesystem usado pelo Docker e disponibilizar espaço para imagens, pesos, cache e dados.
2. Preservar volumes, credenciais e o nome do projeto Compose.
3. Medir RAM, latência e concorrência no hardware real.
4. Testar o overlay do Raspberry e restaurar as verificações de runtime/deploy dos workflows.

`make raspi-up RASPI_RUNTIME=1` é uma liberação manual após essa revisão; não corrige capacidade nem reativa etapas desabilitadas dos workflows. Os atalhos atuam no host atual, sem estabelecer SSH.

## MEC: coleta fora do Glue

Em 15/10/2026, o Glue recebeu HTTP 403 nas sete páginas do Domínio Público; a mesma consulta no notebook retornou 200. A causa exata não foi determinada.

A coleta única no notebook grava PDFs e manifest diretamente no S3 Raw. Bronze, Silver, embeddings e enrichment continuam no Glue por `run-from-raw`. O Raw só segue como completo após validação do recibo. Se o acesso à fonte também falhar no notebook, a coleta fica pendente.

Procedimento: [coleta Raw](REPRODUCAO_AWS.md#3-coletar-somente-o-raw-no-notebook).

## CPU e janela de processamento

A geração de embeddings permanece em CPU, preservando a opção de plano FREE e adiando um executor GPU. Em uma execução com 15.402 chunks, o ritmo observado projetou cerca de 1h45 apenas para o índice textual; esse registro não é uma previsão para outras máquinas ou execuções.

O Glue usa dois workers `G.1X`, timeout de 600 minutos, uma execução simultânea e nenhum retry automático do job. O GitHub acompanha o mesmo `jr_` em cinco janelas com renovação OIDC. Trocar o timeout do job não altera uma execução já iniciada.

Cancelar o workflow não garante parar o Glue. A retomada preserva Raw, `run_id` e revisão; um índice parcial pode recalcular lotes. Locks só devem ser removidos após confirmar que o produtor encerrou.

## Bedrock e quotas

O Qwen no Bedrock Runtime apresentou throttling; uma chamada equivalente funcionou no Mantle. A geração AWS usa Mantle com IAM/SigV4, mantendo Converse para Scout. Uma chamada curta bem-sucedida não comprova capacidade para todo o corpus.

Acesso e quotas dependem da conta, do modelo, da rota e da região. O preflight verifica uma chamada pequena antes do trabalho pesado. Limites do chat, orçamento de alerta e parada de emergência não formam um teto de gastos; armazenamento e IP podem continuar sendo cobrados.

## Respostas incompletas no enrichment

| Falha | Tratamento |
| --- | --- |
| Resposta vazia, truncada ou inválida | Até três novas tentativas, com espera de 10/20/40 s; depois, erro no manifest. |
| Corte exatamente no teto de saída | Retry pode ampliar 1.024 → 2.048 → 4.096 tokens; não aumenta o teto para qualquer erro. |
| Quota, permissão ou indisponibilidade | Tratamento limitado de transporte e pausa do lote. |
| Unidade formada só por numeral romano e linhas pontilhadas | Ignorada antes da inferência, com registro em `plan.skipped_units`. |

A triagem das unidades ornamentais resolveu o caso observado em *O almada*, sem aceitar resultados vazios para trechos com conteúdo. Obras e unidades concluídas são reutilizadas. Pendências impedem índice auxiliar e promoção; código de saída 2 indica resultado salvo ainda incompleto.

Diagnóstico e atualização de revisão: [retomar e diagnosticar o ETL](REPRODUCAO_AWS.md#retomar-e-diagnosticar-o-etl).
