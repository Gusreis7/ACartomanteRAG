# Benchmarks MCQ na AWS

O workflow **AWS MCQ benchmark** usa os gráficos e a navegação de
`benchmark/report_mcq_html.py`: Geral, comparação, modelos e inspeção de
respostas. A configuração textual, o enrichment e a release de dados continuam
sendo os da aplicação; não há benchmark de retrieval nem reindexação.

## Executar

Após commit/push no Gitea e espelhamento da `main`, abra **Actions →
AWS MCQ benchmark → Run workflow**. Comece com duas questões.

| Campo | Preenchimento |
| --- | --- |
| `operation` | `run` executa; `report` regenera os relatórios de uma run salva. |
| `model` | `qwen3-32b` ou `llama4-scout`; ignorado em `report`. |
| `split` | `dev`; `holdout` apenas após congelar as escolhas. |
| `limit_queries` | `2` no teste inicial; `0` executa o split completo. |
| `run_id` | Vazio cria uma run; em `report`, informe o ID salvo. |

Os nomes mostram **Benchmark · modelo · commit · split** no GitHub. O ID
automático também inclui modelo, SHA curto, split e ID da execução. Os
relatórios registram o SHA completo e o modelo efetivamente usado.

Qwen3 32B mantém a rota Bedrock Mantle. Llama 4 Scout usa Bedrock Converse,
com o perfil `us.meta.llama4-scout-17b-instruct-v1:0`; não está disponível no
Mantle. A política Terraform do MCQ já contempla ambos. Acesso e cotas de
inferência do Scout precisam estar disponíveis na conta; selecionar o modelo
não altera o plano AWS nem suas cotas. Preços vêm da configuração existente.

## Abrir os gráficos

Ao terminar, abra a execução no GitHub e baixe **Artifacts → benchmark-…**.
Extraia o ZIP e abra `index.html`: ele contém a visão geral, gráficos de
acertos, latência p95, tokens e custo por run, com filtros de split, protocolo,
modelo e estado. Os links abrem os relatórios individuais em `runs/`.
A cópia `report.html` na raiz corresponde à execução atual.

O histórico é reconstruído a partir dos manifests salvos. DEV e HOLDOUT
não são somados. Grupos comparam as mesmas perguntas, prompt, release,
retrieval e parâmetros, permitindo variar o gerador; commits diferentes
continuam visíveis. Runs incompletas ficam identificadas como parciais.

Os arquivos permanecem privados no bucket de benchmarks:
`mcq/current/index.html` e `mcq/current/runs/<run_id>/`. CSV, Parquet,
configuração e checkpoints continuam disponíveis. O HTML funciona offline;
não precisa publicar o bucket ou configurar outro domínio.

## Retomada e relatórios antigos

Para retomar inferência, mantenha modelo, ID, SHA, split e limite originais;
use **Re-run jobs** na execução original. Respostas válidas, inclusive erradas,
são preservadas. Trocar modelo ou protocolo exige outra run.

Para atualizar o visual de uma execução antiga, escolha `operation=report` e
seu `run_id`. Essa operação usa os dados salvos, sem chamar o LLM, consultar
Qdrant ou refazer o benchmark. Preserva o commit, o modelo e o checkpoint
originais. Os outros campos do formulário não substituem os valores salvos.

Tokens e custo acumulam as tentativas registradas. O histórico mostra latência
de ponta a ponta; o relatório individual também mantém a latência de geração.
Custos de infraestrutura não entram na estimativa. OIDC, parada de emergência,
concorrência e separação DEV/holdout seguem o fluxo existente.
