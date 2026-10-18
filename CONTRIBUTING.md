# Contribuir

Para mexer na aplicação, basta um notebook. AWS fica para quem quiser testar a publicação; Raspberry e Gitea pertencem ao meu homelab e não entram como pedágio para abrir um PR.

## Preparar e desenvolver

Faça um fork para contribuir e clone seu repositório; para experimentar, pode clonar o original:

```bash
git clone https://github.com/Gusreis7/ACartomanteRAG.git
cd ACartomanteRAG
git switch -c sua-alteracao
python3 -m venv .venv
source .venv/bin/activate
python scripts/dependencies.py install ci
make hooks PYTHON=.venv/bin/python
```

As dependências ficam no `pyproject.toml`, por perfil; use `scripts/dependencies.py` em vez de criar outro requirements. Os testes unitários usam serviços simulados. Para uma consulta real, siga [Reprodução local](docs/REPRODUCAO_LOCAL.md), incluindo dados e índices.

| Onde mexer | Responsabilidade |
| --- | --- |
| `src/api/` e `src/api/web/` | HTTP, schemas, sessão e interface |
| `src/generation/` | Providers, saída estruturada, guard e LangGraph |
| `src/retrieval/`, `src/enrichment/` | Busca, reranking e resumos |
| `src/etl/`, `src/indexing/` | Coleta, transformação e publicação dos dados |
| `src/services/`, `src/storage/` | Regras de serviço e persistência |
| `benchmark/`, `AED/` | Avaliações e investigação do corpus |
| `scripts/`, `infra/` | Desenvolvimento, provisionamento e operação |

Conserve essas responsabilidades. Mudança de interface não precisa virar refatoração do ETL. Se alterar modelo, chunking ou schema de artefato, explique a compatibilidade e se será necessário reconstruir dados.

## Formatar e testar

```bash
make test PYTHON=.venv/bin/python
```

O hook de commit aplica Black no Python e `terraform fmt` nos arquivos Terraform alterados. Se modificar arquivos, confira o diff, adicione novamente e repita o commit. Black não formata YAML, HTML ou CSS.

`make format` e `make format-check` percorrem o projeto inteiro; use-os quando esse for o escopo, para não misturar reformatação do legado com sua feature. Para alterações de infraestrutura, com Terraform inicializado:

```bash
terraform -chdir=infra/aws fmt -check
terraform -chdir=infra/aws validate
terraform -chdir=infra/aws test
```

Os testes de permissões usam `plan` e provider simulado. Preserve `prevent_destroy`; não é preciso executar apply na AWS para passar no teste.

Não versione `.env`, tfvars reais, state, planos, credenciais, volumes ou dados de sessão. Exemplos devem ter placeholders. Inclua testes quando houver mudança de comportamento; teste também no navegador se mexer no fluxo do chat.

## Abrir o PR

Envie a branch para seu fork e abra um PR com problema resolvido, mudança e como testou. Informe efeitos sobre configurações ou artefatos, quando houver. Comandos e resultados observados ajudam mais que um relatório de intenções.

No meu fluxo, desenvolvimento e revisão passam pelo Gitea; a `main` é espelhada no GitHub, que publica na AWS. Contribuições externas podem chegar por PR no GitHub, mas precisam ser integradas à origem no Gitea antes do espelhamento. Não há outra promoção `develop` → `main` no espelho. Quem reproduzir com GitHub como origem pode usar seu fluxo normal de PR.
