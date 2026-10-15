# ACartomanteRAG — AWS, GitHub e ETL

Pacote corrigido em 15/10/2026. Aplicar sobre a beta com os patches anteriores e as versões atuais de `.gitignore`, `README.md` e `docs/INFRA_LOCAL_AND_DEPLOYMENT.md` enviadas nesta conversa. O layout dessa versão foi preservado.

```bash
bash apply_patch.sh /home/gustavoreis/Projetos/Homelab/ACartomanteRAG
```

Substituir os arquivos do pacote anterior pelos desta versão; não é necessário desfazer a tentativa recusada, pois ela não alterou o projeto. Usar o `apply_patch.sh` e o `changes.patch` deste mesmo ZIP.

O `.gitignore` existente recebe somente as novas regras de infraestrutura. O README original é preservado, com uma seção adicional de execução e documentação. O guia de infraestrutura parte da versão atual, sem restaurar os trechos locais que foram removidos. Os outros 60 arquivos do diff mantêm a implementação anterior.

O aplicador valida o diff inteiro antes de escrever, reconhece reaplicação e recusa conflitos. Quando a base diverge, mostra os arquivos e trechos informados pelo Git. Não instala bibliotecas, altera dados, envia commits ou executa serviços.

## Entregas

- `docker-compose.local.yml` reutiliza o Compose LAN. `docker-compose.aws.yml` contém API, Redis Stack e Caddy, com somente HTTPS público no IP específico e Redis administrativo no loopback.
- S3 direto por IAM, Bedrock Converse com Qwen3 32B e suporte configurável ao Scout. Enrichment de produção continua exclusivamente no Qwen.
- Terraform em `infra/aws`, bootstrap de estado S3 e operação via SSM. EBS e Elastic IP independentes da EC2; backups Redis e retorno à imagem anterior quando o novo deploy falha.
- Jobs Glue 5.1 reutilizando os módulos existentes, locks/checkpoints, migração seletiva e validação dos índices antes de uma promoção de dados única. Resumos concluídos são reutilizados, inclusive na retomada da obra.
- GitHub Actions: validação na `develop` e nos PRs; validação e deploy na `main`, com OIDC por ID de repositório e branch. `scripts/configure_github.py` aplica a proteção real da branch pelo GitHub CLI. Os workflows do Gitea não vieram no ZIP e não foram alterados.
- Import opcional do cluster Qdrant real para Terraform, preservando seu pacote e recusando alterações remotas.

`infra/` concentra a infraestrutura de cada provedor; `scripts/` reúne os comandos de implantação e empacotamento; `tests/` verifica os novos contratos. As regras de ETL, geração e armazenamento continuam nos seus módulos `src/`.

## Primeira publicação

O roteiro completo ficou em `docs/INFRA_LOCAL_AND_DEPLOYMENT.md`, em primeira pessoa, junto da operação local. Não há credenciais ou IDs de conta inventados no pacote.

1. Versionar o patch na branch de trabalho; definir repositório GitHub, e-mails e escolhas no `terraform.tfvars` privado.
2. Preparar backend, revisar `terraform plan` e aplicar infraestrutura com AWS SSO/profile.
3. Gravar os segredos via `scripts/cloud.py secrets`; usar chaves Qdrant diferentes para leitura e ETL.
4. Delegar o subdomínio no DNS atual; preparar bundle e smoke Glue antes de processar o corpus.
5. Migrar os artefatos aprovados, reutilizar a collection textual e executar enrichment + índice auxiliar, promovendo os dados explicitamente.
6. Configurar proteção da `main` e variáveis pelo script GitHub; então usar o fluxo `develop` → PR → `main` para o deploy.

A configuração do GitHub só é aplicada ao executar o script autenticado. A delegação inicial de DNS no provedor do domínio permanece uma configuração externa. O código permite usar uma zona Route 53 já existente ou delegar somente o subdomínio da aplicação.

A comparação Qwen × Scout nos benchmarks/HTMLs pertence à próxima etapa do plano; os relatórios anteriores permanecem como resultados do baseline local.

## Validação

- 28 testes passaram: configuração local/cloud, IAM no cliente S3, Converse/JSON/retries, readiness sem geração, cotas Redis, migração sem overwrite, cache do enrichment, promoção de dados e restrições do pacote de deploy.
- Sintaxe Python, parser dos Composes local/AWS, configuração Caddy e sintaxe/formatação HCL verificadas.
- Template cloud-init renderizado e checado com Bash; user data comprimido abaixo do limite EC2.
- Bundles de deploy/Glue conferidos, sem dados, credenciais ou caches; scanner de segredos passou na árvore recebida e no histórico disponível desta cópia.
- Aplicação, reaplicação e recusa de conflito do ZIP verificadas em cópias da base.
- Nesta correção, a base usa os três arquivos atuais enviados. Foram conferidos a preservação do README e das regras existentes, os links da documentação e a equivalência dos outros 60 arquivos com a entrega anterior. Também foram verificadas a extração dentro e fora do projeto, raízes com espaços, a ausência de alterações em caso de conflito e a aplicação em árvore Git limpa ou com modificações locais.

Build Docker, execução real Glue/Bedrock, import do Qdrant e restauração em Redis ainda precisam do ambiente real. O `terraform validate` com o provider AWS ficou bloqueado aqui por proibição de sockets locais; o CI o executa no runner do GitHub. Nenhum recurso AWS foi criado nesta preparação. O histórico remoto real do seu repositório não estava no ZIP; o workflow fará a auditoria completa dele.
