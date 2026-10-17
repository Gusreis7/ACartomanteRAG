SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Execute na raiz do projeto ou use make -C /caminho/do/projeto.
PROJECT_ROOT := $(CURDIR)
DOCKER ?= docker
PYTHON ?= python3
GH ?= gh
ENV_FILE ?= config/.env
GITHUB_REPO ?= Gusreis7/ACartomanteRAG
PROFILE ?= ci
SERVICE ?= api
LOG_LINES ?= 100
RUN ?=
RASPI_RUNTIME ?= 0

COMPOSE = $(DOCKER) compose --project-directory "$(PROJECT_ROOT)" --env-file "$(ENV_FILE)"
LOCAL_COMPOSE = $(COMPOSE) -f "$(PROJECT_ROOT)/docker-compose.local.yml"
RASPI_COMPOSE = $(COMPOSE) -f "$(PROJECT_ROOT)/docker-compose.yml" -f "$(PROJECT_ROOT)/docker-compose.raspberry.yml"

.PHONY: help local local-up local-check local-url local-ready local-logs local-status \
        local-restart local-stop local-down raspi raspi-check raspi-config raspi-up \
        raspi-logs raspi-status raspi-stop raspi-down aws aws-deploy aws-status \
        aws-watch aws-logs aws-plan aws-apply aws-stop aws-resume deps hooks format format-check test check \
        _docker _env _gh _run _raspi-runtime

help: ## Lista os atalhos disponíveis.
	@awk -F ':.*## ' '/^[a-z][a-z0-9-]*:.*## / { printf "  %-19s %s\n", $$1, $$2 }' "$(PROJECT_ROOT)/Makefile"
	@printf '%s\n' '' 'Opções: DOCKER="sudo docker"  ENV_FILE=config/.env  PYTHON=.venv/bin/python' \
	  '        SERVICE=api  LOG_LINES=100  PROFILE=ci  GITHUB_REPO=owner/repo  RUN=<id>' \
	  'Local: API_BIND_IP/API_PORT seguem o .env; a URL efetiva aparece após o up.' \
	  'Raspberry: runtime suspenso; raspi-up exige RASPI_RUNTIME=1 após manutenção.' \
	  'AWS: usa gh autenticado; workflows executam somente a main já espelhada.'

local: local-up ## Constrói e inicia a aplicação no notebook.

local-up: local-check ## Sobe o Compose local, preservando volumes e credenciais.
	@$(LOCAL_COMPOSE) up --detach --build
	@$(MAKE) --no-print-directory local-url
	@printf '%s\n' 'A API pode estar carregando os modelos. Confira com make local-ready ou make local-logs.'

local-check: _docker _env ## Valida o Compose local sem imprimir a configuração privada.
	@$(LOCAL_COMPOSE) config --quiet

local-url: _docker _env ## Mostra a URL publicada pelo Docker, inclusive IP e porta reais.
	@address=$$($(LOCAL_COMPOSE) port api 8000); \
	  [[ -n "$$address" ]] || { printf '%s\n' 'Erro: API sem porta publicada. Execute make local-status.' >&2; exit 1; }; \
	  printf 'Aplicação: http://%s\n' "$$address"

local-ready: _docker _env ## Consulta /ready no endereço realmente publicado.
	@address=$$($(LOCAL_COMPOSE) port api 8000); \
	  [[ -n "$$address" ]] || { printf '%s\n' 'Erro: API sem porta publicada. Execute make local-status.' >&2; exit 1; }; \
	  curl --fail-with-body --silent --show-error --max-time 30 "http://$$address/ready"; \
	  printf '\n'

local-logs: _docker _env ## Acompanha os logs; SERVICE=api e LOG_LINES=100 por padrão.
	@$(LOCAL_COMPOSE) logs --tail "$(LOG_LINES)" --follow "$(SERVICE)"

local-status: _docker _env ## Lista os containers e as portas locais.
	@$(LOCAL_COMPOSE) ps --all

local-restart: local-check ## Recria somente a API para aplicar variáveis de ambiente.
	@$(LOCAL_COMPOSE) up --detach --no-deps --force-recreate api
	@$(MAKE) --no-print-directory local-url

local-stop: _docker _env ## Para os containers locais, preservando os dados.
	@$(LOCAL_COMPOSE) stop

local-down: _docker _env ## Remove containers e redes locais, preservando os volumes.
	@$(LOCAL_COMPOSE) down

raspi: raspi-check ## Executa as checagens do Raspberry; runtime continua suspenso.
	@printf '%s\n' 'Após a manutenção, make raspi-up RASPI_RUNTIME=1 inicia o runtime no próprio Raspberry.'

raspi-check: _docker ## Executa as checagens estáticas usadas pelo Gitea.
	@bash scripts/check_raspberry_ci.sh

raspi-config: _docker _env ## Valida o Compose base com o overlay do Raspberry.
	@$(RASPI_COMPOSE) config --quiet

raspi-up: _raspi-runtime raspi-config ## Inicia no Raspberry após manutenção; inclui Ollama em Docker.
	@$(RASPI_COMPOSE) up --detach --build

raspi-logs: _docker _env ## Acompanha logs no Raspberry; SERVICE=ollama também é aceito.
	@$(RASPI_COMPOSE) logs --tail "$(LOG_LINES)" --follow "$(SERVICE)"

raspi-status: _docker _env ## Lista containers no host Raspberry onde o comando for executado.
	@$(RASPI_COMPOSE) ps --all

raspi-stop: _docker _env ## Para os containers do RAG no Raspberry, preservando dados.
	@$(RASPI_COMPOSE) stop

raspi-down: _docker _env ## Remove containers e redes do RAG no Raspberry, preservando volumes.
	@$(RASPI_COMPOSE) down

aws: aws-deploy ## Dispara CI, Terraform e deploy pelo GitHub na main espelhada.

aws-deploy: _gh ## Dispara CI and AWS deploy; respeita AWS_APP_DEPLOY_ENABLED.
	@$(GH) workflow run ci-cd.yml --repo "$(GITHUB_REPO)" --ref main
	@printf '%s\n' 'Workflow solicitado para a main do GitHub. Acompanhe com make aws-status.'

aws-status: _gh ## Lista as dez execuções mais recentes da main no GitHub.
	@$(GH) run list --repo "$(GITHUB_REPO)" --branch main --limit 10

aws-watch: _gh _run ## Acompanha uma execução do GitHub: RUN=<id>.
	@$(GH) run watch "$(RUN)" --repo "$(GITHUB_REPO)" --exit-status

aws-logs: _gh _run ## Mostra os logs de uma execução concluída: RUN=<id>.
	@$(GH) run view "$(RUN)" --repo "$(GITHUB_REPO)" --log

aws-plan: _gh ## Solicita somente o plano Terraform pelo workflow existente.
	@$(GH) workflow run infrastructure.yml --repo "$(GITHUB_REPO)" --ref main --field operation=plan

aws-apply: _gh ## Solicita plan e apply Terraform no GitHub, sem publicar a API.
	@$(GH) workflow run infrastructure.yml --repo "$(GITHUB_REPO)" --ref main --field operation=apply

aws-stop: _gh ## Dispara a emergência: para EC2/Glue e bloqueia novas operações.
	@$(GH) workflow run emergency.yml --repo "$(GITHUB_REPO)" --ref main --field operation=stop

aws-resume: _gh ## Retoma EC2 e libera operações; não reinicia ETL automaticamente.
	@$(GH) workflow run emergency.yml --repo "$(GITHUB_REPO)" --ref main --field operation=resume

deps: ## Instala o perfil do pyproject no Python escolhido; PROFILE=ci por padrão.
	@$(PYTHON) scripts/dependencies.py install "$(PROFILE)"

hooks: ## Instala Black e o hook de commit neste clone; consulte docs/FORMATACAO.md.
	@$(PYTHON) scripts/dependencies.py install format
	@$(PYTHON) -m pre_commit install --install-hooks

format: ## Formata todo o Python e os arquivos Terraform versionados.
	@$(PYTHON) -m black .
	@git ls-files -z -- '*.tf' '*.tfvars' '*.tftest.hcl' | xargs -0 -r bash scripts/format_terraform.sh

format-check: ## Confere a formatação geral sem escrever; pode apontar código legado.
	@$(PYTHON) -m black --check .
	@git ls-files -z -- '*.tf' '*.tfvars' '*.tftest.hcl' | xargs -0 -r bash scripts/format_terraform.sh --check

test: ## Executa os testes no Python escolhido, sem instalar dependências.
	@$(PYTHON) -m pytest -q

check: raspi-check ## Valida sintaxe, Composes e a definição central de dependências.
	@$(PYTHON) scripts/dependencies.py check

_docker:
	@$(DOCKER) compose version >/dev/null

_env:
	@test -f "$(ENV_FILE)" && test -r "$(ENV_FILE)" || { \
	  printf 'Erro: arquivo privado ausente ou sem leitura: %s. Preserve os volumes e as credenciais existentes.\n' "$(ENV_FILE)" >&2; exit 1; }

_gh:
	@$(GH) --version >/dev/null

_run:
	@[[ "$(RUN)" =~ ^[0-9]+$$ ]] || { printf '%s\n' 'Erro: informe RUN=<id numérico de make aws-status>.' >&2; exit 2; }

_raspi-runtime:
	@[[ "$(RASPI_RUNTIME)" == 1 ]] || { \
	  printf '%s\n' 'Erro: runtime do Raspberry suspenso por falta de espaço; consulte docs/IMPEDITIVOS.md.' \
	  'Após a manutenção, use make raspi-up RASPI_RUNTIME=1 no Raspberry.' >&2; exit 2; }
