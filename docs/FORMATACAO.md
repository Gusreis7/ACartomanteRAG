# Formatação antes do commit

O projeto usa Black 26.10.1 para Python: 88 colunas, sintaxe compatível com
Python 3.11+, aspas e quebras de linha no padrão estável do Black. Arquivos
Terraform usam `terraform fmt` 1.11.4. Black não formata YAML, HTML ou CSS.
As dependências ficam no perfil `format` do `pyproject.toml`.

## Ativação

Na raiz do projeto, com o ambiente virtual existente:

```bash
make hooks PYTHON=.venv/bin/python
```

Execute uma vez por clone ou após mover/recriar o ambiente virtual. A instalação
baixa as ferramentas e prepara o hook do Git. Terraform deve estar no `PATH`
quando houver arquivos Terraform no commit; commits apenas de Python não o exigem.

## Uso

Continue usando `git add` e `git commit`, inclusive pela IDE quando ela executar
os hooks do Git. O hook verifica os arquivos incluídos no commit. Se ajustar
algum deles, o commit é interrompido: confira `git diff`, faça `git add` somente
dos arquivos revisados e repita o commit. Não há `git add` automático.

Edições fora do staging são guardadas e restauradas pelo pre-commit. Se houver
conflito com a formatação, ele desfaz a correção automática e preserva sua edição;
revise o arquivo antes de tentar novamente.

```bash
make format PYTHON=.venv/bin/python        # formata o projeto inteiro
make format-check PYTHON=.venv/bin/python  # apenas confere
```

A adoção é gradual: o patch não reformata o código legado. Os comandos acima
percorrem todo o Python; Terraform recebe somente arquivos versionados, sem
incluir `.tfvars` privados não versionados.

## CI e espelho

O GitHub confere Python alterado no intervalo do push antes do deploy. Em um
disparo manual, confere o último commit; sem commit anterior, confere a árvore
inicial. A validação Terraform existente permanece. O CI não envia commits de
formatação: o fluxo continua notebook → Gitea → GitHub.

Hooks são locais e podem ser ignorados por clientes; por isso a checagem do CI
também é necessária. Referências: [Black](https://black.readthedocs.io/en/stable/)
e [pre-commit](https://pre-commit.com/).
