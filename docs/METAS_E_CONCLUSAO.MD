Desde o começo do projeto, eu tinha alguns objetivos claros em relação às decisões de arquitetura e à solução que estava sendo construída.

## Metas

Fico feliz em dizer que **quase todas as metas foram cumpridas!**

- [x] Plataforma web para conversar sobre as obras de Machado de Assis;
- [ ] Uso de modelos de linguagem pequenos (SLMs) na construção da solução — parcialmente concluído;
- [ ] Desenvolvimento de um jogo com as informações extraídas, em uma espécie de quiz — não foi feito, mas incorporei elementos de jogos e cartomancia;
- [x] Comparação entre diferentes abordagens de RAG:
  - [x] Diferentes arquiteturas de recuperação;
  - [x] Diferentes modelos;
- [x] Pipeline de CI/CD para publicação;
- [x] Uso de soluções open source na parte de DevOps do projeto.

## Havia pedras no caminho

Mas vejam só: desenvolvi esse projeto na minha rede interna de homelab, utilizando um Raspberry Pi e um notebook com 4 GB de VRAM.

Construí a solução orientada a esse ambiente, testando modelos como:

- Llama 8B;
- Qwen3 4B.

Foi nesse ambiente que testei várias abordagens de RAG, diferentes modelos, tamanhos de chunks e métodos de retrieval, ajustando tudo com base em um dataset construído especificamente para o projeto.

No meu notebook e no meu ecossistema de deploys, tudo funcionou bem. Mas, ao migrar para a AWS, encontrei uma encruzilhada em relação a um ponto importante do projeto:

**O uso de modelos pequenos de linguagem.**

A AWS não estava disponibilizando esses modelos menores por meio do Bedrock. Tudo bem, eu ainda poderia usar uma instância EC2 e executar os modelos que testei na minha stack, certo?

Errado, em partes.

Escolhi a AWS por conta dos créditos gratuitos que ela oferece para contas novas, de até US$ 200. Porém, sem um cartão de crédito cadastrado, as instâncias disponibilizadas dentro das condições do Free Tier e dos créditos eram limitadas. Eu não tinha acesso a instâncias com GPU para executar esses modelos, por menores que fossem.

Além disso, mesmo que tivesse acesso, a escala do projeto não compensaria. Uma instância com GPU para executar um SLM seria muito mais cara por hora do que utilizar um modelo maior e melhor por meio do Bedrock Mantle, que estava disponível para mim.

Portanto, tomei a decisão de utilizar um modelo maior. O escolhido foi o **Qwen 32B**, considerando os custos, os requisitos técnicos e as limitações do que eu poderia fazer na AWS.