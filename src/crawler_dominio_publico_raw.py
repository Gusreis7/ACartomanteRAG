import csv
import re
import time
import requests

from urllib.parse import urljoin
from bs4 import BeautifulSoup

BASE_URL = "https://dominiopublico.mec.gov.br/"

LIST_URL = (
    BASE_URL + "pesquisa/ResultadoPesquisaObraForm.do"
    "?first=50"
    "&skip={skip}"
    "&ds_titulo="
    "&co_autor="
    "&no_autor=Machado%20de%20Assis"
    "&co_categoria=2"
    "&pagina={pagina}"
    "&select_action=Submit"
    "&co_midia=2"
    "&co_obra="
    "&co_idioma=1"
    "&colunaOrdenar=null"
    "&ordem=null"
)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


session = requests.Session()
session.headers.update(HEADERS)


def limpar_texto(texto: str) -> str:
    """
    Remove espaços duplicados e quebras de linha.
    """
    return re.sub(r"\s+", " ", texto or "").strip()


def parse_acessos(valor: str):
    """
    Converte valores como:

        '1.234'   -> 1234
        '114.081' -> 114081
        '234'     -> 234

    Retorna None caso não encontre número.
    """

    valor = limpar_texto(valor)

    numeros = re.sub(r"\D", "", valor)

    if not numeros:
        return None

    return int(numeros)


def get_download_links(url_detalhe: str):
    """
    Abre a página específica de uma obra e procura
    links candidatos a download.

    Não baixa o PDF.
    """

    resp = session.get(
        url_detalhe,
        timeout=30,
    )

    resp.raise_for_status()

    soup = BeautifulSoup(
        resp.text,
        "html.parser",
    )

    links = []

    for a in soup.find_all("a", href=True):

        href = limpar_texto(a.get("href"))

        if not href:
            continue

        href_lower = href.lower()

        texto_link = limpar_texto(
            a.get_text(
                " ",
                strip=True,
            )
        ).lower()

        #
        # Possíveis formas utilizadas pelo
        # Domínio Público para disponibilizar
        # o arquivo.
        #
        eh_download = (
            ".pdf" in href_lower
            or "downloadobra" in href_lower
            or "download" in href_lower
            or "baixar" in texto_link
            or "download" in texto_link
        )

        if not eh_download:
            continue

        url = urljoin(
            url_detalhe,
            href,
        )

        if url not in links:
            links.append(url)

    return links


def parse_linha_obra(
    tr,
    url_pagina: str,
    debug: bool = False,
):
    """
    Converte uma <tr> da tabela de resultados
    em um registro estruturado.

    Estrutura observada no Domínio Público:

        [0] número
        [1] link/ícone de detalhe (texto vazio)
        [2] título
        [3] autor
        [4] fonte
        [5] formato
        [6] tamanho
        [7] acessos
    """

    # Somente TDs diretamente pertencentes à TR
    tds = tr.find_all(
        "td",
        recursive=False,
    )

    if not tds:
        return None

    #
    # Procura a célula que contém o link
    # DetalheObraForm.do.
    #
    link_detalhe = None
    indice_link = None

    for indice, td in enumerate(tds):

        link = td.find(
            "a",
            href=lambda href: (href and "DetalheObraForm.do" in href),
        )

        if link is not None:
            link_detalhe = link
            indice_link = indice
            break

    #
    # Não é uma linha de obra.
    #
    if link_detalhe is None:
        return None

    if debug:
        print("\n[DEBUG] TDs encontradas:")

        for indice, td in enumerate(tds):
            texto = limpar_texto(
                td.get_text(
                    " ",
                    strip=True,
                )
            )

            print(f"    [{indice}] {texto}")

        print(f"    [DEBUG] Link detalhe encontrado na TD [{indice_link}]")

    #
    # ========================================================
    # DESCOBRIR ONDE ESTÁ O TÍTULO
    # ========================================================
    #
    # No site, a célula que contém o link de detalhe
    # pode estar vazia:
    #
    # [1] ""
    # [2] "Balas de Estalo"
    #
    # Então procuramos a primeira TD não vazia
    # DEPOIS da célula do link.
    #

    indice_titulo = None

    for indice in range(
        indice_link,
        len(tds),
    ):

        texto = limpar_texto(
            tds[indice].get_text(
                " ",
                strip=True,
            )
        )

        if texto:
            indice_titulo = indice
            break

    if indice_titulo is None:
        print("[AVISO] Link de detalhe encontrado, " "mas título não identificado.")
        return None

    #
    # A partir daqui esperamos:
    #
    # indice_titulo     -> título
    # indice_titulo + 1 -> autor
    # indice_titulo + 2 -> fonte
    # indice_titulo + 3 -> formato
    # indice_titulo + 4 -> tamanho
    # indice_titulo + 5 -> acessos
    #

    if len(tds) <= indice_titulo + 5:

        print("[AVISO] Linha não possui " "quantidade suficiente de campos.")

        return None

    titulo = limpar_texto(
        tds[indice_titulo].get_text(
            " ",
            strip=True,
        )
    )

    autor = limpar_texto(
        tds[indice_titulo + 1].get_text(
            " ",
            strip=True,
        )
    )

    fonte = limpar_texto(
        tds[indice_titulo + 2].get_text(
            " ",
            strip=True,
        )
    )

    formato = limpar_texto(
        tds[indice_titulo + 3].get_text(
            " ",
            strip=True,
        )
    )

    tamanho = limpar_texto(
        tds[indice_titulo + 4].get_text(
            " ",
            strip=True,
        )
    )

    acessos_texto = limpar_texto(
        tds[indice_titulo + 5].get_text(
            " ",
            strip=True,
        )
    )

    #
    # URL da página específica da obra
    #
    url_detalhe = urljoin(
        url_pagina,
        link_detalhe["href"],
    )

    print("\n" + "-" * 80)

    print(f"[OBRA] {titulo}")
    print(f"       Autor:   {autor}")
    print(f"       Fonte:   {fonte}")
    print(f"       Formato: {formato}")
    print(f"       Tamanho: {tamanho}")
    print(f"       Acessos: {acessos_texto}")
    print(f"       Detalhe: {url_detalhe}")

    #
    # Busca link real de download.
    #
    try:

        downloads = get_download_links(url_detalhe)

    except requests.RequestException as e:

        print(f"       [ERRO DOWNLOAD] {e}")

        downloads = []

    if downloads:

        for download in downloads:
            print(f"       Download: {download}")

    else:

        print("       Download: NÃO ENCONTRADO")

    return {
        "titulo": titulo,
        "autor": autor,
        "fonte": fonte,
        "formato": formato,
        "tamanho": tamanho,
        "acessos": parse_acessos(acessos_texto),
        "url_detalhe": url_detalhe,
        "url_download": (downloads[0] if downloads else None),
        "downloads_encontrados": len(downloads),
    }


def get_obras_machado(
    pagina: int,
    debug: bool = False,
):
    """
    Extrai todas as obras presentes em uma
    página da busca.

    A paginação utiliza:

        página 1 -> skip 0
        página 2 -> skip 50
        página 3 -> skip 100
        página 4 -> skip 150
        ...
    """

    skip = (pagina - 1) * 50

    url = LIST_URL.format(
        pagina=pagina,
        skip=skip,
    )

    print("\n" + "=" * 80)

    print(f"[+] Consultando página {pagina}")

    print(f"[+] Skip: {skip}")

    print(url)

    print("=" * 80)

    resp = session.get(
        url,
        timeout=30,
    )

    resp.raise_for_status()

    soup = BeautifulSoup(
        resp.text,
        "html.parser",
    )

    obras = []

    #
    # Cada TR pode corresponder a uma obra.
    #
    for tr in soup.find_all("tr"):

        obra = parse_linha_obra(
            tr=tr,
            url_pagina=url,
            debug=debug,
        )

        if obra is None:
            continue

        obras.append(obra)

    return obras


def salvar_csv(
    obras,
    arquivo="obras_machado_assis.csv",
):
    """
    Salva os metadados das obras em CSV.
    """

    if not obras:

        print("[AVISO] Nenhuma obra para salvar.")

        return

    campos = [
        "titulo",
        "autor",
        "fonte",
        "formato",
        "tamanho",
        "acessos",
        "url_detalhe",
        "url_download",
        "downloads_encontrados",
    ]

    with open(
        arquivo,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=campos,
        )

        writer.writeheader()

        writer.writerows(obras)

    print(f"\n[+] CSV salvo em: {arquivo}")


def main():

    pagina_inicial = 1
    pagina_final = 7
    DEBUG_TABELA = False
    DELAY_ENTRE_PAGINAS = 0.5

    todas_obras = []

    for pagina in range(
        pagina_inicial,
        pagina_final + 1,
    ):

        try:

            obras = get_obras_machado(
                pagina=pagina,
                debug=DEBUG_TABELA,
            )

            print(f"\n[+] Página {pagina}: " f"{len(obras)} obras encontradas.")

            todas_obras.extend(obras)

        except requests.RequestException as e:

            print(f"[ERRO HTTP] Página " f"{pagina}: {e}")

        except Exception as e:

            print(f"[ERRO] Falha processando " f"página {pagina}: {e}")

        time.sleep(DELAY_ENTRE_PAGINAS)

    #
    # ========================================================
    # DEDUPLICAÇÃO
    # ========================================================
    obras_unicas = {}

    for obra in todas_obras:

        chave = obra["url_detalhe"]

        obras_unicas[chave] = obra

    todas_obras = list(obras_unicas.values())

    #
    # ========================================================
    # RESULTADO
    # ========================================================
    #

    print("\n" + "=" * 80)

    print("RESULTADO")
    print("=" * 80)
    print(f"Total bruto encontrado: " f"{len(obras_unicas)}")
    print(f"Total de obras únicas: " f"{len(todas_obras)}")
    com_download = sum(1 for obra in todas_obras if obra["url_download"])
    sem_download = len(todas_obras) - com_download
    print(f"Com download encontrado: " f"{com_download}")
    print(f"Sem download encontrado: " f"{sem_download}")
    salvar_csv(
        todas_obras,
        "obras_machado_assis.csv",
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
