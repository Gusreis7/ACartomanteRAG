#!/usr/bin/env python3
"""
Prepara a camada bronze do corpus de Machado de Assis.

Entradas:
    Dataset/metadata/obras_machado_assis_deduplicadas.csv
    Dataset/raw/**/*.pdf

Saídas:
    Dataset/bronze/*.txt
    Dataset/metadata/bronze/obras_machado_assis_bronze.csv

A camada raw não é alterada.

Metadados calculados:
    - is_carta
    - qtd_assinaturas_machado
    - is_coletanea
    - qtd_titulos_similares
    - titulos_similares
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_METADATA = (
    PROJECT_ROOT
    / "Dataset"
    / "metadata"
    / "obras_machado_assis_deduplicadas.csv"
)

DEFAULT_BRONZE_DIR = (
    PROJECT_ROOT
    / "Dataset"
    / "bronze"
)

DEFAULT_BRONZE_METADATA_DIR = (
    PROJECT_ROOT
    / "Dataset"
    / "metadata"
    / "bronze"
)

DEFAULT_OUTPUT_METADATA_NAME = (
    "obras_machado_assis_bronze.csv"
)

DEFAULT_SIMILARIDADE_THRESHOLD = 0.1
DEFAULT_QTD_TITULOS_COLETANEA = 2
DEFAULT_TAMANHO_FRAGMENTO = 1000
DEFAULT_TAMANHO_MINIMO_FRAGMENTO = 100
DEFAULT_MIN_ASSINATURAS_CARTA = 10

TITULOS_CARTA_EXPLICITOS = {
    "correspondencia de machado de assis",
    "pareceres de machado de assis",
}

TITULOS_EXCLUIR_CARTA = {
    "na academia brasileira de letras",
    "criticas teatrais",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara a camada bronze do corpus "
            "de Machado de Assis."
        )
    )

    parser.add_argument(
        "--input-metadata",
        type=Path,
        default=DEFAULT_INPUT_METADATA,
        help=(
            "CSV de metadata de entrada. "
            f"Padrão: {DEFAULT_INPUT_METADATA}"
        ),
    )

    parser.add_argument(
        "--bronze-dir",
        type=Path,
        default=DEFAULT_BRONZE_DIR,
        help=(
            "Diretório raiz da camada bronze. "
            f"Padrão: {DEFAULT_BRONZE_DIR}"
        ),
    )

    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=None,
        help=(
            "Caminho do CSV de metadata bronze. "
            "Se omitido, usa "
            "Dataset/metadata/bronze/"
            f"{DEFAULT_OUTPUT_METADATA_NAME}"
        ),
    )

    parser.add_argument(
        "--similaridade-threshold",
        type=float,
        default=DEFAULT_SIMILARIDADE_THRESHOLD,
        help=(
            "Threshold mínimo de similaridade "
            "para confirmar um título. "
            f"Padrão: {DEFAULT_SIMILARIDADE_THRESHOLD}"
        ),
    )

    parser.add_argument(
        "--qtd-titulos-coletanea",
        type=int,
        default=DEFAULT_QTD_TITULOS_COLETANEA,
        help=(
            "Quantidade mínima de títulos confirmados "
            "para classificar como coletânea. "
            f"Padrão: {DEFAULT_QTD_TITULOS_COLETANEA}"
        ),
    )

    parser.add_argument(
        "--tamanho-fragmento",
        type=int,
        default=DEFAULT_TAMANHO_FRAGMENTO,
        help=(
            "Tamanho máximo do fragmento usado "
            "na comparação textual. "
            f"Padrão: {DEFAULT_TAMANHO_FRAGMENTO}"
        ),
    )

    parser.add_argument(
        "--tamanho-minimo-fragmento",
        type=int,
        default=DEFAULT_TAMANHO_MINIMO_FRAGMENTO,
        help=(
            "Tamanho mínimo do fragmento para comparação. "
            f"Padrão: {DEFAULT_TAMANHO_MINIMO_FRAGMENTO}"
        ),
    )

    parser.add_argument(
        "--min-assinaturas-carta",
        type=int,
        default=DEFAULT_MIN_ASSINATURAS_CARTA,
        help=(
            "Número de assinaturas acima do qual o documento "
            "é classificado como correspondência. "
            f"Padrão: > {DEFAULT_MIN_ASSINATURAS_CARTA}"
        ),
    )

    return parser.parse_args()


# LIMPEZA HEADERS


def limpar_inicio_ufsc(texto: str) -> str:
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]

    inicio_conteudo = 0

    for i, linha in enumerate(linhas[:15]):
        linha_lower = linha.lower()

        # Caso quebrado pela extração:
        # "Ob"
        # "ra Completa, de Machado..."
        if (
            linha_lower == "ob"
            and i + 1 < len(linhas)
            and linhas[i + 1].lower().startswith("ra completa")
        ):
            inicio_conteudo = i + 2
            continue

        # Referências curtas, ex.: [JF. nov. 1864]
        referencia_colchetes = re.match(
            r"^\[[^\]]+\]$",
            linha.strip(),
        )

        eh_header = (
            "literatura brasileira" in linha_lower
            or "textos literários em meio eletrônico" in linha_lower
            or "de machado de assis" in linha_lower
            or "edição de referência" in linha_lower
            or "edição referência" in linha_lower
            or "publicado originalmente" in linha_lower
            or "nova aguilar" in linha_lower
            or "martins fontes" in linha_lower
            or "obra completa" in linha_lower
            or "texto-fonte:" in linha_lower
            or bool(referencia_colchetes)
        )

        if eh_header:
            inicio_conteudo = i + 1
            continue

        if i > 0:
            anterior = linhas[i - 1].lower()

            if (
                "edição de referência" in anterior
                or "edição referência" in anterior
                or "nova aguilar" in anterior
                or "obra completa" in anterior
                or "texto-fonte:" in anterior
                or anterior == "ob"
            ):
                inicio_conteudo = i + 1
                continue

        if inicio_conteudo > 0:
            break

    return "\n".join(linhas[inicio_conteudo:]).strip()


def limpar_inicio_biblioteca_nacional(texto: str) -> str:
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]

    primeiras_linhas = " ".join(linhas[:10]).lower()

    tem_header_institucional = (
        "fundação biblioteca nacional" in primeiras_linhas
        or "departamento nacional do livro" in primeiras_linhas
        or "ministério da cultura" in primeiras_linhas
    )

    # Com header institucional, "Machado de Assis" pode aparecer
    # um pouco mais adiante. Sem ele, cobre casos como
    # "Lição de Botânica" + "Machado de Assis".
    limite_busca = 15 if tem_header_institucional else 5

    for i, linha in enumerate(linhas[:limite_busca]):
        if linha.lower() == "machado de assis":
            return "\n".join(linhas[i + 1 :]).strip()

    return texto.strip()


def limpar_inicio_bv(texto: str) -> str:
    padrao_fim_header = (
        r"Estamos\s+em\s+busca\s+de\s+patrocinadores\s+e\s+voluntários"
        r".*?"
        r"<bibvirt@futuro\.usp\.br>"
        r".*?"
        r"saiba\s+como\s+isso\s+é\s+possível\."
    )

    match = re.search(
        padrao_fim_header,
        texto,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return texto.strip()

    return texto[match.end() :].strip()


def limpar_unama(doc: pymupdf.Document) -> str:
    # Na UNAMA, a primeira página inteira é header.
    return "\n".join(page.get_text() for page in doc[1:]).strip()


def extrair_texto_sem_header(
    pdf_path: Path,
    fonte: str,
) -> str:
    fonte = str(fonte).lower()

    with pymupdf.open(pdf_path) as doc:
        if doc.page_count == 0:
            return ""

        if "universidade da amazônia" in fonte:
            return limpar_unama(doc)

        primeira_pagina = doc[0].get_text()

        if "universidade federal de santa catarina" in fonte:
            primeira_pagina = limpar_inicio_ufsc(primeira_pagina)

        elif "fundação biblioteca nacional" in fonte:
            primeira_pagina = limpar_inicio_biblioteca_nacional(primeira_pagina)

        elif "biblioteca virtual do estudante brasileiro" in fonte:
            primeira_pagina = limpar_inicio_bv(primeira_pagina)

        # Biblioteca Virtual de Literatura e Universidade de Brasília:
        # sem limpeza específica validada no notebook.
        restante = "\n".join(page.get_text() for page in doc[1:])

    return "\n".join(
        trecho
        for trecho in [
            primeira_pagina,
            restante,
        ]
        if trecho.strip()
    ).strip()


# ============================================================
# LIMPEZA GERAL DA CAMADA BRONZE
# ============================================================

URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\]\)]+",
    flags=re.IGNORECASE,
)

EMAIL_RE = re.compile(
    r"<?[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}>?",
    flags=re.IGNORECASE,
)

PLACEHOLDER_NUPILL_RE = re.compile(
    r"Núcleo\s+de\s+Pesquisas\s+em\s+Informática,\s*"
    r"Literatura\s+e\s+Ling[uü]ística",
    flags=re.IGNORECASE,
)


def limpar_artefatos_textuais(texto: str) -> str:
    """
    Remove artefatos editoriais que não fazem parte do conteúdo da obra:
    URLs, e-mails e o placeholder institucional identificado no corpus.

    Não normaliza a ortografia nem altera o conteúdo literário.
    """
    texto = URL_RE.sub("", texto)
    texto = EMAIL_RE.sub("", texto)
    texto = PLACEHOLDER_NUPILL_RE.sub("", texto)

    # Remove espaços no fim das linhas.
    linhas = [linha.rstrip() for linha in texto.splitlines()]

    texto = "\n".join(linhas)

    # Evita grandes blocos de linhas vazias sem destruir parágrafos.
    texto = re.sub(
        r"\n[ \t]*\n(?:[ \t]*\n)+",
        "\n\n",
        texto,
    )

    return texto.strip()


def extrair_texto_bronze(
    pdf_path: Path,
    fonte: str,
) -> str:
    texto = extrair_texto_sem_header(
        pdf_path=pdf_path,
        fonte=fonte,
    )

    return limpar_artefatos_textuais(texto)


# ============================================================
# NORMALIZAÇÕES USADAS NAS HEURÍSTICAS
# ============================================================


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize(
        "NFKD",
        str(texto),
    )

    texto = "".join(c for c in texto if not unicodedata.combining(c))

    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def normalizar_linha(texto: str) -> str:
    return normalizar_texto(texto)


# ============================================================
# HEURÍSTICA DE CARTAS
# ============================================================

PADROES_ASSINATURA = {
    normalizar_linha(x)
    for x in [
        "MACHADO DE ASSIS",
        "M. DE ASSIS",
        "M DE ASSIS",
        "MACHADO D'ASSIS",
        "J. M. MACHADO DE ASSIS",
        "JOAQUIM MARIA MACHADO DE ASSIS",
    ]
}


def encontrar_assinaturas_machado(
    texto: str,
) -> list[dict]:
    assinaturas = []
    linhas = texto.splitlines()

    for i, linha in enumerate(linhas):
        linha_norm = normalizar_linha(linha)

        if linha_norm not in PADROES_ASSINATURA:
            continue

        inicio = max(
            0,
            i - 3,
        )

        fim = min(
            len(linhas),
            i + 2,
        )

        assinaturas.append(
            {
                "assinatura": linha.strip(),
                "contexto": "\n".join(linhas[inicio:fim]),
            }
        )

    return assinaturas


# ============================================================
# HEURÍSTICA DE COLETÂNEAS
# ============================================================


def encontrar_titulos(
    texto: str,
    titulo_atual: str,
    titulos_normalizados: dict[str, str],
) -> list[str]:
    titulo_atual_norm = normalizar_texto(titulo_atual)

    texto_norm = normalizar_texto(texto)

    encontrados = {
        titulo_original
        for titulo_original, titulo_norm in titulos_normalizados.items()
        if (titulo_norm != titulo_atual_norm and titulo_norm in texto_norm)
    }

    return sorted(encontrados)


def encontrar_ancoras_titulo(
    texto: str,
    titulo: str,
) -> list[dict]:
    titulo_norm = normalizar_texto(titulo)

    linhas = texto.splitlines()

    candidatos = []
    pos_acumulada = 0

    for i, linha in enumerate(linhas):
        linha_norm = normalizar_texto(linha)

        if titulo_norm not in linha_norm:
            pos_acumulada += len(linha) + 1
            continue

        titulo_isolado = linha_norm == titulo_norm

        linha_seguinte_vazia = i + 1 < len(linhas) and linhas[i + 1].strip() == ""

        prioridade = 0

        if titulo_isolado:
            prioridade += 2

        if linha_seguinte_vazia:
            prioridade += 1

        candidatos.append(
            {
                "linha": i,
                "posicao": pos_acumulada,
                "titulo_isolado": titulo_isolado,
                "linha_seguinte_vazia": linha_seguinte_vazia,
                "prioridade": prioridade,
            }
        )

        pos_acumulada += len(linha) + 1

    return sorted(
        candidatos,
        key=lambda x: (
            -x["prioridade"],
            x["posicao"],
        ),
    )


def comparar_fragmento_com_original(
    texto_documento: str,
    titulo_encontrado: str,
    titulos_info: dict[str, dict],
    tamanho_fragmento: int = DEFAULT_TAMANHO_FRAGMENTO,
    tamanho_minimo: int = DEFAULT_TAMANHO_MINIMO_FRAGMENTO,
) -> dict | None:
    titulo_norm = normalizar_texto(titulo_encontrado)

    ancoras = encontrar_ancoras_titulo(
        texto_documento,
        titulo_encontrado,
    )

    if not ancoras:
        return None

    info_original = titulos_info.get(titulo_norm)

    if info_original is None:
        return None

    texto_original = info_original["texto_bronze"]

    texto_original_norm = normalizar_texto(texto_original)

    pos_original = texto_original_norm.find(titulo_norm)

    if pos_original != -1 and pos_original < 500:
        texto_original_conteudo = texto_original_norm[
            pos_original + len(titulo_norm) :
        ].strip()
    else:
        texto_original_conteudo = texto_original_norm

    if not texto_original_conteudo:
        return None

    resultados_ancoras = []
    linhas_documento = texto_documento.splitlines()

    for ancora in ancoras:
        linhas_depois = linhas_documento[ancora["linha"] + 1 :]

        texto_depois_ancora_norm = normalizar_texto("\n".join(linhas_depois))

        tamanho_real = min(
            tamanho_fragmento,
            len(texto_original_conteudo),
            len(texto_depois_ancora_norm),
        )

        if tamanho_real < tamanho_minimo:
            continue

        fragmento_documento = texto_depois_ancora_norm[:tamanho_real]

        fragmento_original = texto_original_conteudo[:tamanho_real]

        similaridade = SequenceMatcher(
            None,
            fragmento_documento,
            fragmento_original,
        ).ratio()

        resultados_ancoras.append(
            {
                "similaridade": similaridade,
                "tamanho_fragmento_usado": tamanho_real,
                "titulo_isolado": ancora["titulo_isolado"],
                "linha_seguinte_vazia": ancora["linha_seguinte_vazia"],
                "prioridade_ancora": ancora["prioridade"],
            }
        )

    if not resultados_ancoras:
        return None

    melhor = max(
        resultados_ancoras,
        key=lambda x: (
            x["prioridade_ancora"],
            x["similaridade"],
        ),
    )

    return {
        "titulo": titulo_encontrado,
        **melhor,
        "qtd_ancoras_encontradas": len(ancoras),
    }


def avaliar_coletanea(
    texto: str,
    titulo_atual: str,
    titulos_normalizados: dict[str, str],
    titulos_info: dict[str, dict],
    similaridade_threshold: float = DEFAULT_SIMILARIDADE_THRESHOLD,
    qtd_titulos_coletanea: int = DEFAULT_QTD_TITULOS_COLETANEA,
    tamanho_fragmento: int = DEFAULT_TAMANHO_FRAGMENTO,
    tamanho_minimo_fragmento: int = DEFAULT_TAMANHO_MINIMO_FRAGMENTO,
) -> dict:
    titulos_encontrados = encontrar_titulos(
        texto=texto,
        titulo_atual=titulo_atual,
        titulos_normalizados=titulos_normalizados,
    )

    resultados = []

    for titulo_encontrado in titulos_encontrados:
        resultado = comparar_fragmento_com_original(
            texto_documento=texto,
            titulo_encontrado=titulo_encontrado,
            titulos_info=titulos_info,
            tamanho_fragmento=tamanho_fragmento,
            tamanho_minimo=tamanho_minimo_fragmento,
        )

        if resultado is None:
            continue

        resultados.append(
            {
                "titulo": titulo_encontrado,
                "similaridade": resultado["similaridade"],
            }
        )

    similares = [
        r
        for r in resultados
        if r["similaridade"] > similaridade_threshold
    ]

    titulos_similares = [r["titulo"] for r in similares]

    return {
        "qtd_titulos_similares": len(titulos_similares),
        "titulos_similares": titulos_similares,
        "is_coletanea": (
            len(titulos_similares)
            >= qtd_titulos_coletanea
        ),
    }


# ============================================================
# UTILITÁRIOS
# ============================================================


def resolver_pdf_path(
    pdf_path: str,
) -> Path:
    path = Path(str(pdf_path))

    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def caminho_txt_bronze(
    pdf_path: Path,
    bronze_dir: Path,
) -> Path:
    return (
        bronze_dir
        / f"{pdf_path.stem}.txt"
    )


def caminho_relativo_projeto(
    path: Path,
) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ============================================================
# PIPELINE
# ============================================================


def main() -> None:
    args = parse_args()

    input_metadata = args.input_metadata
    bronze_dir = args.bronze_dir

    output_metadata = (
        args.output_metadata
        if args.output_metadata is not None
        else (
            DEFAULT_BRONZE_METADATA_DIR
            / DEFAULT_OUTPUT_METADATA_NAME
        )
    )

    # A camada bronze contém apenas os arquivos TXT.
    bronze_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Os metadados enriquecidos ficam separados da camada textual.
    output_metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[+] Lendo metadata: {input_metadata}")

    df = pd.read_csv(input_metadata)

    # Enquanto o crawler ainda não trata o edge case de obras sem PDF,
    # removemos da camada bronze registros sem arquivo disponível.
    registros_validos = []

    for _, row in df.iterrows():
        if pd.isna(row.get("pdf_path")):
            continue

        pdf_path = resolver_pdf_path(row["pdf_path"])

        if not pdf_path.is_file():
            print("[!] PDF não encontrado, ignorando: " f"{pdf_path}")
            continue

        registro = row.to_dict()
        registro["_pdf_path_resolvido"] = pdf_path

        registros_validos.append(registro)

    df = pd.DataFrame(registros_validos)

    if df.empty:
        raise RuntimeError("Nenhum PDF válido foi encontrado.")

    textos_bronze = []

    print(f"[+] Gerando textos bronze para {len(df)} obras...")

    for _, row in df.iterrows():
        pdf_path = row["_pdf_path_resolvido"]

        texto = extrair_texto_bronze(
            pdf_path=pdf_path,
            fonte=row.get(
                "fonte",
                "",
            ),
        )

        txt_path = caminho_txt_bronze(
            pdf_path,
            bronze_dir,
        )

        txt_path.write_text(
            texto,
            encoding="utf-8",
        )

        textos_bronze.append(texto)

    df["_texto_bronze"] = textos_bronze

    titulos_normalizados = {
        titulo: normalizar_texto(titulo) for titulo in (df["titulo"].dropna().unique())
    }

    titulos_info = {}

    for _, row in df.iterrows():
        titulo_norm = normalizar_texto(row["titulo"])

        titulos_info[titulo_norm] = {
            "titulo": row["titulo"],
            "texto_bronze": row["_texto_bronze"],
        }

    print("[+] Calculando flags de carta e coletânea...")

    qtd_assinaturas = []
    flags_carta = []

    qtd_titulos_similares = []
    titulos_similares_col = []
    flags_coletanea = []

    for _, row in df.iterrows():
        texto = row["_texto_bronze"]

        # Carta
        assinaturas = encontrar_assinaturas_machado(texto)

        qtd_assinaturas.append(len(assinaturas))

        titulo_norm = normalizar_texto(
            row["titulo"]
        )

        is_carta = (
            titulo_norm in TITULOS_CARTA_EXPLICITOS
            or (
                len(assinaturas) > args.min_assinaturas_carta
                and titulo_norm not in TITULOS_EXCLUIR_CARTA
            )
        )

        flags_carta.append(
            is_carta
        )

        # Coletânea
        resultado_coletanea = avaliar_coletanea(
            texto=texto,
            titulo_atual=row["titulo"],
            titulos_normalizados=(titulos_normalizados),
            titulos_info=(titulos_info),
            similaridade_threshold=(
                args.similaridade_threshold
            ),
            qtd_titulos_coletanea=(
                args.qtd_titulos_coletanea
            ),
            tamanho_fragmento=(
                args.tamanho_fragmento
            ),
            tamanho_minimo_fragmento=(
                args.tamanho_minimo_fragmento
            ),
        )

        qtd_titulos_similares.append(resultado_coletanea["qtd_titulos_similares"])

        titulos_similares_col.append(
            json.dumps(
                resultado_coletanea["titulos_similares"],
                ensure_ascii=False,
            )
        )

        flags_coletanea.append(resultado_coletanea["is_coletanea"])

    df["qtd_assinaturas_machado"] = qtd_assinaturas

    df["is_carta"] = flags_carta

    df["qtd_titulos_similares"] = qtd_titulos_similares

    df["titulos_similares"] = titulos_similares_col

    df["is_coletanea"] = flags_coletanea

    df["bronze_text_path"] = df["_pdf_path_resolvido"].apply(
        lambda path: caminho_relativo_projeto(
            caminho_txt_bronze(
                path,
                bronze_dir,
            )
        )
    )

    df.drop(
        columns=[
            "_pdf_path_resolvido",
            "_texto_bronze",
        ],
        inplace=True,
    )

    # Metadata bronze enxuto.
    colunas_metadata = [
        "titulo",
        "autor",
        "fonte",
        "acessos",
        "pdf_path",
        "bronze_text_path",
        "qtd_assinaturas_machado",
        "is_carta",
        "qtd_titulos_similares",
        "titulos_similares",
        "is_coletanea",
    ]

    colunas_metadata = [
        coluna
        for coluna in colunas_metadata
        if coluna in df.columns
    ]

    df = df[
        colunas_metadata
    ].copy()

    df.to_csv(
        output_metadata,
        index=False,
    )

    print()
    print("[OK] Camada bronze criada.")
    print(f"     Textos:   {bronze_dir}")
    print(f"     Metadata: {output_metadata}")
    print(f"     Cartas detectadas: " f"{int(df['is_carta'].sum())}")
    print(f"     Coletâneas detectadas: " f"{int(df['is_coletanea'].sum())}")


if __name__ == "__main__":
    main()
