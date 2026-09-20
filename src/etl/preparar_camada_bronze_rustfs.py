#!/usr/bin/env python3
"""
Prepara a camada Bronze do corpus de Machado de Assis a partir da Raw
persistida no RustFS.

Entrada:
    s3://<raw-bucket>/manifests/obras_raw.parquet
    s3://<raw-bucket>/documents/*.pdf

Saídas:
    s3://<bronze-bucket>/texts/*.txt
    s3://<bronze-bucket>/manifests/obras_bronze.parquet

A camada Raw não é alterada.

Na transição Raw -> Bronze:
    1. seleciona uma versão canônica por título;
    2. extrai e limpa o texto;
    3. classifica a obra;
    4. persiste texto e metadados enriquecidos.

Prioridade para seleção da versão canônica:
    1. fonte contendo "Universidade";
    2. maior número de acessos;
    3. menor document_id, como desempate determinístico.

Categorias:
    - Crítica
    - Carta
    - Coletânea
    - Teatro
    - Poema/Cântico
    - Romance/Conto
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import pymupdf
from dotenv import load_dotenv

from src.storage.rustfs_manager import RustFSManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / "config" / ".env"

DEFAULT_RAW_MANIFEST_KEY = "manifests/obras_raw.parquet"
DEFAULT_BRONZE_MANIFEST_KEY = "manifests/obras_bronze.parquet"
DEFAULT_BRONZE_TEXT_PREFIX = "texts"

DEFAULT_SIMILARIDADE_THRESHOLD = 0.1
DEFAULT_QTD_TITULOS_COLETANEA = 2
DEFAULT_TAMANHO_FRAGMENTO = 1000
DEFAULT_TAMANHO_MINIMO_FRAGMENTO = 100

DEFAULT_MIN_ASSINATURAS_CARTA = 5

DEFAULT_MAX_PALAVRAS_LINHA_POEMA = 10
DEFAULT_MIN_PCT_LINHAS_CURTAS_POEMA = 0.85

DEFAULT_MIN_REPETICOES_LOCUTOR = 3
DEFAULT_MIN_LOCUTORES_TEATRO = 2
DEFAULT_MIN_TURNOS_TEATRO = 10
DEFAULT_MIN_PCT_LOCUTORES_TEATRO = 0.91

CRITICAS_MANUAIS = {
    "textos críticos",
    "críticas teatrais",
}

PALAVRAS_ESTRUTURA_TEATRO = {
    "cena",
    "ato",
    "quadro",
    "prólogo",
    "prologo",
    "epílogo",
    "epilogo",
    "capítulo",
    "capitulo",
    "canto",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara a camada Bronze do corpus de Machado de Assis "
            "a partir da camada Raw no RustFS."
        )
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=(
            "Arquivo .env com as configurações do RustFS. "
            f"Padrão: {DEFAULT_ENV_PATH}"
        ),
    )

    parser.add_argument(
        "--raw-bucket",
        default=None,
        help="Sobrescreve RUSTFS_RAW_BUCKET definido no .env.",
    )

    parser.add_argument(
        "--bronze-bucket",
        default=None,
        help="Sobrescreve RUSTFS_BRONZE_BUCKET definido no .env.",
    )

    parser.add_argument(
        "--raw-manifest-key",
        default=DEFAULT_RAW_MANIFEST_KEY,
    )

    parser.add_argument(
        "--bronze-manifest-key",
        default=DEFAULT_BRONZE_MANIFEST_KEY,
    )

    parser.add_argument(
        "--bronze-text-prefix",
        default=DEFAULT_BRONZE_TEXT_PREFIX,
    )

    parser.add_argument(
        "--similaridade-threshold",
        type=float,
        default=DEFAULT_SIMILARIDADE_THRESHOLD,
    )

    parser.add_argument(
        "--qtd-titulos-coletanea",
        type=int,
        default=DEFAULT_QTD_TITULOS_COLETANEA,
    )

    parser.add_argument(
        "--tamanho-fragmento",
        type=int,
        default=DEFAULT_TAMANHO_FRAGMENTO,
    )

    parser.add_argument(
        "--tamanho-minimo-fragmento",
        type=int,
        default=DEFAULT_TAMANHO_MINIMO_FRAGMENTO,
    )

    parser.add_argument(
        "--min-assinaturas-carta",
        type=int,
        default=DEFAULT_MIN_ASSINATURAS_CARTA,
    )

    parser.add_argument(
        "--max-palavras-linha-poema",
        type=int,
        default=DEFAULT_MAX_PALAVRAS_LINHA_POEMA,
    )

    parser.add_argument(
        "--min-pct-linhas-curtas-poema",
        type=float,
        default=DEFAULT_MIN_PCT_LINHAS_CURTAS_POEMA,
    )

    parser.add_argument(
        "--min-pct-locutores-teatro",
        type=float,
        default=DEFAULT_MIN_PCT_LOCUTORES_TEATRO,
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
# HEURÍSTICA DE POEMAS / CÂNTICOS
# ============================================================


def avaliar_estrutura_poetica(
    texto: str,
    max_palavras_linha: int = DEFAULT_MAX_PALAVRAS_LINHA_POEMA,
    min_pct_linhas_curtas: float = DEFAULT_MIN_PCT_LINHAS_CURTAS_POEMA,
) -> dict:
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]

    if not linhas:
        return {
            "is_poema": False,
            "qtd_linhas": 0,
            "qtd_linhas_curtas": 0,
            "pct_linhas_curtas": 0.0,
        }

    qtd_linhas_curtas = sum(
        len(linha.split()) <= max_palavras_linha for linha in linhas
    )

    pct_linhas_curtas = qtd_linhas_curtas / len(linhas)

    return {
        "is_poema": pct_linhas_curtas >= min_pct_linhas_curtas,
        "qtd_linhas": len(linhas),
        "qtd_linhas_curtas": qtd_linhas_curtas,
        "pct_linhas_curtas": pct_linhas_curtas,
    }


# ============================================================
# HEURÍSTICA DE TEATRO
# ============================================================


def normalizar_locutor(linha: str) -> str:
    linha = linha.strip().casefold()
    linha = linha.strip(".,;:!?\"'()[]{}—–-")
    return re.sub(r"\s+", " ", linha)


def linha_parece_locutor(
    linha: str,
    max_palavras: int = 4,
    max_chars: int = 40,
) -> bool:
    linha = linha.strip()

    if not linha or len(linha) > max_chars:
        return False

    palavras = linha.split()

    if len(palavras) > max_palavras:
        return False

    primeira_palavra = palavras[0].strip(".,;:!?\"'()[]{}—–-").casefold()

    if primeira_palavra in PALAVRAS_ESTRUTURA_TEATRO:
        return False

    letras = [char for char in linha if char.isalpha()]

    if not letras:
        return False

    proporcao_maiusculas = sum(char.isupper() for char in letras) / len(letras)

    return proporcao_maiusculas >= 0.90


def extrair_locutor_linha(linha: str) -> str | None:
    linha = linha.strip()

    if not linha:
        return None

    if linha_parece_locutor(linha):
        return normalizar_locutor(linha)

    match = re.match(
        r"""
        ^
        (?P<locutor>
            [A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ]
            [A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ\s.'-]{0,39}?
        )
        \s*
        (?:
            :|
            \.|
            —|
            –|
            -
        )
        \s*
        (?P<fala>\S.+)
        $
        """,
        linha,
        flags=re.VERBOSE,
    )

    if not match:
        return None

    locutor = match.group("locutor").strip()

    if not linha_parece_locutor(locutor):
        return None

    return normalizar_locutor(locutor)


def avaliar_estrutura_teatral(
    texto: str,
    min_repeticoes_locutor: int = DEFAULT_MIN_REPETICOES_LOCUTOR,
    min_locutores: int = DEFAULT_MIN_LOCUTORES_TEATRO,
    min_turnos: int = DEFAULT_MIN_TURNOS_TEATRO,
    min_pct_locutores_recorrentes: float = DEFAULT_MIN_PCT_LOCUTORES_TEATRO,
) -> dict:
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]

    candidatos = []

    for indice, linha in enumerate(linhas):
        locutor = extrair_locutor_linha(linha)

        if locutor is None:
            continue

        candidatos.append(
            {
                "indice": indice,
                "linha": linha,
                "locutor": locutor,
            }
        )

    contagem = Counter(candidato["locutor"] for candidato in candidatos)

    locutores_recorrentes = {
        locutor for locutor, qtd in contagem.items() if qtd >= min_repeticoes_locutor
    }

    turnos_recorrentes = [
        candidato
        for candidato in candidatos
        if candidato["locutor"] in locutores_recorrentes
    ]

    qtd_turnos = len(turnos_recorrentes)

    pct_locutores_recorrentes = qtd_turnos / len(candidatos) if candidatos else 0.0

    is_teatro = (
        len(locutores_recorrentes) >= min_locutores
        and qtd_turnos >= min_turnos
        and pct_locutores_recorrentes >= min_pct_locutores_recorrentes
    )

    return {
        "is_teatro": is_teatro,
        "qtd_candidatos_locutor": len(candidatos),
        "qtd_locutores_recorrentes": len(locutores_recorrentes),
        "qtd_turnos": qtd_turnos,
        "pct_locutores_recorrentes": pct_locutores_recorrentes,
        "locutores": sorted(locutores_recorrentes),
        "contagem_locutores": {
            locutor: contagem[locutor] for locutor in sorted(locutores_recorrentes)
        },
    }


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


def avaliar_carta(
    texto: str,
    qtd_assinaturas_minima: int = DEFAULT_MIN_ASSINATURAS_CARTA,
) -> dict:
    assinaturas = encontrar_assinaturas_machado(texto)

    return {
        "is_carta": len(assinaturas) > qtd_assinaturas_minima,
        "qtd_assinaturas_machado": len(assinaturas),
        "assinaturas": assinaturas,
    }


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

    similares = [r for r in resultados if r["similaridade"] > similaridade_threshold]

    titulos_similares = [r["titulo"] for r in similares]

    return {
        "qtd_titulos_similares": len(titulos_similares),
        "titulos_similares": titulos_similares,
        "is_coletanea": (len(titulos_similares) >= qtd_titulos_coletanea),
    }


# ============================================================
# UTILITÁRIOS DE STORAGE / SELEÇÃO CANÔNICA
# ============================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalizar_titulo_deduplicacao(titulo: str) -> str:
    """
    Normalização conservadora usada somente para agrupar versões
    da mesma obra na transição Raw -> Bronze.
    """
    if pd.isna(titulo):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(titulo).strip().casefold(),
    )


def nome_arquivo_seguro(titulo: str) -> str:
    titulo = str(titulo).strip()
    titulo = re.sub(
        r'[<>:"/\\|?*]',
        "",
        titulo,
    )
    titulo = re.sub(
        r"\s+",
        "_",
        titulo,
    )
    titulo = titulo.strip("._")
    return titulo[:170] or "sem_titulo"


def carregar_parquet_rustfs(
    storage: RustFSManager,
    bucket: str,
    key: str,
) -> pd.DataFrame:
    if not storage.object_exists(
        bucket,
        key,
    ):
        raise FileNotFoundError(f"Objeto não encontrado: s3://{bucket}/{key}")

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / Path(key).name

        storage.download_file(
            bucket=bucket,
            key=key,
            local_path=local_path,
        )

        return pd.read_parquet(local_path)


def salvar_parquet_rustfs(
    storage: RustFSManager,
    bucket: str,
    key: str,
    df: pd.DataFrame,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / Path(key).name

        df.to_parquet(
            local_path,
            index=False,
        )

        storage.upload_file(
            bucket=bucket,
            key=key,
            local_path=local_path,
        )


def selecionar_versoes_canonicas(
    raw_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Seleciona uma versão representativa de cada título para a Bronze.

    A Raw continua preservando todas as versões.

    Prioridade:
        1. fonte contendo "Universidade";
        2. maior número de acessos;
        3. menor document_id.
    """
    obrigatorias = {
        "document_id",
        "titulo",
        "fonte",
        "raw_object_key",
    }

    faltantes = obrigatorias - set(raw_manifest.columns)

    if faltantes:
        raise ValueError(
            "Manifest Raw sem colunas obrigatórias: " + ", ".join(sorted(faltantes))
        )

    df = raw_manifest.copy()

    if "status_ingestao" in df.columns:
        df = df[df["status_ingestao"].isin(["uploaded", "existing"])].copy()

    df = df[df["titulo"].notna() & df["raw_object_key"].notna()].copy()

    df["titulo_normalizado"] = df["titulo"].apply(normalizar_titulo_deduplicacao)

    df["fonte_universidade"] = (
        df["fonte"]
        .fillna("")
        .str.contains(
            "universidade",
            case=False,
            regex=False,
        )
    )

    if "acessos" not in df.columns:
        df["acessos"] = 0

    df["acessos"] = (
        pd.to_numeric(
            df["acessos"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["_document_id_sort"] = pd.to_numeric(
        df["document_id"],
        errors="coerce",
    ).fillna(10**18)

    df = df.sort_values(
        by=[
            "titulo_normalizado",
            "fonte_universidade",
            "acessos",
            "_document_id_sort",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )

    df = df.drop_duplicates(
        subset=["titulo_normalizado"],
        keep="first",
    )

    return df.drop(
        columns=["_document_id_sort"],
        errors="ignore",
    ).reset_index(drop=True)


def bronze_object_key(
    document_id: str,
    titulo: str,
    prefix: str,
) -> str:
    filename = f"{str(document_id).zfill(6)}_" f"{nome_arquivo_seguro(titulo)}.txt"

    return f"{prefix.rstrip('/')}/{filename}"


# ============================================================
# PIPELINE
# ============================================================


def main() -> None:
    args = parse_args()

    if args.env_file.exists():
        load_dotenv(
            dotenv_path=args.env_file,
            override=False,
        )
        print(f"[+] Configuração carregada de: {args.env_file}")
    else:
        print(
            f"[AVISO] Arquivo .env não encontrado: {args.env_file}. "
            "Usando apenas variáveis já presentes no ambiente."
        )

    raw_bucket = args.raw_bucket or os.getenv(
        "RUSTFS_RAW_BUCKET",
        "acartomante-raw",
    )

    bronze_bucket = args.bronze_bucket or os.getenv(
        "RUSTFS_BRONZE_BUCKET",
        "acartomante-bronze",
    )

    storage = RustFSManager.from_env()

    if not storage.bucket_exists(raw_bucket):
        raise RuntimeError(f"Bucket Raw não encontrado: {raw_bucket}")

    storage.create_bucket(bronze_bucket)

    print(f"[+] Lendo manifest Raw: " f"s3://{raw_bucket}/{args.raw_manifest_key}")

    raw_manifest = carregar_parquet_rustfs(
        storage=storage,
        bucket=raw_bucket,
        key=args.raw_manifest_key,
    )

    total_raw = len(raw_manifest)

    df = selecionar_versoes_canonicas(raw_manifest)

    print(f"[+] Registros Raw: {total_raw}")
    print(f"[+] Obras canônicas selecionadas: {len(df)}")
    print(f"[+] Versões não selecionadas: " f"{total_raw - len(df)}")

    registros = []

    print(f"[+] Gerando textos Bronze para {len(df)} obras...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for indice, row in df.iterrows():
            titulo = str(row["titulo"])
            document_id = str(row["document_id"]).zfill(6)
            raw_object_key = str(row["raw_object_key"])

            bronze_key = bronze_object_key(
                document_id=document_id,
                titulo=titulo,
                prefix=args.bronze_text_prefix,
            )

            print(f"[{indice + 1}/{len(df)}] {titulo}")

            registro = row.to_dict()

            registro.update(
                {
                    "source_document_id": document_id,
                    "source_raw_bucket": raw_bucket,
                    "source_raw_object_key": raw_object_key,
                    "bronze_bucket": bronze_bucket,
                    "bronze_object_key": bronze_key,
                    "bronze_uri": (f"s3://{bronze_bucket}/{bronze_key}"),
                    "categoria": None,
                    "status_bronze": None,
                    "processed_at_utc": utc_now(),
                    "erro_bronze": None,
                    "_texto_bronze": None,
                }
            )

            local_pdf = tmp_dir / f"{document_id}.pdf"

            try:
                storage.download_file(
                    bucket=raw_bucket,
                    key=raw_object_key,
                    local_path=local_pdf,
                )

                texto = extrair_texto_bronze(
                    pdf_path=local_pdf,
                    fonte=row.get(
                        "fonte",
                        "",
                    ),
                )

                if not texto.strip():
                    raise ValueError("Texto Bronze vazio.")

                storage.upload_bytes(
                    bucket=bronze_bucket,
                    key=bronze_key,
                    data=texto.encode("utf-8"),
                    content_type=("text/plain; charset=utf-8"),
                )

                registro["_texto_bronze"] = texto
                registro["status_bronze"] = "success"

            except Exception as exc:
                registro["status_bronze"] = "erro"
                registro["erro_bronze"] = str(exc)

                print(f"    [ERRO] {exc}")

            finally:
                if local_pdf.exists():
                    local_pdf.unlink()

            registros.append(registro)

    df_bronze = pd.DataFrame(registros)

    df_sucesso = df_bronze[df_bronze["status_bronze"] == "success"].copy()

    if df_sucesso.empty:
        raise RuntimeError("Nenhum documento Bronze foi processado com sucesso.")

    titulos_normalizados = {
        titulo: normalizar_texto(titulo)
        for titulo in (df_sucesso["titulo"].dropna().unique())
    }

    titulos_info = {}

    for _, row in df_sucesso.iterrows():
        titulo_norm = normalizar_texto(row["titulo"])

        titulos_info[titulo_norm] = {
            "titulo": row["titulo"],
            "texto_bronze": row["_texto_bronze"],
        }

    print("[+] Classificando categorias das obras...")

    for indice, row in df_sucesso.iterrows():
        titulo = str(row["titulo"])
        texto = row["_texto_bronze"]

        is_critica = titulo.strip().casefold() in CRITICAS_MANUAIS

        resultado_carta = avaliar_carta(
            texto=texto,
            qtd_assinaturas_minima=(args.min_assinaturas_carta),
        )

        resultado_coletanea = avaliar_coletanea(
            texto=texto,
            titulo_atual=titulo,
            titulos_normalizados=titulos_normalizados,
            titulos_info=titulos_info,
            similaridade_threshold=(args.similaridade_threshold),
            qtd_titulos_coletanea=(args.qtd_titulos_coletanea),
            tamanho_fragmento=(args.tamanho_fragmento),
            tamanho_minimo_fragmento=(args.tamanho_minimo_fragmento),
        )

        resultado_teatro = avaliar_estrutura_teatral(
            texto=texto,
            min_pct_locutores_recorrentes=(args.min_pct_locutores_teatro),
        )

        resultado_poema = avaliar_estrutura_poetica(
            texto=texto,
            max_palavras_linha=(args.max_palavras_linha_poema),
            min_pct_linhas_curtas=(args.min_pct_linhas_curtas_poema),
        )

        if is_critica:
            categoria = "Crítica"
        elif resultado_carta["is_carta"]:
            categoria = "Carta"
        elif resultado_coletanea["is_coletanea"]:
            categoria = "Coletânea"
        elif resultado_teatro["is_teatro"]:
            categoria = "Teatro"
        elif resultado_poema["is_poema"]:
            categoria = "Poema/Cântico"
        else:
            categoria = "Romance/Conto"

        df_bronze.at[
            indice,
            "categoria",
        ] = categoria

    df_bronze.drop(
        columns=[
            "_texto_bronze",
            "titulo_normalizado",
            "fonte_universidade",
        ],
        errors="ignore",
        inplace=True,
    )

    colunas_preferidas = [
        "source_document_id",
        "titulo",
        "autor",
        "fonte",
        "formato",
        "acessos",
        "url_detalhe",
        "url_download",
        "source_raw_bucket",
        "source_raw_object_key",
        "sha256",
        "bronze_bucket",
        "bronze_object_key",
        "bronze_uri",
        "categoria",
        "status_bronze",
        "processed_at_utc",
        "erro_bronze",
    ]

    colunas_manifest = [
        coluna for coluna in colunas_preferidas if coluna in df_bronze.columns
    ]

    df_manifest = (
        df_bronze[colunas_manifest]
        .copy()
        .sort_values(
            "titulo",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    salvar_parquet_rustfs(
        storage=storage,
        bucket=bronze_bucket,
        key=args.bronze_manifest_key,
        df=df_manifest,
    )

    print()
    print("[OK] Camada Bronze criada.")
    print(f"     Textos: s3://{bronze_bucket}/" f"{args.bronze_text_prefix}/")
    print(f"     Manifest: s3://{bronze_bucket}/" f"{args.bronze_manifest_key}")

    sucessos = df_manifest["status_bronze"].eq("success").sum()

    erros = df_manifest["status_bronze"].eq("erro").sum()

    print()
    print(f"[+] Sucessos: {sucessos}")
    print(f"[+] Erros:    {erros}")

    distribuicao = df_manifest["categoria"].dropna().value_counts()

    if not distribuicao.empty:
        print()
        print("[+] Distribuição por categoria:")

        total_classificado = int(distribuicao.sum())

        for categoria, quantidade in distribuicao.items():
            percentual = quantidade / total_classificado * 100

            print(f"     {categoria}: " f"{quantidade} " f"({percentual:.2f}%)")


if __name__ == "__main__":
    main()
