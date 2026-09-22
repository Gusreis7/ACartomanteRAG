from __future__ import annotations

import re
from statistics import median

from transformers import AutoTokenizer


PADRAO_CAPITULO = re.compile(
    r"""
    ^[ \t]*
    cap[ií]tulo
    [ \t]+
    (?P<numero>
        \d{1,3}
        |
        [ivxlcdm]+
        |
        primeiro
    )
    (?:
        [ \t]*
        [/\-–—.:]
        [ \t]*
        (?P<titulo_inline>[^\n]+)
    )?
    [ \t]*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

PADRAO_CAPITULO_NA_ARCA = re.compile(
    r"""
    ^[ \t]*
    cap[ií]tulo
    [ \t]+
    (?P<numero>[a-z])
    [ \t]*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

PADRAO_ROMANO_ISOLADO = re.compile(
    r"""
    ^[ \t]*
    (?P<numero>[ivxlcdm]+)
    [ \t]*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


# Falsos positivos de títulos confirmados manualmente no notebook.
# Mantemos explícitos em vez de aumentar a heurística geral e
# prejudicar títulos válidos.
TITULOS_EDGE_CASE_BLOQUEADOS = {
    "crisálidas": {
        "Há dez anos!... sim... dez anos!...",
        "A que escola pertence o autor deste livro?",
        "Mon pauvre coeur, reprends ton sublime courage",
        "Ne vois-tu pas ?",
    },
    "história de quinze dias": {
        "A. S. EX.ª REVMA. SR. BISPO CAPELÃO-MOR",
        "O CARNAVAL morreu, viva a quaresma!",
    },
    "notas semanais": {
        "Anuncia-se um congresso agrícola, um congresso oficial, presidido pelo",
    },
    "quincas borba": {
        "Rubião perguntou-lhe uma vez",
    },
}


class HuggingFaceChunkTokenizer:
    """
    Tokenizer usado somente para definir as janelas da Silver.

    O texto dos chunks é recortado pelos offsets do tokenizer Fast,
    preservando exatamente o texto da Bronze.
    """

    def __init__(
        self,
        model: str,
        revision: str = "main",
    ) -> None:
        self.model = model
        self.revision = revision

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            use_fast=True,
        )

        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"O tokenizer '{model}' precisa possuir implementação "
                "Fast para fornecer offset_mapping."
            )

        self.resolved_revision = (
            self.tokenizer.init_kwargs.get("_commit_hash")
            or revision
        )

    def split(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[dict]:
        if overlap >= chunk_size:
            raise ValueError(
                "overlap deve ser menor que chunk_size."
            )

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )

        offsets = encoded["offset_mapping"]

        if not offsets:
            return []

        step = chunk_size - overlap
        chunks = []

        for token_start in range(
            0,
            len(offsets),
            step,
        ):
            token_end = min(
                token_start + chunk_size,
                len(offsets),
            )

            valid_offsets = [
                (start, end)
                for start, end
                in offsets[token_start:token_end]
                if end > start
            ]

            if not valid_offsets:
                continue

            char_start = valid_offsets[0][0]
            char_end = valid_offsets[-1][1]

            chunk_text = text[
                char_start:char_end
            ]

            if chunk_text.strip():
                chunks.append(
                    {
                        "text": chunk_text,
                        "token_count": (
                            token_end - token_start
                        ),
                        "token_start": token_start,
                        "token_end": token_end,
                        "char_start": char_start,
                        "char_end": char_end,
                    }
                )

            if token_end == len(offsets):
                break

        return chunks


def linha_e_numero_pagina(
    linha: str,
) -> bool:
    return bool(
        re.fullmatch(
            r"\s*\d{1,4}\s*",
            linha,
        )
    )


def encontrar_titulo_apos_capitulo(
    linhas: list[str],
    indice_capitulo: int,
    titulo_obra: str | None = None,
    max_palavras_titulo: int = 10,
) -> dict:
    """
    Heurística V4 consolidada no notebook para títulos na linha
    seguinte ao marcador de capítulo.
    """
    titulo_obra_norm = (
        str(titulo_obra)
        .strip()
        .casefold()
        if titulo_obra
        else ""
    )

    for i in range(
        indice_capitulo + 1,
        min(
            indice_capitulo + 6,
            len(linhas),
        ),
    ):
        linha_original = linhas[i]
        linha = linha_original.strip()

        if not linha:
            continue

        if linha_e_numero_pagina(
            linha
        ):
            continue

        titulos_bloqueados = (
            TITULOS_EDGE_CASE_BLOQUEADOS.get(
                titulo_obra_norm,
                set(),
            )
        )

        if linha in titulos_bloqueados:
            return {
                "titulo": None,
                "regra_titulo": None,
            }

        # A linha textual precisa estar completa no arquivo.
        if not linha_original.endswith("\n"):
            return {
                "titulo": None,
                "regra_titulo": None,
            }

        if (
            len(linha.split())
            > max_palavras_titulo
        ):
            return {
                "titulo": None,
                "regra_titulo": None,
            }

        # Fala/prosa.
        if linha.startswith(
            ("-", "–", "—")
        ):
            return {
                "titulo": None,
                "regra_titulo": None,
            }

        # Padrões que se mostraram continuação de prosa.
        if linha.endswith(
            (",", ":")
        ):
            return {
                "titulo": None,
                "regra_titulo": None,
            }

        # Ponto simples normalmente é prosa.
        # Mantém títulos em caixa alta, como "O FINAL DO § 4º.".
        termina_ponto_simples = (
            linha.endswith(".")
            and not linha.endswith("...")
        )

        if termina_ponto_simples:
            letras = [
                char
                for char in linha
                if char.isalpha()
            ]

            proporcao_maiusculas = (
                sum(
                    char.isupper()
                    for char in letras
                )
                / len(letras)
                if letras
                else 0
            )

            if (
                proporcao_maiusculas
                < 0.80
            ):
                return {
                    "titulo": None,
                    "regra_titulo": None,
                }

        # Se a próxima linha textual começa em minúscula,
        # o candidato atual provavelmente é uma quebra de prosa.
        proxima_linha = None

        for j in range(
            i + 1,
            min(
                i + 6,
                len(linhas),
            ),
        ):
            candidata = (
                linhas[j].strip()
            )

            if not candidata:
                continue

            if linha_e_numero_pagina(
                candidata
            ):
                continue

            proxima_linha = candidata
            break

        if proxima_linha:
            primeiro_char = (
                proxima_linha[0]
            )

            if (
                primeiro_char.isalpha()
                and primeiro_char.islower()
            ):
                return {
                    "titulo": None,
                    "regra_titulo": None,
                }

        return {
            "titulo": linha,
            "regra_titulo": (
                "proxima_linha"
            ),
        }

    return {
        "titulo": None,
        "regra_titulo": None,
    }


def extrair_capitulos_com_padrao(
    text: str,
    pattern: re.Pattern,
    regra_capitulo: str,
    titulo_obra: str,
    buscar_titulo: bool = True,
) -> list[dict]:
    linhas = text.splitlines(
        keepends=True
    )

    capitulos = []
    posicao = 0

    for indice_linha, linha_original in enumerate(
        linhas
    ):
        linha = linha_original.strip()
        match = pattern.match(
            linha
        )

        if not match:
            posicao += len(
                linha_original
            )
            continue

        titulo_inline = (
            match.groupdict().get(
                "titulo_inline"
            )
        )

        titulo = None
        regra_titulo = None

        if buscar_titulo:
            if titulo_inline:
                titulo = (
                    titulo_inline.strip()
                )
                regra_titulo = "inline"
            else:
                resultado_titulo = (
                    encontrar_titulo_apos_capitulo(
                        linhas=linhas,
                        indice_capitulo=(
                            indice_linha
                        ),
                        titulo_obra=(
                            titulo_obra
                        ),
                    )
                )

                titulo = (
                    resultado_titulo[
                        "titulo"
                    ]
                )
                regra_titulo = (
                    resultado_titulo[
                        "regra_titulo"
                    ]
                )

        capitulos.append(
            {
                "numero": match.group(
                    "numero"
                ),
                "cabecalho": linha,
                "titulo": titulo,
                "regra_titulo": (
                    regra_titulo
                ),
                "regra_capitulo": (
                    regra_capitulo
                ),
                "linha": indice_linha,
                "inicio": posicao,
            }
        )

        posicao += len(
            linha_original
        )

    return capitulos


def validar_estrutura_capitulos(
    capitulos: list[dict],
    min_capitulos: int,
    min_mediana_chars: int,
) -> tuple[bool, float]:
    if (
        len(capitulos)
        < min_capitulos
    ):
        return False, 0.0

    distancias = [
        atual["inicio"]
        - anterior["inicio"]
        for anterior, atual
        in zip(
            capitulos,
            capitulos[1:],
        )
    ]

    mediana_distancia = (
        median(distancias)
        if distancias
        else 0.0
    )

    valido = (
        len(capitulos)
        >= min_capitulos
        and mediana_distancia
        >= min_mediana_chars
    )

    return (
        valido,
        float(
            mediana_distancia
        ),
    )


def detectar_capitulos(
    text: str,
    title: str,
    min_capitulos: int,
    min_mediana_chars: int,
) -> list[dict]:
    """
    V4 consolidada no notebook:

    1. CAPÍTULO + número/romano/PRIMEIRO;
    2. Na Arca: CAPÍTULO A/B/C;
    3. fallback para numerais romanos isolados.

    A detecção é aplicada independentemente da categoria Bronze.
    """
    titulo_norm = (
        str(title)
        .strip()
        .casefold()
    )

    capitulos = (
        extrair_capitulos_com_padrao(
            text=text,
            pattern=PADRAO_CAPITULO,
            regra_capitulo=(
                "capitulo_explicito"
            ),
            titulo_obra=titulo_norm,
            buscar_titulo=True,
        )
    )

    valido, _ = (
        validar_estrutura_capitulos(
            capitulos=capitulos,
            min_capitulos=min_capitulos,
            min_mediana_chars=(
                min_mediana_chars
            ),
        )
    )

    # Edge case confirmado manualmente:
    # "Na Arca" usa CAPÍTULO A/B/C.
    if (
        not valido
        and titulo_norm
        in {
            "na arca",
            "na arca.txt",
        }
    ):
        capitulos = (
            extrair_capitulos_com_padrao(
                text=text,
                pattern=(
                    PADRAO_CAPITULO_NA_ARCA
                ),
                regra_capitulo=(
                    "capitulo_letra"
                ),
                titulo_obra=titulo_norm,
                buscar_titulo=True,
            )
        )

        valido, _ = (
            validar_estrutura_capitulos(
                capitulos=capitulos,
                min_capitulos=(
                    min_capitulos
                ),
                min_mediana_chars=(
                    min_mediana_chars
                ),
            )
        )

    if not valido:
        capitulos_romanos = (
            extrair_capitulos_com_padrao(
                text=text,
                pattern=(
                    PADRAO_ROMANO_ISOLADO
                ),
                regra_capitulo=(
                    "romano_isolado"
                ),
                titulo_obra=titulo_norm,
                buscar_titulo=False,
            )
        )

        valido_romano, _ = (
            validar_estrutura_capitulos(
                capitulos=(
                    capitulos_romanos
                ),
                min_capitulos=(
                    min_capitulos
                ),
                min_mediana_chars=(
                    min_mediana_chars
                ),
            )
        )

        if valido_romano:
            capitulos = (
                capitulos_romanos
            )
            valido = True

    return (
        capitulos
        if valido
        else []
    )


def build_sections(
    text: str,
    title: str,
    strategy: str,
    min_capitulos: int,
    min_mediana_chars: int,
) -> tuple[list[dict], str]:
    """
    fixed:
        documento inteiro.

    hybrid:
        usa qualquer divisão capitular/semântica validada pelas
        heurísticas, independentemente da categoria da obra.
        Se nada for validado, cai para fixed.
    """
    document_section = [
        {
            "section_index": 0,
            "section_type": "document",
            "section_number": None,
            "section_header": None,
            "section_title": None,
            "section_title_rule": None,
            "chapter_rule": None,
            "start_char": 0,
            "end_char": len(text),
            "text": text,
        }
    ]

    if strategy == "fixed":
        return (
            document_section,
            "fixed",
        )

    capitulos = detectar_capitulos(
        text=text,
        title=title,
        min_capitulos=min_capitulos,
        min_mediana_chars=(
            min_mediana_chars
        ),
    )

    if not capitulos:
        return (
            document_section,
            "fixed_fallback",
        )

    sections = []
    section_index = 0

    first_start = (
        capitulos[0]["inicio"]
    )

    if text[:first_start].strip():
        sections.append(
            {
                "section_index": (
                    section_index
                ),
                "section_type": (
                    "preamble"
                ),
                "section_number": None,
                "section_header": None,
                "section_title": None,
                "section_title_rule": None,
                "chapter_rule": None,
                "start_char": 0,
                "end_char": first_start,
                "text": (
                    text[:first_start]
                ),
            }
        )

        section_index += 1

    for index, capitulo in enumerate(
        capitulos
    ):
        start = capitulo["inicio"]

        if (
            index + 1
            < len(capitulos)
        ):
            end = capitulos[
                index + 1
            ]["inicio"]
        else:
            end = len(text)

        sections.append(
            {
                "section_index": (
                    section_index
                ),
                "section_type": (
                    "chapter"
                ),
                "section_number": (
                    capitulo["numero"]
                ),
                "section_header": (
                    capitulo["cabecalho"]
                ),
                "section_title": (
                    capitulo["titulo"]
                ),
                "section_title_rule": (
                    capitulo[
                        "regra_titulo"
                    ]
                ),
                "chapter_rule": (
                    capitulo[
                        "regra_capitulo"
                    ]
                ),
                "start_char": start,
                "end_char": end,
                "text": text[start:end],
            }
        )

        section_index += 1

    return sections, "hybrid"
