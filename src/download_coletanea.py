import re
from pathlib import Path

import pandas as pd
import requests

CSV_PATH = Path(
    "/home/gustavoreis/Projetos/Homelab/ACartomanteRAG/"
    "Dataset/metadata/obras_machado_assis.csv"
)

# /ACartomanteRAG
PROJECT_DIR = CSV_PATH.parent.parent.parent

# /ACartomanteRAG/Dataset
DATASET_DIR = CSV_PATH.parent.parent

# /ACartomanteRAG/Dataset/raw
RAW_DIR = DATASET_DIR / "raw"

CSV_DEDUP_PATH = CSV_PATH.parent / "obras_machado_assis_deduplicadas.csv"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def normalizar_titulo(titulo: str) -> str:
    """
    Normalização usada apenas para deduplicação.
    """

    if pd.isna(titulo):
        return ""

    titulo = str(titulo).strip().lower()

    titulo = re.sub(
        r"\s+",
        " ",
        titulo,
    )

    return titulo


def nome_arquivo_seguro(titulo: str) -> str:
    """
    Converte o título da obra em um nome
    de arquivo seguro.

    Exemplo:

        A Carteira
            ->
        A_Carteira.pdf
    """

    titulo = str(titulo).strip()

    # Remove caracteres inadequados para filename
    titulo = re.sub(
        r'[<>:"/\\|?*]',
        "",
        titulo,
    )

    # Espaços -> _
    titulo = re.sub(
        r"\s+",
        "_",
        titulo,
    )

    # Evita nomes excessivamente grandes
    titulo = titulo[:180]

    return f"{titulo}.pdf"


def caminho_relativo_pdf(
    destino: Path,
) -> str:
    """
    Retorna o caminho relativo à raiz do projeto.

    Exemplo:

    /home/gustavoreis/.../ACartomanteRAG/Dataset/raw/A_Carteira.pdf

    vira:

    Dataset/raw/A_Carteira.pdf
    """

    return destino.relative_to(PROJECT_DIR).as_posix()


def selecionar_obras(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deduplica por título.

    Prioridade:

    1. Fonte contendo "Universidade"
    2. Maior número de acessos
    """

    df = df.copy()

    #
    # Normalização para deduplicação
    #
    df["titulo_normalizado"] = df["titulo"].apply(normalizar_titulo)

    #
    # Garante acessos numéricos
    #
    df["acessos"] = (
        pd.to_numeric(
            df["acessos"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    #
    # Indicador de preferência:
    #
    # True se fonte contém "Universidade".
    #
    df["fonte_universidade"] = (
        df["fonte"]
        .fillna("")
        .str.contains(
            "universidade",
            case=False,
            regex=False,
        )
    )

    #
    # Ordenação:
    #
    # título
    # universidade primeiro
    # maior acesso primeiro
    #
    df = df.sort_values(
        by=[
            "titulo_normalizado",
            "fonte_universidade",
            "acessos",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    #
    # Mantém somente o melhor registro
    # de cada título.
    #
    df = df.drop_duplicates(
        subset=["titulo_normalizado"],
        keep="first",
    )

    return df.reset_index(drop=True)


def baixar_pdf(
    session: requests.Session,
    url: str,
    destino: Path,
) -> bool:
    """
    Baixa um PDF caso ainda não exista.

    Retorna True quando:

    - o arquivo já existe;
    - ou foi baixado com sucesso.

    Retorna False quando houve erro.
    """

    if destino.exists():

        print(f"[SKIP] Já existe: " f"{destino.name}")

        return True

    try:

        response = session.get(
            url,
            timeout=60,
            stream=True,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        #
        # O servidor pode não retornar
        # application/pdf corretamente,
        # portanto é apenas um aviso.
        #
        if "pdf" not in content_type:

            print(
                "[AVISO] Content-Type "
                f"inesperado para "
                f"{destino.name}: "
                f"{content_type}"
            )

        with open(
            destino,
            "wb",
        ) as f:

            for chunk in response.iter_content(chunk_size=1024 * 1024):

                if chunk:
                    f.write(chunk)

        print(f"[OK] {destino.name}")

        return True

    except requests.RequestException as e:

        print(f"[ERRO] {destino.name}: {e}")

        #
        # Se começou a criar o arquivo
        # mas a requisição falhou,
        # removemos o arquivo incompleto.
        #
        if destino.exists():
            destino.unlink()

        return False


def main():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[+] Lendo CSV: " f"{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    total_original = len(df)

    print(f"[+] Registros originais: " f"{total_original}")

    #
    # Remove registros sem título.
    #
    df = df[df["titulo"].notna()].copy()

    #
    # ========================================================
    # DEDUPLICAÇÃO
    # ========================================================
    #

    df_dedup = selecionar_obras(df)

    print(f"[+] Obras após deduplicação: " f"{len(df_dedup)}")

    print(f"[+] Duplicatas removidas: " f"{len(df) - len(df_dedup)}")

    #
    # ========================================================
    # COLUNA DO PDF LOCAL
    # ========================================================
    #
    # Começa vazia.
    #
    # Só será preenchida quando o PDF:
    #
    # - já existir;
    # - ou for baixado com sucesso.
    #
    df_dedup["pdf_path"] = None

    #
    # ========================================================
    # DOWNLOAD
    # ========================================================
    #

    session = requests.Session()

    session.headers.update(HEADERS)

    total = len(df_dedup)

    baixados = 0
    erros = 0
    sem_url = 0

    for indice, row in df_dedup.iterrows():

        titulo = row["titulo"]

        url = row.get("url_download")

        print(f"\n[{indice + 1}/{total}] " f"{titulo}")

        #
        # Sem URL de download.
        #
        if pd.isna(url) or not str(url).strip():

            print("[AVISO] Sem URL de download.")

            sem_url += 1

            continue

        #
        # Filename a partir do título.
        #
        filename = nome_arquivo_seguro(titulo)

        destino = RAW_DIR / filename

        sucesso = baixar_pdf(
            session=session,
            url=str(url).strip(),
            destino=destino,
        )

        if sucesso:

            baixados += 1

            #
            # Exemplo:
            #
            # Dataset/raw/A_Carteira.pdf
            #
            df_dedup.at[
                indice,
                "pdf_path",
            ] = caminho_relativo_pdf(destino)

        else:

            erros += 1

    #
    # ========================================================
    # SALVA METADATA FINAL
    # ========================================================
    #
    # Fazemos isso somente DEPOIS dos downloads
    # para que pdf_path esteja preenchido.
    #

    colunas_auxiliares = [
        "titulo_normalizado",
        "fonte_universidade",
    ]

    df_saida = df_dedup.drop(
        columns=colunas_auxiliares,
        errors="ignore",
    )

    df_saida.to_csv(
        CSV_DEDUP_PATH,
        index=False,
    )

    print("\n[+] Metadata deduplicado " f"salvo em: {CSV_DEDUP_PATH}")

    #
    # ========================================================
    # RESULTADO
    # ========================================================
    #

    print("\n" + "=" * 80)

    print("RESULTADO")

    print("=" * 80)

    print(f"Registros originais:       " f"{total_original}")

    print(f"Obras únicas selecionadas: " f"{len(df_dedup)}")

    print(f"Duplicatas removidas:       " f"{len(df) - len(df_dedup)}")

    print(f"PDFs disponíveis/baixados:  " f"{baixados}")

    print(f"Sem URL de download:        " f"{sem_url}")

    print(f"Erros de download:          " f"{erros}")

    print(f"Diretório raw:              " f"{RAW_DIR}")

    print(f"Metadata final:             " f"{CSV_DEDUP_PATH}")

    print("=" * 80)


if __name__ == "__main__":
    main()
