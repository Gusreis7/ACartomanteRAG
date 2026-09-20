import argparse
import hashlib
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from src.storage.rustfs_manager import RustFSManager

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

MANIFEST_KEY = "manifests/obras_raw.parquet"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / "config" / ".env"

session = requests.Session()
session.headers.update(HEADERS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def limpar_texto(texto: str) -> str:
    return re.sub(r"\s+", " ", texto or "").strip()


def parse_acessos(valor: str):
    valor = limpar_texto(valor)
    numeros = re.sub(r"\D", "", valor)

    if not numeros:
        return None

    return int(numeros)


def nome_arquivo_seguro(titulo: str) -> str:
    titulo = limpar_texto(titulo)
    titulo = re.sub(r'[<>:"/\\|?*]', "", titulo)
    titulo = re.sub(r"\s+", "_", titulo)
    titulo = titulo.strip("._")
    titulo = titulo[:170]

    return titulo or "sem_titulo"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def arquivo_parece_pdf(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(5) == b"%PDF-"


def get_download_links(url_detalhe: str) -> list[str]:
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
        texto_link = limpar_texto(a.get_text(" ", strip=True)).lower()

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
):
    tds = tr.find_all(
        "td",
        recursive=False,
    )

    if not tds:
        return None

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

    if link_detalhe is None:
        return None

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
        return None

    if len(tds) <= indice_titulo + 5:
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

    url_detalhe = urljoin(
        url_pagina,
        link_detalhe["href"],
    )

    try:
        downloads = get_download_links(url_detalhe)

    except requests.RequestException as exc:
        print(f"[AVISO] Falha buscando download " f"de '{titulo}': {exc}")
        downloads = []

    return {
        "titulo": titulo,
        "autor": autor,
        "fonte": fonte,
        "formato": formato,
        "tamanho_origem": tamanho,
        "acessos": parse_acessos(acessos_texto),
        "url_detalhe": url_detalhe,
        "url_download": (downloads[0] if downloads else None),
        "downloads_encontrados": len(downloads),
    }


def get_obras_machado(
    pagina: int,
) -> list[dict]:
    skip = (pagina - 1) * 50

    url = LIST_URL.format(
        pagina=pagina,
        skip=skip,
    )

    print(f"[+] Consultando página {pagina} " f"(skip={skip})")

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

    for tr in soup.find_all("tr"):
        obra = parse_linha_obra(
            tr=tr,
            url_pagina=url,
        )

        if obra is None:
            continue

        obra["pagina_origem"] = pagina
        obras.append(obra)

    return obras


def carregar_manifest_existente(
    storage: RustFSManager,
    bucket: str,
) -> pd.DataFrame:
    if not storage.object_exists(
        bucket,
        MANIFEST_KEY,
    ):
        return pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "obras_raw.parquet"

        storage.download_file(
            bucket=bucket,
            key=MANIFEST_KEY,
            local_path=local_path,
        )

        return pd.read_parquet(local_path)


def proximo_document_id(
    manifest: pd.DataFrame,
) -> int:
    if manifest.empty or "document_id" not in manifest.columns:
        return 1

    ids = pd.to_numeric(
        manifest["document_id"],
        errors="coerce",
    ).dropna()

    if ids.empty:
        return 1

    return int(ids.max()) + 1


def baixar_pdf_temporario(
    url: str,
    destino: Path,
) -> tuple[int, str]:
    with session.get(
        url,
        timeout=60,
        stream=True,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()

        content_type = (
            response.headers.get(
                "Content-Type",
                "",
            )
            .lower()
            .strip()
        )

        with destino.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    if not arquivo_parece_pdf(destino):
        raise ValueError("Conteúdo baixado não possui " "assinatura de arquivo PDF.")

    return (
        destino.stat().st_size,
        content_type,
    )


def salvar_manifest(
    storage: RustFSManager,
    bucket: str,
    registros: list[dict],
) -> None:
    df = pd.DataFrame(registros)

    if not df.empty:
        df = df.sort_values("document_id").reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "obras_raw.parquet"

        df.to_parquet(
            local_path,
            index=False,
        )

        storage.upload_file(
            bucket=bucket,
            key=MANIFEST_KEY,
            local_path=local_path,
        )

    print(f"[OK] Manifest: " f"s3://{bucket}/{MANIFEST_KEY}")


def processar_obra(
    storage: RustFSManager,
    bucket: str,
    obra: dict,
    document_id: str,
    registro_anterior: dict | None,
) -> dict:
    titulo = obra["titulo"]

    filename = f"{document_id}_" f"{nome_arquivo_seguro(titulo)}.pdf"

    object_key = registro_anterior.get("raw_object_key") if registro_anterior else None

    if not object_key:
        object_key = f"documents/{filename}"

    registro = {
        **obra,
        "document_id": document_id,
        "raw_bucket": bucket,
        "raw_object_key": object_key,
        "raw_uri": (f"s3://{bucket}/{object_key}"),
        "sha256": None,
        "size_bytes": None,
        "content_type": None,
        "status_ingestao": None,
        "ingested_at_utc": None,
        "crawled_at_utc": utc_now(),
        "erro_ingestao": None,
    }

    if registro_anterior:
        for coluna in [
            "sha256",
            "size_bytes",
            "content_type",
            "ingested_at_utc",
        ]:
            valor = registro_anterior.get(coluna)

            if pd.notna(valor):
                registro[coluna] = valor

    if storage.object_exists(
        bucket,
        object_key,
    ):
        registro["status_ingestao"] = "existing"

        print(f"[SKIP] {object_key}")
        return registro

    url_download = obra.get("url_download")

    if not url_download:
        registro["status_ingestao"] = "sem_download"

        print(f"[AVISO] Sem download: " f"{titulo}")
        return registro

    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_path = Path(tmp) / filename

            size_bytes, content_type = baixar_pdf_temporario(
                url=url_download,
                destino=local_path,
            )

            checksum = sha256_file(local_path)

            storage.upload_file(
                bucket=bucket,
                key=object_key,
                local_path=local_path,
            )

        registro.update(
            {
                "sha256": checksum,
                "size_bytes": size_bytes,
                "content_type": (content_type),
                "status_ingestao": ("uploaded"),
                "ingested_at_utc": (utc_now()),
            }
        )

        print(f"[OK] {object_key}")

    except Exception as exc:
        registro["status_ingestao"] = "erro"

        registro["erro_ingestao"] = str(exc)

        print(f"[ERRO] {titulo}: {exc}")

    return registro


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pagina-inicial",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--pagina-final",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=("Arquivo .env com as configurações do RustFS. " "Padrão: config/.env"),
    )

    parser.add_argument(
        "--bucket",
        default=None,
        help=("Sobrescreve RUSTFS_RAW_BUCKET definido no .env."),
    )

    args = parser.parse_args()

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

    bucket = args.bucket or os.getenv(
        "RUSTFS_RAW_BUCKET",
        "acartomante-raw",
    )

    storage = RustFSManager.from_env()
    storage.create_bucket(bucket)

    manifest_anterior = carregar_manifest_existente(
        storage=storage,
        bucket=bucket,
    )

    anteriores_por_url = {}

    if not manifest_anterior.empty and "url_detalhe" in manifest_anterior.columns:
        anteriores_por_url = {
            row["url_detalhe"]: row.to_dict()
            for _, row in manifest_anterior.iterrows()
            if pd.notna(row.get("url_detalhe"))
        }

    next_id = proximo_document_id(manifest_anterior)

    #
    # Esta deduplicação é apenas técnica:
    # evita processar a mesma URL de detalhe
    # mais de uma vez caso o site repita o
    # mesmo registro entre páginas.
    #
    # NÃO deduplicamos por título.
    #
    urls_vistas = set()
    obras = []

    for pagina in range(
        args.pagina_inicial,
        args.pagina_final + 1,
    ):
        try:
            obras_pagina = get_obras_machado(pagina)

            print(f"[+] Página {pagina}: " f"{len(obras_pagina)} registros")

            for obra in obras_pagina:
                url_detalhe = obra["url_detalhe"]

                if url_detalhe in urls_vistas:
                    continue

                urls_vistas.add(url_detalhe)
                obras.append(obra)

        except requests.RequestException as exc:
            print(f"[ERRO HTTP] " f"Página {pagina}: {exc}")

        except Exception as exc:
            print(f"[ERRO] " f"Página {pagina}: {exc}")

        time.sleep(args.delay)

    registros = []

    for indice, obra in enumerate(
        obras,
        start=1,
    ):
        print(f"\n[{indice}/{len(obras)}] " f"{obra['titulo']}")

        anterior = anteriores_por_url.get(obra["url_detalhe"])

        if anterior:
            document_id = str(anterior["document_id"]).zfill(6)

        else:
            document_id = f"{next_id:06d}"
            next_id += 1

        registro = processar_obra(
            storage=storage,
            bucket=bucket,
            obra=obra,
            document_id=document_id,
            registro_anterior=anterior,
        )

        registros.append(registro)

    #
    # Preserva no manifest também entradas
    # antigas que eventualmente não tenham
    # aparecido no crawl atual.
    #
    urls_atuais = {item["url_detalhe"] for item in registros}

    if not manifest_anterior.empty:
        for _, row in manifest_anterior.iterrows():
            url = row.get("url_detalhe")

            if pd.notna(url) and url not in urls_atuais:
                antigo = row.to_dict()
                antigo["status_crawl"] = "nao_encontrado_neste_crawl"
                registros.append(antigo)

    salvar_manifest(
        storage=storage,
        bucket=bucket,
        registros=registros,
    )

    df_resultado = pd.DataFrame(registros)

    print("\n" + "=" * 80)
    print("RESULTADO")
    print("=" * 80)
    print(f"Registros encontrados: " f"{len(obras)}")

    if not df_resultado.empty and "status_ingestao" in df_resultado.columns:
        print(df_resultado["status_ingestao"].value_counts(dropna=False).to_string())

    print(f"Manifest: " f"s3://{bucket}/" f"{MANIFEST_KEY}")
    print("=" * 80)


if __name__ == "__main__":
    main()
