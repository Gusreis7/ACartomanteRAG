import argparse
import hashlib
import json
import math
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.etl.raw_dataset import (
    crawl_lock,
    load_raw_dataset,
    pdf_metadata,
    raw_prefix,
    validate_raw_manifest,
)
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
    fetch_downloads: bool = True,
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
        downloads = get_download_links(url_detalhe) if fetch_downloads else []

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
    fetch_downloads: bool = True,
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
            fetch_downloads=fetch_downloads,
        )

        if obra is None:
            continue

        obra["pagina_origem"] = pagina
        obras.append(obra)

    return obras


def carregar_manifest_existente(
    storage: RustFSManager,
    bucket: str,
    manifest_key: str = MANIFEST_KEY,
) -> pd.DataFrame:
    if not storage.object_exists(
        bucket,
        manifest_key,
    ):
        return pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "obras_raw.parquet"

        storage.download_file(
            bucket=bucket,
            key=manifest_key,
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
    manifest_key: str = MANIFEST_KEY,
    quiet: bool = False,
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
            key=manifest_key,
            local_path=local_path,
        )

    if not quiet:
        print(f"[OK] Manifest: s3://{bucket}/{manifest_key}")


def processar_obra(
    storage: RustFSManager,
    bucket: str,
    obra: dict,
    document_id: str,
    registro_anterior: dict | None,
    object_prefix: str | None = None,
) -> dict:
    titulo = obra["titulo"]

    filename = f"{document_id}_" f"{nome_arquivo_seguro(titulo)}.pdf"

    object_key = registro_anterior.get("raw_object_key") if registro_anterior else None

    if not object_key:
        object_key = (
            f"{object_prefix}/{hashlib.sha256(obra['url_detalhe'].encode()).hexdigest()}.pdf"
            if object_prefix
            else f"documents/{filename}"
        )

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
        "attempt_count": attempt_count(registro_anterior) if registro_anterior else 0,
        "last_attempt_at_utc": (
            registro_anterior.get("last_attempt_at_utc") if registro_anterior else None
        ),
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
        # Uma interrupção pode ocorrer entre o upload e o checkpoint.
        if object_prefix and not registro["sha256"]:
            checksum, size = pdf_metadata(storage, bucket, object_key)
            registro.update(
                sha256=checksum,
                size_bytes=size,
                content_type="application/pdf",
                ingested_at_utc=utc_now(),
            )
        registro["status_ingestao"] = "existing"

        print(f"[SKIP] {object_key}")
        return registro

    url_download = obra.get("url_download")
    registro["attempt_count"] += 1
    registro["last_attempt_at_utc"] = utc_now()

    if not isinstance(url_download, str) or not url_download.strip():
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
        "--strict",
        action="store_true",
        help="Produção: falha se alguma página ou download tiver erro.",
    )
    parser.add_argument(
        "--raw-bucket",
        help="Bucket AWS existente; coleta somente Raw, usando o perfil AWS do notebook.",
    )
    parser.add_argument(
        "--raw-id",
        help="Snapshot único, por exemplo machado-v1; obrigatório com --raw-bucket.",
    )
    parser.add_argument("--region", default="us-east-1")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--retry",
        action="store_true",
        help="Retenta apenas erro/sem_download do manifest, preservando os PDFs concluídos.",
    )
    actions.add_argument(
        "--mark-unavailable",
        nargs="+",
        metavar="ID",
        help="Registra indisponibilidade confirmada para IDs pendentes, sem baixar arquivos.",
    )
    parser.add_argument("--reason", help="Motivo obrigatório para --mark-unavailable.")

    parser.add_argument(
        "--pagina-inicial",
        type=int,
        default=1,
        help="Página inicial do crawler. Padrão: 1.",
    )

    parser.add_argument(
        "--pagina-final",
        type=int,
        default=7,
        help="Página final do crawler. Padrão: 7.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2,
        help="Delay entre páginas e, no modo S3 Raw, entre documentos. Padrão: 2.",
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Sobrescreve o .env padrão do ConfigManager. "
            "Se omitido, usa config/.env."
        ),
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "Sobrescreve o YAML padrão do ConfigManager. "
            "Se omitido, usa config/config.yaml."
        ),
    )

    args = parser.parse_args()
    if (
        args.pagina_inicial < 1
        or args.pagina_final < args.pagina_inicial
        or not math.isfinite(args.delay)
        or args.delay < 0
    ):
        parser.error("Intervalo de páginas ou delay inválido.")
    if bool(args.raw_bucket) != bool(args.raw_id):
        parser.error("Informe --raw-bucket e --raw-id juntos.")
    if (args.retry or args.mark_unavailable) and not args.raw_id:
        parser.error("Retentativa e confirmação exigem --raw-bucket e --raw-id.")
    if bool(args.mark_unavailable) != bool(args.reason and args.reason.strip()):
        parser.error("Use --reason com --mark-unavailable, preenchendo o motivo.")
    if args.raw_id:
        prefix = raw_prefix(args.raw_id)
        if args.env_file or args.config_file:
            parser.error(
                "O modo S3 Raw usa o perfil AWS; não aceita configurações de serviços locais."
            )
        storage = RustFSManager(region=args.region)
        if not storage.bucket_exists(args.raw_bucket):
            raise ValueError(
                "Bucket Raw inexistente; use o bucket já criado pelo Terraform."
            )
        with crawl_lock(storage, args.raw_bucket, args.raw_id):
            request = {
                "raw_id": args.raw_id,
                "source_query": LIST_URL,
                "first_page": args.pagina_inicial,
                "last_page": args.pagina_final,
            }
            request_key = prefix + "/request.json"
            if (args.retry or args.mark_unavailable) and (
                not storage.object_exists(args.raw_bucket, request_key)
                or not storage.object_exists(
                    args.raw_bucket, prefix + "/manifest.parquet"
                )
            ):
                parser.error(
                    "Este raw-id não tem consulta e manifest para retomar. Execute primeiro a coleta normal."
                )
            if storage.object_exists(args.raw_bucket, request_key):
                if (
                    json.loads(storage.download_bytes(args.raw_bucket, request_key))
                    != request
                ):
                    raise ValueError(
                        "Este raw-id já tem outra consulta ou intervalo. Use um novo raw-id."
                    )
            else:
                storage.client.put_object(
                    Bucket=args.raw_bucket,
                    Key=request_key,
                    Body=json.dumps(request).encode(),
                    ContentType="application/json",
                    IfNoneMatch="*",
                )
            if storage.object_exists(args.raw_bucket, prefix + "/ready.json"):
                if args.mark_unavailable:
                    parser.error("Raw já concluído; alterações exigem um novo raw-id.")
                receipt, _ = load_raw_dataset(storage, args.raw_bucket, args.raw_id)
                if (receipt["first_page"], receipt["last_page"]) != (
                    args.pagina_inicial,
                    args.pagina_final,
                ):
                    raise ValueError(
                        "Este raw-id já tem outro intervalo. Use um novo raw-id."
                    )
                print(
                    f"[ok] Raw já concluído: {args.raw_id}; {receipt['documents']} PDFs. Nenhuma coleta repetida."
                )
                return
            if args.retry or args.mark_unavailable:
                try:
                    code = resume_manifest(args, storage, args.raw_bucket, prefix)
                except ValueError as exc:
                    parser.error(str(exc))
            else:
                code = collect(
                    args,
                    storage,
                    args.raw_bucket,
                    prefix + "/manifest.parquet",
                    prefix + "/documents",
                )
            if code:
                raise SystemExit(code)
        return

    from src.config.config_manager import ConfigManager

    config_kwargs = {}

    if args.env_file is not None:
        config_kwargs["env_file"] = args.env_file

    if args.config_file is not None:
        config_kwargs["yaml_file"] = args.config_file

    config = ConfigManager.load(**config_kwargs)

    bucket = config.storage.raw_bucket

    storage = RustFSManager.from_config(config.environment.rustfs)

    storage.create_bucket(bucket)
    code = collect(args, storage, bucket)
    if code:
        raise SystemExit(code)


def urls_fingerprint(rows):
    urls = sorted({row["url_detalhe"] for row in rows})
    return hashlib.sha256(json.dumps(urls, ensure_ascii=False).encode()).hexdigest()


def scan_pages(args, raw_mode, fetch_downloads=True):
    obras, pages, seen = [], [], set()
    for pagina in range(args.pagina_inicial, args.pagina_final + 1):
        try:
            rows = (
                get_obras_machado(pagina)
                if fetch_downloads
                else get_obras_machado(pagina, fetch_downloads=False)
            )
            print(f"[+] Página {pagina}: {len(rows)} registros")
            if raw_mode and not rows:
                raise ValueError("Página vazia ou HTML inesperado.")
            for row in rows:
                if row["url_detalhe"] not in seen:
                    seen.add(row["url_detalhe"])
                    obras.append(row)
            pages.append({"page": pagina, "status": "ok", "records": len(rows)})
        except Exception as exc:
            pages.append({"page": pagina, "status": "error", "error": str(exc)})
            print(f"[ERRO] Página {pagina}: {exc}")
        time.sleep(args.delay)
    return obras, pages


def save_listing_status(args, storage, bucket, obras, pages):
    report = {
        "version": 1,
        "raw_id": args.raw_id,
        "source_query": LIST_URL,
        "first_page": args.pagina_inicial,
        "last_page": args.pagina_final,
        "complete": bool(obras) and all(page["status"] == "ok" for page in pages),
        "pages": pages,
        "urls_sha256": urls_fingerprint(obras),
        "updated_at": utc_now(),
    }
    storage.upload_bytes(
        bucket,
        raw_prefix(args.raw_id) + "/crawl-status.json",
        json.dumps(report, ensure_ascii=False, indent=2).encode(),
        "application/json",
    )
    return report


def listing_matches(args, report, rows):
    return (
        report.get("version") == 1
        and report.get("raw_id") == args.raw_id
        and report.get("source_query") == LIST_URL
        and (report.get("first_page"), report.get("last_page"))
        == (args.pagina_inicial, args.pagina_final)
        and report.get("complete") is True
        and [page.get("page") for page in report.get("pages", [])]
        == list(range(args.pagina_inicial, args.pagina_final + 1))
        and all(
            page.get("status") == "ok" and page.get("records", 0) > 0
            for page in report.get("pages", [])
        )
        and report.get("urls_sha256") == urls_fingerprint(rows)
    )


def ensure_listing(args, storage, bucket, rows):
    key = raw_prefix(args.raw_id) + "/crawl-status.json"
    if storage.object_exists(bucket, key):
        report = json.loads(storage.download_bytes(bucket, key))
        if listing_matches(args, report, rows):
            return True
    print("[i] Conferindo somente as listagens; PDFs concluídos serão preservados.")
    obras, pages = scan_pages(args, raw_mode=True, fetch_downloads=False)
    report = save_listing_status(args, storage, bucket, obras, pages)
    return listing_matches(args, report, rows)


def attempt_count(row):
    value = row.get("attempt_count")
    return int(value) if pd.notna(value) else 0


def resume_manifest(args, storage, bucket, prefix):
    key = prefix + "/manifest.parquet"
    frame = carregar_manifest_existente(storage, bucket, key)
    if frame.empty:
        print(
            "[pendente] Manifest vazio. Execute a coleta normal para obter o catálogo."
        )
        return 2
    rows = frame.to_dict("records")
    if args.mark_unavailable:
        if any(not re.fullmatch(r"\d{1,6}", value) for value in args.mark_unavailable):
            raise ValueError("Informe IDs numéricos do manifest, com até seis dígitos.")
        selected = {value.zfill(6) for value in args.mark_unavailable}
        indexed = {str(row["document_id"]).zfill(6): row for row in rows}
        if selected - set(indexed):
            raise ValueError(
                "IDs ausentes do manifest: "
                + ", ".join(sorted(selected - set(indexed)))
            )
        for document_id in selected:
            row = indexed[document_id]
            if row["status_ingestao"] in {
                "uploaded",
                "existing",
            } or storage.object_exists(bucket, row["raw_object_key"]):
                raise ValueError(
                    f"{document_id} já possui PDF. Confirme a integridade com --retry antes de alterar o catálogo."
                )
        for document_id in selected:
            indexed[document_id].update(
                status_ingestao="unavailable",
                unavailable_reason=args.reason.strip(),
                unavailable_at_utc=utc_now(),
            )
        salvar_manifest(storage, bucket, rows, key, quiet=True)
    else:
        for index, row in enumerate(rows):
            if row["status_ingestao"] in {"uploaded", "existing", "unavailable"}:
                continue
            print(f"[retry] {row['document_id']}: {row['titulo']}")
            try:
                links = get_download_links(row["url_detalhe"])
                obra = {
                    **row,
                    "url_download": links[0] if links else row.get("url_download"),
                    "downloads_encontrados": len(links),
                }
                rows[index] = processar_obra(
                    storage,
                    bucket,
                    obra,
                    str(row["document_id"]).zfill(6),
                    row,
                    prefix + "/documents",
                )
            except requests.RequestException as exc:
                row.update(
                    status_ingestao="erro",
                    erro_ingestao=f"Falha no detalhe: {exc}",
                    attempt_count=attempt_count(row) + 1,
                    last_attempt_at_utc=utc_now(),
                )
                print(f"[AVISO] {row['document_id']}: {row['erro_ingestao']}")
            salvar_manifest(storage, bucket, rows, key, quiet=True)
            time.sleep(args.delay)
    listing_ok = ensure_listing(args, storage, bucket, rows)
    if listing_ok:
        changed = False
        for row in rows:
            if row.get("status_crawl") == "nao_encontrado_neste_crawl":
                row["status_crawl"] = "encontrado"
                changed = True
        if changed:
            salvar_manifest(storage, bucket, rows, key, quiet=True)
    return finish_collection(
        args, storage, bucket, key, rows, listing_ok, raw_mode=True
    )


def finish_collection(args, storage, bucket, manifest_key, rows, listing_ok, raw_mode):
    frame = pd.DataFrame(rows)
    statuses = frame.get("status_ingestao", pd.Series(dtype=str))
    available = statuses.isin(["uploaded", "existing"])
    excluded = (
        statuses.eq("unavailable") if raw_mode else pd.Series(False, index=frame.index)
    )
    pending = ~(available | excluded)
    missing = (
        frame.get("status_crawl", pd.Series(index=frame.index, dtype=str)).eq(
            "nao_encontrado_neste_crawl"
        )
        & ~excluded
    )
    print(
        f"[resumo] Catálogo: {len(frame)} | PDFs: {int(available.sum())} | Pendências: {int(pending.sum())} | Indisponíveis confirmados: {int(excluded.sum())}"
    )
    if pending.any():
        columns = [
            name
            for name in ("document_id", "titulo", "status_ingestao", "erro_ingestao")
            if name in frame.columns
        ]
        print(frame.loc[pending, columns].head(10).to_string(index=False))
    print(f"Manifest: s3://{bucket}/{manifest_key}")
    if not listing_ok or missing.any() or pending.any() or frame.empty:
        if not listing_ok or missing.any():
            print(
                "[pendente] Listagens incompletas ou diferentes do manifest. Confira crawl-status.json e execute a coleta normal se faltarem registros."
            )
        print(
            "[pendente] Checkpoint salvo. Use --retry para pendências; confirme indisponibilidade por ID somente após revisão da fonte."
        )
        return 2 if args.strict or raw_mode else 0
    if raw_mode:
        payload = storage.download_bytes(bucket, manifest_key)
        count = validate_raw_manifest(
            storage,
            bucket,
            args.raw_id,
            payload,
            allow_unavailable=bool(excluded.any()),
        )
        receipt = {
            "version": 2 if excluded.any() else 1,
            "status": "ready",
            "raw_id": args.raw_id,
            "raw_bucket": bucket,
            "manifest_key": manifest_key,
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "documents": count,
            "first_page": args.pagina_inicial,
            "last_page": args.pagina_final,
            "source_query": LIST_URL,
            "created_at": utc_now(),
        }
        if excluded.any():
            receipt.update(
                catalog_records=len(frame),
                unavailable=int(excluded.sum()),
                excluded_document_ids=sorted(
                    frame.loc[excluded, "document_id"].astype(str)
                ),
            )
        storage.client.put_object(
            Bucket=bucket,
            Key=raw_prefix(args.raw_id) + "/ready.json",
            Body=json.dumps(receipt, ensure_ascii=False, indent=2).encode(),
            ContentType="application/json",
            IfNoneMatch="*",
        )
        print(
            f"[ok] Raw pronto para o Glue: {args.raw_id}; {count} PDFs verificados; {int(excluded.sum())} indisponíveis confirmados."
        )
    return 0


def collect(args, storage, bucket, manifest_key=MANIFEST_KEY, object_prefix=None):

    print(f"[+] Bucket Raw: {bucket}")

    manifest_anterior = carregar_manifest_existente(
        storage=storage,
        bucket=bucket,
        manifest_key=manifest_key,
    )

    anteriores_por_url = {}

    if not manifest_anterior.empty and "url_detalhe" in manifest_anterior.columns:
        anteriores_por_url = {
            row["url_detalhe"]: row.to_dict()
            for _, row in manifest_anterior.iterrows()
            if pd.notna(row.get("url_detalhe"))
        }

    next_id = proximo_document_id(manifest_anterior)

    obras, pages = scan_pages(args, raw_mode=bool(object_prefix))
    if object_prefix:
        report = save_listing_status(args, storage, bucket, obras, pages)

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

        if (
            object_prefix
            and anterior
            and anterior.get("status_ingestao") == "unavailable"
        ):
            registro = anterior
        else:
            registro = processar_obra(
                storage=storage,
                bucket=bucket,
                obra=obra,
                document_id=document_id,
                registro_anterior=anterior,
                object_prefix=object_prefix,
            )

        registros.append(registro)
        if object_prefix:
            # O checkpoint inclui também registros da tentativa anterior ainda não revisitados.
            checkpoint = {
                **anteriores_por_url,
                **{row["url_detalhe"]: row for row in registros},
            }
            salvar_manifest(
                storage, bucket, list(checkpoint.values()), manifest_key, quiet=True
            )
            time.sleep(args.delay)

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
        manifest_key=manifest_key,
    )

    listing_ok = (
        listing_matches(args, report, registros)
        if object_prefix
        else bool(obras) and all(page["status"] == "ok" for page in pages)
    )
    return finish_collection(
        args,
        storage,
        bucket,
        manifest_key,
        registros,
        listing_ok,
        raw_mode=bool(object_prefix),
    )


if __name__ == "__main__":
    main()
