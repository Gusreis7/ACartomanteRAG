#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import webbrowser
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.matrix import BENCHMARK_PREFIX
from src.config.config_manager import ConfigManager
from src.storage.rustfs_manager import RustFSManager

SPLIT_KEY = f"{BENCHMARK_PREFIX}/prepared/splits/benchmark_with_split.csv"
REPORT_KEY_NAME = "mcq_dashboard.html"
OPTION_KEYS = ("A", "B", "C", "D", "E")

RESULT_COLUMNS = [
    "query_id", "config_id", "selection_rank", "consensus_score", "context_id",
    "model", "canonical_work", "difficulty", "query_type", "query", "answer_text",
    "selected_option", "selected_class", "correct_option", "is_correct", "is_ambiguous",
    "is_wrong", "is_invalid", "generation_latency_ms", "input_tokens", "output_tokens",
    "total_tokens", "tokens_per_second", "generation_attempts", "context_chars",
    "used_evidence_count", "generation_error_type", "generation_error_message",
    "structured_parsing_error", "actual_digest", "expected_ollama_id",
]

QUESTION_COLUMNS = [
    "query_id", "canonical_work", "query_type", "difficulty", "query_rewritten",
    "correct_option", "gold_answer", "mcq_answer",
] + [item for option in OPTION_KEYS for item in (f"option_{option}", f"option_{option}_class")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera dashboard HTML self-contained para um run MCQ do ACartomante."
    )
    parser.add_argument("--run-id", required=True, help="Run principal, normalmente o DEV. Ex.: mcq_dev_20261001T212056Z")
    parser.add_argument(
        "--holdout-run-id",
        default=None,
        help=(
            "Run HOLDOUT opcional. Quando informado, o HTML ganha seletor "
            "GERAL / DEV / HOLDOUT. Sem ele, HOLDOUT fica desabilitado."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML local. Padrão: reports/mcq_<run-id>.html",
    )
    parser.add_argument("--no-upload", action="store_true", help="Não envia o HTML ao RustFS.")
    parser.add_argument("--open", action="store_true", dest="open_browser")
    return parser.parse_args()


def read_json(storage: RustFSManager, bucket: str, key: str) -> dict[str, Any]:
    return json.loads(storage.download_bytes(bucket, key).decode("utf-8"))


def read_parquet(storage: RustFSManager, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(storage.download_bytes(bucket, key)))


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def frame_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    present = [column for column in columns if column in frame.columns]
    return [
        {key: scalar(value) for key, value in row.items()}
        for row in frame[present].to_dict(orient="records")
    ]


def safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )


def config_label(row: dict[str, Any]) -> str:
    if str(row.get("label") or "").strip():
        return str(row["label"])
    strategy = str(row.get("chunk_strategy") or "").strip()
    size = str(row.get("chunk_size") or "").strip()
    retrieval = str(row.get("retrieval_strategy") or "").strip()
    embedder = str(row.get("embedder") or "").strip()
    head = "/".join(item for item in (strategy, size) if item)
    parts = [item for item in (head, retrieval, embedder) if item]
    return " · ".join(parts) if parts else str(row.get("config_id") or "config")


def build_config_rows(selection: dict[str, Any], results: pd.DataFrame) -> list[dict[str, Any]]:
    selected = selection.get("selected") or []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(selected, start=1):
        row = dict(item)
        config_id = str(row.get("config_id") or "").strip()
        if not config_id:
            continue
        seen.add(config_id)
        row["selection_rank"] = int(row.get("selection_rank") or index)
        row["label"] = config_label(row)
        rows.append({key: scalar(value) for key, value in row.items()})

    for config_id in results["config_id"].astype(str).drop_duplicates():
        if config_id in seen:
            continue
        rank_values = pd.to_numeric(
            results.loc[results["config_id"].astype(str).eq(config_id), "selection_rank"],
            errors="coerce",
        ).dropna()
        rank = int(rank_values.iloc[0]) if len(rank_values) else len(rows) + 1
        rows.append({"config_id": config_id, "selection_rank": rank, "label": config_id})

    rows.sort(key=lambda row: (int(row.get("selection_rank") or 999), str(row.get("config_id"))))
    return rows


def overall_payload(results: pd.DataFrame) -> dict[str, Any]:
    def rate(column: str) -> float:
        return float(results[column].astype(float).mean()) if len(results) else 0.0

    latency = pd.to_numeric(results["generation_latency_ms"], errors="coerce").dropna()
    return {
        "generations": int(len(results)),
        "accuracy": rate("is_correct"),
        "ambiguous_rate": rate("is_ambiguous"),
        "wrong_rate": rate("is_wrong"),
        "invalid_rate": rate("is_invalid"),
        "latency_mean_ms": float(latency.mean()) if len(latency) else 0.0,
        "latency_p50_ms": float(latency.quantile(0.50)) if len(latency) else 0.0,
        "latency_p95_ms": float(latency.quantile(0.95)) if len(latency) else 0.0,
    }


def render_html(
    *,
    run_id: str,
    run_manifest: dict[str, Any],
    run_config: dict[str, Any],
    results: pd.DataFrame,
    questions: pd.DataFrame,
    configs: list[dict[str, Any]],
) -> str:
    result_rows = frame_records(results, RESULT_COLUMNS)
    question_rows = frame_records(questions, QUESTION_COLUMNS)
    model_order = [
        str(item.get("name"))
        for item in (run_config.get("models") or [])
        if str(item.get("name") or "").strip()
    ]
    seen_models = set(model_order)
    for model in results["model"].astype(str).drop_duplicates():
        if model not in seen_models:
            model_order.append(model)
            seen_models.add(model)

    meta = {
        "run_id": run_id,
        "split": run_config.get("split") or run_manifest.get("split"),
        "source_retrieval_run_id": run_config.get("source_retrieval_run_id") or run_manifest.get("source_retrieval_run_id"),
        "benchmark_sha256": run_config.get("benchmark_sha256") or run_manifest.get("benchmark_sha256"),
        "prompt_version": (run_config.get("prompt") or {}).get("version"),
        "generation_overrides": run_config.get("generation_overrides") or {},
        "models": model_order,
        "configs": configs,
        "overall": overall_payload(results),
    }

    html = r'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>ACartomante · MCQ Benchmark</title>
<style>
:root{--bg:#f4f7fb;--card:#fff;--text:#182033;--muted:#667085;--line:#e5eaf2;--blue:#315efb;--purple:#7655d9;--green:#15803d;--amber:#b45309;--red:#b42318;--gray:#667085;--shadow:0 8px 24px rgba(17,24,39,.055);--radius:14px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit;color:inherit}.shell{max-width:1600px;margin:0 auto;padding:22px}.top{display:flex;gap:18px;align-items:flex-start;justify-content:space-between;margin-bottom:14px}.top-meta{display:flex;gap:8px;align-items:center;justify-content:flex-end;flex-wrap:wrap}.title h1{margin:0;font-size:24px;letter-spacing:-.02em}.title p{margin:4px 0 0;color:var(--muted)}.pill{display:inline-flex;align-items:center;border:1px solid var(--line);background:#fff;border-radius:999px;padding:6px 10px;color:var(--muted);font-size:12px}.tabs{display:flex;gap:6px;overflow:auto;border-bottom:1px solid var(--line);margin-bottom:16px;padding:0 2px}.tab{border:0;background:transparent;padding:10px 14px;border-radius:9px 9px 0 0;cursor:pointer;white-space:nowrap;color:var(--muted);font-weight:650}.tab.active{color:var(--blue);background:#fff;box-shadow:0 -1px 0 var(--line),1px 0 0 var(--line),-1px 0 0 var(--line)}.panel{display:none}.panel.active{display:block}.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px}.span-12{grid-column:span 12}.span-8{grid-column:span 8}.span-6{grid-column:span 6}.span-4{grid-column:span 4}.span-3{grid-column:span 3}.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);padding:16px;min-width:0}.card h2,.card h3{margin:0 0 4px;letter-spacing:-.01em}.card h2{font-size:17px}.card h3{font-size:14px}.sub{color:var(--muted);font-size:12px;margin-bottom:12px}.kpis{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:10px}.kpi{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px}.kpi .v{font-size:22px;font-weight:760;letter-spacing:-.025em}.kpi .l{color:var(--muted);font-size:11px;margin-top:2px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}.toolbar select,.toolbar input{height:36px;border:1px solid var(--line);border-radius:9px;background:#fff;padding:0 10px;min-width:150px}.toolbar input[type=search]{min-width:260px;flex:1}.btn{height:36px;border:1px solid var(--line);background:#fff;border-radius:9px;padding:0 12px;cursor:pointer;font-weight:650}.btn:hover{border-color:#cbd5e1}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:11px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{background:#f8fafc;color:#475467;font-size:11px;position:sticky;top:0;z-index:1;cursor:pointer}th:first-child,td:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#fafcff}.badge{display:inline-flex;padding:3px 7px;border-radius:999px;font-size:11px;font-weight:700;border:1px solid transparent}.correct{background:#ecfdf3;color:#067647;border-color:#abefc6}.ambiguous{background:#fff7ed;color:#b54708;border-color:#fed7aa}.wrong{background:#fef3f2;color:#b42318;border-color:#fecdca}.invalid{background:#f2f4f7;color:#475467;border-color:#d0d5dd}.bar-stack{display:flex;height:22px;border-radius:7px;overflow:hidden;background:#eef2f6;min-width:220px}.bar-stack>span{height:100%}.bar-correct{background:#22c55e}.bar-ambiguous{background:#f59e0b}.bar-wrong{background:#ef4444}.bar-invalid{background:#94a3b8}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin:8px 0}.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}.metric-bars{display:grid;gap:9px}.metric-row{display:grid;grid-template-columns:minmax(160px,260px) 1fr 68px;gap:10px;align-items:center}.metric-row .name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.track{height:12px;background:#eef2f7;border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--purple));border-radius:999px}.heat{border-collapse:separate;border-spacing:4px;width:100%}.heat th{position:static;background:transparent;border:0;cursor:default;text-align:center}.heat td{border:0;border-radius:7px;text-align:center;font-weight:750;padding:10px 6px}.question-box{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:14px}.question-text{font-size:16px;font-weight:680;margin-bottom:12px}.options{display:grid;gap:7px}.option{border:1px solid var(--line);background:#fff;border-radius:9px;padding:9px 10px;display:grid;grid-template-columns:26px 1fr auto;gap:8px;align-items:start}.option-letter{font-weight:800}.option.selected{outline:2px solid #93c5fd}.option.truth{border-color:#86efac;background:#f0fdf4}.model-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:12px}.response{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fff}.response h3{display:flex;justify-content:space-between;gap:10px;align-items:center}.answer{white-space:pre-wrap;background:#f8fafc;border-radius:9px;padding:10px;margin-top:9px;min-height:58px}.meta-line{color:var(--muted);font-size:11px;margin-top:8px}.answer-list{display:grid;gap:10px}.answer-card{border:1px solid var(--line);border-radius:12px;background:#fff;overflow:hidden}.answer-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;padding:12px 14px;background:#fbfcfe;border-bottom:1px solid var(--line)}.answer-body{padding:14px}.answer-body .question-text{font-size:14px}.compact-options{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px;margin-top:9px}.compact-option{font-size:11px;border:1px solid var(--line);border-radius:8px;padding:7px;background:#fff;overflow:hidden}.compact-option.sel{outline:2px solid #93c5fd}.compact-option.ok{background:#f0fdf4;border-color:#86efac}.model-summary{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:14px}.empty{padding:24px;color:var(--muted);text-align:center}.muted{color:var(--muted)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.pager{display:flex;align-items:center;justify-content:flex-end;gap:8px;margin-top:10px}.section-gap{height:2px}.note{border-left:3px solid #a78bfa;padding:8px 10px;background:#faf8ff;border-radius:0 8px 8px 0;color:#5b4b84;font-size:12px}.sort-arrow{font-size:10px;margin-left:3px}.config-label{white-space:normal;min-width:220px}.status-strip{display:flex;gap:3px;width:100%;height:10px;border-radius:999px;overflow:hidden;background:#eef2f7}.status-strip span{height:100%}@media(max-width:1000px){.span-8,.span-6,.span-4,.span-3{grid-column:span 12}.kpis{grid-template-columns:repeat(2,1fr)}.model-grid{grid-template-columns:1fr}.model-summary{grid-template-columns:repeat(2,1fr)}.compact-options{grid-template-columns:1fr}.shell{padding:12px}.top{flex-direction:column}}
</style>
</head>
<body><div class="shell">
<div class="top"><div class="title"><h1>ACartomante · MCQ Benchmark</h1><p id="subtitle"></p></div><div class="top-meta"><div class="pill" id="datasetPill"></div><div class="pill" id="protocolPill"></div></div></div>
<div class="tabs" id="tabs"></div>
<div id="panels"></div>
</div>
<script>
const META=__META__;
const ROWS=__ROWS__;
const QUESTIONS=__QUESTIONS__;
const CONFIGS=__CONFIGS__;
const $=s=>document.querySelector(s); const $$=s=>Array.from(document.querySelectorAll(s));
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
const pct=v=>`${(100*Number(v||0)).toFixed(1)}%`; const ms=v=>`${Number(v||0).toFixed(1)} ms`; const num=v=>Number(v||0).toFixed(3);
const clsLabel=v=>({correct:"correta",ambiguous:"ambígua",wrong:"errada",invalid:"inválida"}[v]||v||"—");
const qmap=new Map(QUESTIONS.map(q=>[String(q.query_id),q]));
const cmap=new Map(CONFIGS.map(c=>[String(c.config_id),c]));
const configLabel=id=>cmap.get(String(id))?.label||String(id);
const modelShort=m=>({"llama3:latest":"Llama 3","qwen3.5:4b":"Qwen 3.5 4B","ministral-3:3b":"Ministral 3B","mistral:7b-instruct-v0.3-q4_K_M":"Mistral 7B"}[m]||m);
const difficultyOrder=["easy","medium","hard"];
const difficultyWeights={easy:0.25,medium:0.35,hard:0.40};
function weightedAccuracy(rows){
  let score=0, weightSum=0;
  difficultyOrder.forEach(d=>{
    const subset=rows.filter(r=>r.difficulty===d);
    if(!subset.length)return;
    const acc=subset.reduce((sum,r)=>sum+(r.is_correct?1:0),0)/subset.length;
    const weight=difficultyWeights[d];
    score+=weight*acc;
    weightSum+=weight;
  });
  return weightSum?score/weightSum:0;
}
function agg(rows){
  const n=rows.length||1, vals=k=>rows.reduce((s,r)=>s+(r[k]?1:0),0)/n;
  const lats=rows.map(r=>Number(r.generation_latency_ms)).filter(Number.isFinite).sort((a,b)=>a-b);
  const quant=q=>lats.length?lats[Math.min(lats.length-1,Math.floor((lats.length-1)*q))]:0;
  const mean=k=>{const x=rows.map(r=>Number(r[k])).filter(Number.isFinite);return x.length?x.reduce((a,b)=>a+b,0)/x.length:0};
  const labels=['A','B','C','D','E'];
  const precisionByClass=[]; const recallByClass=[];
  labels.forEach(label=>{
    let tp=0,fp=0,fn=0,support=0;
    rows.forEach(r=>{
      const truth=String(r.correct_option||'').toUpperCase();
      const pred=String(r.selected_option||'').toUpperCase();
      if(truth===label)support++;
      if(pred===label&&truth===label)tp++;
      else if(pred===label&&truth!==label)fp++;
      if(truth===label&&pred!==label)fn++;
    });
    if(support>0){
      precisionByClass.push((tp+fp)>0?tp/(tp+fp):0);
      recallByClass.push((tp+fn)>0?tp/(tp+fn):0);
    }
  });
  const macro=x=>x.length?x.reduce((a,b)=>a+b,0)/x.length:0;
  return {n:rows.length,accuracy:vals("is_correct"),weightedAccuracy:weightedAccuracy(rows),precision:macro(precisionByClass),recall:macro(recallByClass),ambiguous:vals("is_ambiguous"),wrong:vals("is_wrong"),invalid:vals("is_invalid"),latency:mean("generation_latency_ms"),p95:quant(.95),input:mean("input_tokens"),output:mean("output_tokens"),tps:mean("tokens_per_second")};
}
function by(rows,key){const m=new Map();rows.forEach(r=>{const k=String(r[key]??"");if(!m.has(k))m.set(k,[]);m.get(k).push(r)});return m}
function statusStack(a){return `<div class="bar-stack" title="correta ${pct(a.accuracy)} · ambígua ${pct(a.ambiguous)} · errada ${pct(a.wrong)} · inválida ${pct(a.invalid)}"><span class="bar-correct" style="width:${a.accuracy*100}%"></span><span class="bar-ambiguous" style="width:${a.ambiguous*100}%"></span><span class="bar-wrong" style="width:${a.wrong*100}%"></span><span class="bar-invalid" style="width:${a.invalid*100}%"></span></div>`}
function kpi(v,l){return `<div class="kpi"><div class="v">${v}</div><div class="l">${esc(l)}</div></div>`}
function tabButton(id,label,active=false){return `<button class="tab ${active?'active':''}" data-tab="${esc(id)}">${esc(label)}</button>`}
function setTab(id){$$('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===id));$$('.panel').forEach(x=>x.classList.toggle('active',x.id===`panel-${id}`));if(id==='compare')renderCompare();if(id.startsWith('model-'))renderModelTab(id.slice(6));}
function setupTabs(){
  const tabs=[['general','Geral'],['compare','Comparação lado a lado'],...META.models.map((m,i)=>[`model-${i}`,modelShort(m)])];
  $('#tabs').innerHTML=tabs.map((t,i)=>tabButton(t[0],t[1],i===0)).join('');
  $('#panels').innerHTML=`<section class="panel active" id="panel-general"></section><section class="panel" id="panel-compare"></section>`+META.models.map((m,i)=>`<section class="panel" id="panel-model-${i}"></section>`).join('');
  $$('.tab').forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
  META.models.forEach((m,i)=>document.querySelector(`#panel-model-${i}`).dataset.model=m);
}
function renderGeneral(){
  const overall=agg(ROWS); const mg=by(ROWS,'model'); const cg=by(ROWS,'config_id');
  const modelRows=META.models.map(m=>({model:m,...agg(mg.get(m)||[])}));
  const configRows=CONFIGS.map(c=>({config:c,...agg(cg.get(String(c.config_id))||[])}));
  const difficultyRows=[];
  META.models.forEach(m=>difficultyOrder.forEach(d=>difficultyRows.push({model:m,difficulty:d,...agg((mg.get(m)||[]).filter(x=>x.difficulty===d))})));
  $('#panel-general').innerHTML=`<div class="grid">
    <div class="span-12"><div class="kpis">${kpi(overall.n.toLocaleString('pt-BR'),'gerações')}${kpi(pct(overall.accuracy),'accuracy')}${kpi(pct(overall.weightedAccuracy),'accuracy ponderada')}${kpi(pct(overall.precision),'precisão macro')}${kpi(pct(overall.recall),'recall macro')}${kpi(pct(overall.wrong),'erradas')}${kpi(ms(overall.p95),'latência p95')}</div></div>
    <div class="card span-12"><h2>Desempenho médio por modelo</h2><div class="sub">Cada modelo agrega as 10 configurações de retrieval e as ${new Set(ROWS.map(r=>r.query_id)).size} questões ${String(META.split||'').toUpperCase()}. Accuracy ponderada = 0,25 × easy + 0,35 × medium + 0,40 × hard. Precisão e recall são macro sobre as alternativas A–E; respostas inválidas reduzem o recall da alternativa correta.</div><div class="legend"><span><i style="background:#22c55e"></i>correta</span><span><i style="background:#f59e0b"></i>ambígua</span><span><i style="background:#ef4444"></i>errada</span><span><i style="background:#94a3b8"></i>inválida</span></div><div class="table-wrap"><table><thead><tr><th>Modelo</th><th>N</th><th>Distribuição</th><th>Accuracy</th><th>Acc. ponderada</th><th>Precisão</th><th>Recall</th><th>Ambígua</th><th>Errada</th><th>Inválida</th><th>Lat. média</th><th>p95</th><th>tok/s</th></tr></thead><tbody>${modelRows.map(r=>`<tr><td><b>${esc(modelShort(r.model))}</b><div class="muted mono">${esc(r.model)}</div></td><td>${r.n}</td><td>${statusStack(r)}</td><td><b>${pct(r.accuracy)}</b></td><td><b>${pct(r.weightedAccuracy)}</b></td><td><b>${pct(r.precision)}</b></td><td><b>${pct(r.recall)}</b></td><td>${pct(r.ambiguous)}</td><td>${pct(r.wrong)}</td><td>${pct(r.invalid)}</td><td>${ms(r.latency)}</td><td>${ms(r.p95)}</td><td>${r.tps.toFixed(1)}</td></tr>`).join('')}</tbody></table></div></div>
    <div class="card span-12"><h2>Desempenho por dificuldade</h2><div class="sub">Uma linha por modelo e dificuldade, sem repetir a distribuição global do dataset.</div><div class="table-wrap"><table><thead><tr><th>Modelo</th><th>Dificuldade</th><th>N</th><th>Accuracy</th><th>Precisão</th><th>Recall</th><th>Ambígua</th><th>Errada</th><th>Inválida</th></tr></thead><tbody>${difficultyRows.map(r=>`<tr><td><b>${esc(modelShort(r.model))}</b></td><td><b>${esc(r.difficulty)}</b></td><td>${r.n}</td><td><b>${pct(r.accuracy)}</b></td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.ambiguous)}</td><td>${pct(r.wrong)}</td><td>${pct(r.invalid)}</td></tr>`).join('')}</tbody></table></div></div>
    <div class="card span-12"><h2>Desempenho médio das 10 abordagens de retrieval</h2><div class="sub">Cada linha agrega os quatro modelos. Accuracy ponderada = 0,25 × easy + 0,35 × medium + 0,40 × hard. Útil para verificar se a configuração de retrieval continua influenciando a etapa downstream.</div><div class="table-wrap"><table id="configTable"><thead><tr><th>Abordagem</th><th>Rank seleção</th><th>N</th><th>Distribuição</th><th>Accuracy</th><th>Acc. ponderada</th><th>Precisão</th><th>Recall</th><th>Ambígua</th><th>Errada</th><th>Inválida</th><th>Lat. média</th><th>p95</th></tr></thead><tbody>${configRows.map(r=>`<tr><td class="config-label"><b>${esc(r.config.label)}</b><div class="muted mono">${esc(r.config.config_id)}</div></td><td>${r.config.selection_rank??'—'}</td><td>${r.n}</td><td>${statusStack(r)}</td><td><b>${pct(r.accuracy)}</b></td><td><b>${pct(r.weightedAccuracy)}</b></td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.ambiguous)}</td><td>${pct(r.wrong)}</td><td>${pct(r.invalid)}</td><td>${ms(r.latency)}</td><td>${ms(r.p95)}</td></tr>`).join('')}</tbody></table></div></div>
    <div class="card span-12"><h2>Modelo × retrieval · Accuracy</h2><div class="sub">Cada célula contém ${new Set(ROWS.map(r=>r.query_id)).size} questões. Quanto maior a intensidade, maior a accuracy da combinação.</div>${heatmapAccuracy()}</div>
  </div>`;
}
function heatmapAccuracy(){
  const vals=[]; CONFIGS.forEach(c=>META.models.forEach(m=>vals.push(agg(ROWS.filter(r=>r.config_id===c.config_id&&r.model===m)).accuracy))); const lo=Math.min(...vals),hi=Math.max(...vals);
  const color=v=>{const t=hi>lo?(v-lo)/(hi-lo):.5;const a=.10+.55*t;return `rgba(49,94,251,${a})`};
  return `<div class="table-wrap"><table class="heat"><thead><tr><th>Retrieval</th>${META.models.map(m=>`<th>${esc(modelShort(m))}</th>`).join('')}</tr></thead><tbody>${CONFIGS.map(c=>`<tr><td style="text-align:left;white-space:normal"><b>${esc(c.label)}</b></td>${META.models.map(m=>{const a=agg(ROWS.filter(r=>r.config_id===c.config_id&&r.model===m));return `<td style="background:${color(a.accuracy)}">${pct(a.accuracy)}</td>`}).join('')}</tr>`).join('')}</tbody></table></div>`
}
function questionOptions(q, selected=null){return `<div class="options">${['A','B','C','D','E'].map(o=>{const klass=q[`option_${o}_class`];return `<div class="option ${selected===o?'selected':''} ${klass==='correct'?'truth':''}"><div class="option-letter">${o}</div><div>${esc(q[`option_${o}`])}</div><div>${klass==='correct'?'<span class="badge correct">gabarito</span>':''}</div></div>`}).join('')}</div>`}
function compareFilters(){
  const works=[...new Set(ROWS.map(r=>r.canonical_work))].sort(); const types=[...new Set(ROWS.map(r=>r.query_type))].sort();
  return `<div class="toolbar"><select id="cmpConfig">${CONFIGS.map(c=>`<option value="${esc(c.config_id)}">${esc(c.label)}</option>`).join('')}</select><select id="cmpDifficulty"><option value="">Todas dificuldades</option>${difficultyOrder.map(d=>`<option>${d}</option>`).join('')}</select><select id="cmpType"><option value="">Todos tipos</option>${types.map(x=>`<option>${esc(x)}</option>`).join('')}</select><select id="cmpWork"><option value="">Todas obras</option>${works.map(x=>`<option>${esc(x)}</option>`).join('')}</select><select id="cmpQuery"></select><button class="btn" id="cmpPrev">←</button><button class="btn" id="cmpNext">→</button></div>`
}
function renderCompare(){
  const p=$('#panel-compare'); if(!p.dataset.ready){p.innerHTML=`<div class="card"> <h2>Comparação lado a lado</h2><div class="sub">Mesma questão e mesmo contexto de retrieval, com a resposta dos quatro modelos em paralelo.</div>${compareFilters()}<div id="compareBody"></div></div>`;p.dataset.ready='1';['cmpConfig','cmpDifficulty','cmpType','cmpWork'].forEach(id=>$('#'+id).onchange=()=>{refreshCompareQueries();renderCompareBody()});$('#cmpQuery').onchange=renderCompareBody;$('#cmpPrev').onclick=()=>stepCompare(-1);$('#cmpNext').onclick=()=>stepCompare(1);refreshCompareQueries();}
  renderCompareBody();
}
function filteredCompareQueryIds(){const c=$('#cmpConfig').value,d=$('#cmpDifficulty').value,t=$('#cmpType').value,w=$('#cmpWork').value;return [...new Set(ROWS.filter(r=>r.config_id===c&&(!d||r.difficulty===d)&&(!t||r.query_type===t)&&(!w||r.canonical_work===w)).map(r=>r.query_id))].sort((a,b)=>{const qa=qmap.get(a),qb=qmap.get(b);return String(qa?.canonical_work||'').localeCompare(String(qb?.canonical_work||''),'pt-BR')||String(qa?.query_rewritten||'').localeCompare(String(qb?.query_rewritten||''),'pt-BR')})}
function refreshCompareQueries(){const sel=$('#cmpQuery');const old=sel.value;const ids=filteredCompareQueryIds();sel.innerHTML=ids.map(id=>{const q=qmap.get(id);return `<option value="${esc(id)}">${esc((q?.canonical_work||'')+' · '+(q?.query_rewritten||id))}</option>`}).join('');if(ids.includes(old))sel.value=old;}
function stepCompare(dir){const sel=$('#cmpQuery');if(!sel.options.length)return;sel.selectedIndex=(sel.selectedIndex+dir+sel.options.length)%sel.options.length;renderCompareBody()}
function renderCompareBody(){const config=$('#cmpConfig')?.value,qid=$('#cmpQuery')?.value;if(!config||!qid){$('#compareBody').innerHTML='<div class="empty">Nenhuma questão para os filtros.</div>';return}const q=qmap.get(qid);const rows=META.models.map(m=>ROWS.find(r=>r.config_id===config&&r.query_id===qid&&r.model===m)).filter(Boolean);$('#compareBody').innerHTML=`<div class="question-box"><div class="muted">${esc(q.canonical_work)} · ${esc(q.difficulty)} · ${esc(q.query_type)} · <span class="mono">${esc(qid)}</span></div><div class="question-text">${esc(q.query_rewritten)}</div>${questionOptions(q)}</div><div class="model-grid">${rows.map(r=>`<div class="response"><h3><span>${esc(modelShort(r.model))}</span><span class="badge ${esc(r.selected_class)}">${esc(clsLabel(r.selected_class))}</span></h3><div><b>Escolheu ${esc(r.selected_option||'—')}</b> · gabarito ${esc(r.correct_option)}</div><div class="answer">${esc(r.answer_text||'(sem resposta textual)')}</div><div class="meta-line">${ms(r.generation_latency_ms)} · ${Number(r.input_tokens||0)} in / ${Number(r.output_tokens||0)} out · ${Number(r.tokens_per_second||0).toFixed(1)} tok/s · tentativas ${Number(r.generation_attempts||0)}</div>${r.generation_error_message?`<div class="note" style="margin-top:8px">${esc(r.generation_error_type||'erro')}: ${esc(r.generation_error_message)}</div>`:''}</div>`).join('')}</div>`}
function modelPanelIndex(index){return document.querySelector(`#panel-model-${index}`)}
const modelState={};
function renderModelTab(index){const panel=modelPanelIndex(index);const model=panel.dataset.model;if(!modelState[model])modelState[model]={page:0,size:25};const state=modelState[model];if(!panel.dataset.ready){panel.innerHTML=modelTabSkeleton(model);panel.dataset.ready='1';panel.querySelectorAll('select,input').forEach(el=>el.addEventListener(el.tagName==='INPUT'?'input':'change',()=>{state.page=0;renderModelContents(model,index)}));panel.querySelector('[data-prev]').onclick=()=>{state.page=Math.max(0,state.page-1);renderModelContents(model,index)};panel.querySelector('[data-next]').onclick=()=>{state.page++;renderModelContents(model,index)};}renderModelContents(model,index)}
function modelTabSkeleton(model){const rows=ROWS.filter(r=>r.model===model),works=[...new Set(rows.map(r=>r.canonical_work))].sort(),types=[...new Set(rows.map(r=>r.query_type))].sort();return `<div class="grid"><div class="span-12" data-model-summary></div><div class="card span-12"><h2>Respostas · ${esc(modelShort(model))}</h2><div class="sub">Inspeção das ${rows.length} combinações questão × retrieval. A alternativa verde é o gabarito; contorno azul indica a escolha do modelo.</div><div class="toolbar"><input type="search" data-search placeholder="Buscar pergunta, obra, resposta ou query_id"><select data-config><option value="">Todas configs</option>${CONFIGS.map(c=>`<option value="${esc(c.config_id)}">${esc(c.label)}</option>`).join('')}</select><select data-difficulty><option value="">Todas dificuldades</option>${difficultyOrder.map(d=>`<option>${d}</option>`).join('')}</select><select data-type><option value="">Todos tipos</option>${types.map(x=>`<option>${esc(x)}</option>`).join('')}</select><select data-work><option value="">Todas obras</option>${works.map(x=>`<option>${esc(x)}</option>`).join('')}</select><select data-status><option value="">Todos resultados</option><option value="correct">correta</option><option value="ambiguous">ambígua</option><option value="wrong">errada</option><option value="invalid">inválida</option></select><select data-size><option value="25">25/pág.</option><option value="50">50/pág.</option><option value="100">100/pág.</option></select></div><div data-answer-list class="answer-list"></div><div class="pager"><span data-page-label class="muted"></span><button class="btn" data-prev>← Anterior</button><button class="btn" data-next>Próxima →</button></div></div></div>`}
function filteredModelRows(model,index){const p=modelPanelIndex(index),term=(p.querySelector('[data-search]').value||'').toLowerCase(),cfg=p.querySelector('[data-config]').value,d=p.querySelector('[data-difficulty]').value,t=p.querySelector('[data-type]').value,w=p.querySelector('[data-work]').value,s=p.querySelector('[data-status]').value;return ROWS.filter(r=>r.model===model&&(!cfg||r.config_id===cfg)&&(!d||r.difficulty===d)&&(!t||r.query_type===t)&&(!w||r.canonical_work===w)&&(!s||r.selected_class===s)&&(!term||[r.query,r.answer_text,r.canonical_work,r.query_id,configLabel(r.config_id)].some(v=>String(v||'').toLowerCase().includes(term))))}
function renderModelContents(model,index){const panel=modelPanelIndex(index),state=modelState[model],all=filteredModelRows(model,index),a=agg(all),mg=by(all,'config_id');state.size=Number(panel.querySelector('[data-size]').value||25);const pages=Math.max(1,Math.ceil(all.length/state.size));state.page=Math.min(state.page,pages-1);const slice=all.slice(state.page*state.size,(state.page+1)*state.size);panel.querySelector('[data-model-summary]').innerHTML=`<div class="model-summary">${kpi(all.length.toLocaleString('pt-BR'),'itens filtrados')}${kpi(pct(a.accuracy),'accuracy')}${kpi(pct(a.precision),'precisão macro')}${kpi(pct(a.recall),'recall macro')}${kpi(pct(a.wrong),'erradas')}${kpi(ms(a.p95),'latência p95')}</div><div class="card"><h2>${esc(modelShort(model))} · por retrieval</h2><div class="sub">Accuracy dentro do filtro atual.</div><div class="metric-bars">${CONFIGS.map(c=>({c,a:agg(mg.get(c.config_id)||[])})).filter(x=>x.a.n).sort((x,y)=>y.a.accuracy-x.a.accuracy).map(x=>`<div class="metric-row"><div class="name" title="${esc(x.c.label)}">${esc(x.c.label)}</div><div class="track"><div class="fill" style="width:${x.a.accuracy*100}%"></div></div><b>${pct(x.a.accuracy)}</b></div>`).join('')||'<div class="empty">Sem dados.</div>'}</div>`;panel.querySelector('[data-answer-list]').innerHTML=slice.length?slice.map(answerCard).join(''):'<div class="empty">Nenhum resultado para os filtros.</div>';panel.querySelector('[data-page-label]').textContent=`${all.length} itens · página ${state.page+1}/${pages}`;panel.querySelector('[data-prev]').disabled=state.page<=0;panel.querySelector('[data-next]').disabled=state.page>=pages-1}
function answerCard(r){const q=qmap.get(String(r.query_id))||{};return `<article class="answer-card"><div class="answer-head"><div><b>${esc(q.canonical_work||r.canonical_work)}</b> · ${esc(r.difficulty)} · ${esc(r.query_type)}<div class="muted">${esc(configLabel(r.config_id))} · <span class="mono">${esc(r.query_id)}</span></div></div><span class="badge ${esc(r.selected_class)}">${esc(clsLabel(r.selected_class))}</span></div><div class="answer-body"><div class="question-text">${esc(q.query_rewritten||r.query)}</div><div class="compact-options">${['A','B','C','D','E'].map(o=>`<div class="compact-option ${r.selected_option===o?'sel':''} ${q[`option_${o}_class`]==='correct'?'ok':''}"><b>${o}</b> ${esc(q[`option_${o}`]||'')} ${r.selected_option===o?'<div><b>← escolhida</b></div>':''}${q[`option_${o}_class`]==='correct'?'<div>✓ gabarito</div>':''}</div>`).join('')}</div><div class="answer"><b>Resposta salva:</b> ${esc(r.answer_text||'(vazia)')}</div><div class="meta-line">escolhida ${esc(r.selected_option||'—')} · gabarito ${esc(r.correct_option||q.correct_option||'—')} · ${ms(r.generation_latency_ms)} · ${Number(r.input_tokens||0)} in / ${Number(r.output_tokens||0)} out · ${Number(r.tokens_per_second||0).toFixed(1)} tok/s · contexto ${Number(r.context_chars||0).toLocaleString('pt-BR')} chars / ${Number(r.used_evidence_count||0)} evidências · tentativas ${Number(r.generation_attempts||0)}</div>${r.generation_error_message?`<div class="note" style="margin-top:9px"><b>${esc(r.generation_error_type||'erro')}</b>: ${esc(r.generation_error_message)}</div>`:''}</div></article>`}
function init(){setupTabs();const uniqueQuestions=[...new Map(QUESTIONS.map(q=>[String(q.query_id),q])).values()];const counts=Object.fromEntries(difficultyOrder.map(d=>[d,uniqueQuestions.filter(q=>q.difficulty===d).length]));$('#subtitle').textContent=`run ${META.run_id} · split=${META.split} · ${META.overall.generations.toLocaleString('pt-BR')} gerações · retrieval ${META.source_retrieval_run_id}`;$('#datasetPill').textContent=`dataset: ${uniqueQuestions.length} questões · ${counts.easy||0} easy · ${counts.medium||0} medium · ${counts.hard||0} hard`;$('#protocolPill').textContent=`${META.models.length} modelos × ${CONFIGS.length} retrievals`;renderGeneral();}
init();
</script></body></html>'''
    return (
        html.replace("__META__", safe_json(meta))
        .replace("__ROWS__", safe_json(result_rows))
        .replace("__QUESTIONS__", safe_json(question_rows))
        .replace("__CONFIGS__", safe_json(configs))
    )



def _classification_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "n": 0, "accuracy": 0.0, "weighted_accuracy": 0.0,
            "precision": 0.0, "recall": 0.0, "ambiguous": 0.0,
            "wrong": 0.0, "invalid": 0.0, "latency_p95_ms": 0.0,
        }

    def rate(column: str) -> float:
        return float(frame[column].astype(float).mean())

    weights = {"easy": 0.25, "medium": 0.35, "hard": 0.40}
    weighted = 0.0
    weight_sum = 0.0
    for difficulty, weight in weights.items():
        subset = frame.loc[frame["difficulty"].astype(str).eq(difficulty)]
        if subset.empty:
            continue
        weighted += weight * float(subset["is_correct"].astype(float).mean())
        weight_sum += weight
    weighted_accuracy = weighted / weight_sum if weight_sum else 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    truth = frame["correct_option"].astype(str).str.upper()
    pred = frame["selected_option"].fillna("").astype(str).str.upper()
    for label in OPTION_KEYS:
        support = int((truth == label).sum())
        if support <= 0:
            continue
        tp = int(((truth == label) & (pred == label)).sum())
        fp = int(((truth != label) & (pred == label)).sum())
        fn = int(((truth == label) & (pred != label)).sum())
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)

    latency = pd.to_numeric(frame["generation_latency_ms"], errors="coerce").dropna()
    return {
        "n": int(len(frame)),
        "accuracy": rate("is_correct"),
        "weighted_accuracy": float(weighted_accuracy),
        "precision": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        "recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "ambiguous": rate("is_ambiguous"),
        "wrong": rate("is_wrong"),
        "invalid": rate("is_invalid"),
        "latency_p95_ms": float(latency.quantile(0.95)) if len(latency) else 0.0,
    }


def _question_counts(questions: pd.DataFrame) -> dict[str, int]:
    unique = questions.drop_duplicates(subset=["query_id"]).copy()
    return {
        "total": int(len(unique)),
        "easy": int(unique["difficulty"].astype(str).eq("easy").sum()),
        "medium": int(unique["difficulty"].astype(str).eq("medium").sum()),
        "hard": int(unique["difficulty"].astype(str).eq("hard").sum()),
    }


def load_mcq_run(storage: RustFSManager, bucket: str, run_id: str) -> dict[str, Any]:
    run_id = str(run_id).strip().replace("/", "_")
    run_prefix = f"{BENCHMARK_PREFIX}/runs/{run_id}"
    mcq_prefix = f"{run_prefix}/mcq"
    manifest_key = f"{run_prefix}/manifest.json"
    config_key = f"{run_prefix}/config.json"
    results_key = f"{mcq_prefix}/results.parquet"
    summary_key = f"{mcq_prefix}/summary.parquet"

    for key in (manifest_key, config_key, results_key, summary_key, SPLIT_KEY):
        if not storage.object_exists(bucket, key):
            raise FileNotFoundError(f"Artefato ausente: s3://{bucket}/{key}")

    run_manifest = read_json(storage, bucket, manifest_key)
    run_config = read_json(storage, bucket, config_key)
    if str(run_manifest.get("status")) != "success":
        raise RuntimeError(f"Run MCQ {run_id} não está com status=success: {run_manifest.get('status')}")
    if str(run_config.get("phase")) != "mcq":
        raise RuntimeError(f"{run_id}: config.json não pertence à fase MCQ.")

    results = read_parquet(storage, bucket, results_key)
    missing = sorted(set(RESULT_COLUMNS[:18]) - set(results.columns))
    if missing:
        raise RuntimeError(f"{run_id}: results.parquet sem colunas obrigatórias: {missing}")

    split = str(run_config.get("split") or run_manifest.get("split") or "").strip()
    questions = pd.read_csv(io.BytesIO(storage.download_bytes(bucket, SPLIT_KEY)))
    used_query_ids = set(results["query_id"].astype(str))
    questions = questions[questions["query_id"].astype(str).isin(used_query_ids)].copy()
    if "split" in questions.columns and split:
        questions = questions[questions["split"].astype(str).eq(split)].copy()
    if questions["query_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{run_id}: benchmark_with_split.csv contém query_id duplicado no escopo do report.")

    source_retrieval = str(run_config.get("source_retrieval_run_id") or run_manifest.get("source_retrieval_run_id") or "")
    selection_source = str(run_config.get("selection_run_id") or run_manifest.get("selection_run_id") or source_retrieval)
    selection_key = f"{BENCHMARK_PREFIX}/runs/{selection_source}/retrieval/final_selection_10.json"
    selection = read_json(storage, bucket, selection_key) if storage.object_exists(bucket, selection_key) else {"selected": []}
    configs = build_config_rows(selection, results)

    html = render_html(
        run_id=run_id,
        run_manifest=run_manifest,
        run_config=run_config,
        results=results,
        questions=questions,
        configs=configs,
    )
    return {
        "run_id": run_id, "run_prefix": run_prefix, "manifest": run_manifest,
        "config": run_config, "split": split, "results": results,
        "questions": questions, "configs": configs, "html": html,
        "benchmark_sha256": run_config.get("benchmark_sha256") or run_manifest.get("benchmark_sha256"),
        "models": sorted(results["model"].astype(str).unique().tolist()),
        "config_ids": sorted(results["config_id"].astype(str).unique().tolist()),
    }


def _validate_pair(dev: dict[str, Any], holdout: dict[str, Any]) -> None:
    if dev["split"] != "dev":
        raise RuntimeError(f"--run-id deve apontar para DEV quando --holdout-run-id é usado; split={dev['split']}")
    if holdout["split"] != "holdout":
        raise RuntimeError(f"--holdout-run-id deve apontar para HOLDOUT; split={holdout['split']}")
    if dev.get("benchmark_sha256") != holdout.get("benchmark_sha256"):
        raise RuntimeError("DEV e HOLDOUT usam benchmark_sha256 diferentes.")
    if set(dev["models"]) != set(holdout["models"]):
        raise RuntimeError("DEV e HOLDOUT não possuem o mesmo conjunto de modelos.")
    if set(dev["config_ids"]) != set(holdout["config_ids"]):
        raise RuntimeError("DEV e HOLDOUT não possuem exatamente as mesmas configs de retrieval.")


def _comparison_payload(dev: dict[str, Any], holdout: dict[str, Any] | None) -> dict[str, Any]:
    def split_payload(bundle: dict[str, Any]) -> dict[str, Any]:
        overall = _classification_metrics(bundle["results"])
        model_rows = [
            {"model": str(model), **_classification_metrics(group)}
            for model, group in bundle["results"].groupby("model", sort=False)
        ]
        labels = {str(item["config_id"]): str(item.get("label") or item["config_id"]) for item in bundle["configs"]}
        config_rows = [
            {"config_id": str(config_id), "label": labels.get(str(config_id), str(config_id)), **_classification_metrics(group)}
            for config_id, group in bundle["results"].groupby("config_id", sort=False)
        ]
        return {
            "run_id": bundle["run_id"], "split": bundle["split"], "overall": overall,
            "question_counts": _question_counts(bundle["questions"]),
            "models": model_rows, "configs": config_rows,
        }

    return {"dev": split_payload(dev), "holdout": split_payload(holdout) if holdout is not None else None}


def render_dual_html(*, dev: dict[str, Any], holdout: dict[str, Any] | None) -> str:
    import base64

    comparison = _comparison_payload(dev, holdout)
    dev_b64 = base64.b64encode(dev["html"].encode("utf-8")).decode("ascii")
    holdout_b64 = base64.b64encode(holdout["html"].encode("utf-8")).decode("ascii") if holdout is not None else ""

    template = r'''<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ACartomante · MCQ DEV × HOLDOUT</title>
<style>
:root{--bg:#f4f7fb;--card:#fff;--text:#182033;--muted:#667085;--line:#e5eaf2;--blue:#315efb;--green:#15803d;--red:#b42318;--shadow:0 8px 24px rgba(17,24,39,.055)}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.top{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid var(--line);padding:14px 20px;box-shadow:0 3px 14px rgba(17,24,39,.04)}.topline{max-width:1600px;margin:auto;display:flex;align-items:center;justify-content:space-between;gap:16px}.title h1{font-size:20px;margin:0}.title p{margin:3px 0 0;color:var(--muted);font-size:12px}.split-nav{display:flex;gap:6px}.split-btn{border:1px solid var(--line);background:#fff;border-radius:9px;padding:8px 14px;font-weight:700;cursor:pointer}.split-btn.active{background:var(--blue);border-color:var(--blue);color:#fff}.split-btn:disabled{opacity:.38;cursor:not-allowed}.view{display:none}.view.active{display:block}.compare{max-width:1500px;margin:0 auto;padding:20px}.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px}.span-12{grid-column:span 12}.span-6{grid-column:span 6}.card{background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:16px}.card h2{font-size:16px;margin:0 0 5px}.sub{color:var(--muted);font-size:12px;margin-bottom:12px}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}.kpi{border:1px solid var(--line);border-radius:11px;padding:11px}.kpi .v{font-size:21px;font-weight:760}.kpi .l{font-size:11px;color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{background:#f8fafc;color:#475467}th:first-child,td:first-child{text-align:left}.delta-pos{color:var(--green);font-weight:700}.delta-neg{color:var(--red);font-weight:700}.chart{height:300px;position:relative}.chart svg{width:100%;height:100%;overflow:visible}.chart .gridline{stroke:#edf0f6;stroke-width:1}.chart .axis{stroke:#cbd3df;stroke-width:1}.chart .tick{fill:#667085;font-size:11px}.chart .bar-dev{fill:#315efb}.chart .bar-holdout{fill:#7655d9}.chart .seg-correct{fill:#22c55e}.chart .seg-ambiguous{fill:#f59e0b}.chart .seg-wrong{fill:#ef4444}.chart .seg-invalid{fill:#94a3b8}.chart-legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin-top:7px}.chart-legend span:before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;background:#315efb}.chart-legend span.holdout:before{background:#7655d9}.chart-legend span.correct:before{background:#22c55e}.chart-legend span.ambiguous:before{background:#f59e0b}.chart-legend span.wrong:before{background:#ef4444}.chart-legend span.invalid:before{background:#94a3b8}.frame{width:100%;height:calc(100vh - 82px);border:0;background:#fff}.empty{padding:50px;text-align:center;color:var(--muted)}@media(max-width:900px){.span-6{grid-column:span 12}.kpis{grid-template-columns:repeat(2,1fr)}.topline{align-items:flex-start;flex-direction:column}}
</style></head><body>
<header class="top"><div class="topline"><div class="title"><h1>ACartomante · MCQ Benchmark</h1><p id="headerMeta"></p></div><nav class="split-nav"><button class="split-btn" id="btn-general">GERAL</button><button class="split-btn" id="btn-dev">DEV</button><button class="split-btn" id="btn-holdout">HOLDOUT</button></nav></div></header>
<section id="view-general" class="view"><div class="compare" id="generalContent"></div></section><section id="view-dev" class="view"><iframe class="frame" id="frame-dev" title="Dashboard DEV"></iframe></section><section id="view-holdout" class="view"><iframe class="frame" id="frame-holdout" title="Dashboard HOLDOUT"></iframe></section>
<script>
const DATA=__COMPARISON__,DEV_B64="__DEV_B64__",HOLDOUT_B64="__HOLDOUT_B64__",hasHoldout=Boolean(DATA.holdout);const $=id=>document.getElementById(id);const pct=v=>`${(100*Number(v||0)).toFixed(1)}%`;const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));const delta=(a,b)=>Number(b||0)-Number(a||0);const dclass=v=>v>1e-12?'delta-pos':v<-1e-12?'delta-neg':'';const dPct=v=>`${v>=0?'+':''}${(100*v).toFixed(1)} pp`;function decode64(x){if(!x)return '';const bin=atob(x),bytes=Uint8Array.from(bin,c=>c.charCodeAt(0));return new TextDecoder().decode(bytes)}
function setView(name){['general','dev','holdout'].forEach(x=>{$(`view-${x}`).classList.toggle('active',x===name);$(`btn-${x}`).classList.toggle('active',x===name)});if(name==='dev'&&!$('frame-dev').dataset.loaded){$('frame-dev').srcdoc=decode64(DEV_B64);$('frame-dev').dataset.loaded='1'}if(name==='holdout'&&hasHoldout&&!$('frame-holdout').dataset.loaded){$('frame-holdout').srcdoc=decode64(HOLDOUT_B64);$('frame-holdout').dataset.loaded='1'}}function modelMap(rows){return new Map(rows.map(x=>[x.model,x]))}function configMap(rows){return new Map(rows.map(x=>[x.config_id,x]))}
function svgEl(name,attrs={}){const e=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e}function svgText(svg,x,y,text,anchor='middle'){const t=svgEl('text',{x,y,class:'tick','text-anchor':anchor});t.textContent=text;svg.appendChild(t);return t}function chartSvg(id,w=900,h=280){const el=$(id);el.innerHTML='';const svg=svgEl('svg',{viewBox:`0 0 ${w} ${h}`});el.appendChild(svg);return svg}
function groupedPctChart(id,rows,valueKey){const svg=chartSvg(id),w=900,h=280,l=58,r=20,t=18,b=62,iw=w-l-r,ih=h-t-b,max=1;for(let i=0;i<=4;i++){const y=t+ih*(1-i/4),v=i/4;svg.appendChild(svgEl('line',{x1:l,y1:y,x2:w-r,y2:y,class:'gridline'}));svgText(svg,l-8,y+4,pct(v),'end')}svg.appendChild(svgEl('line',{x1:l,y1:t+ih,x2:w-r,y2:t+ih,class:'axis'}));const gap=iw/rows.length,bw=Math.min(34,gap*.28);rows.forEach((row,i)=>{const cx=l+gap*(i+.5),dv=Number(row.dev?.[valueKey]||0),hv=Number(row.holdout?.[valueKey]||0);[[dv,-bw*.65,'bar-dev','DEV'],[hv,bw*.65,'bar-holdout','HOLDOUT']].forEach(([v,off,cls,label])=>{const bh=ih*v,rect=svgEl('rect',{x:cx+off-bw/2,y:t+ih-bh,width:bw,height:Math.max(1,bh),rx:4,class:cls});const title=svgEl('title');title.textContent=`${row.label} · ${label}: ${pct(v)}`;rect.appendChild(title);svg.appendChild(rect)});let label=String(row.label).replace('mistral:7b-instruct-v0.3-q4_K_M','Mistral 7B').replace('ministral-3:3b','Ministral 3B').replace('qwen3.5:4b','Qwen 3.5').replace('llama3:latest','Llama 3');svgText(svg,cx,t+ih+18,label);});}
function stackedOutcomeChart(id,d,hout){const svg=chartSvg(id,900,260),w=900,l=110,r=25,t=45,barH=42,iw=w-l-r;const rows=[['DEV',d],['HOLDOUT',hout]],segments=[['accuracy','seg-correct','correta'],['ambiguous','seg-ambiguous','ambígua'],['wrong','seg-wrong','errada'],['invalid','seg-invalid','inválida']];rows.forEach(([label,obj],ri)=>{const y=t+ri*85;svgText(svg,l-15,y+barH/2+4,label,'end');let x=l;segments.forEach(([key,cls,name])=>{const v=Number(obj[key]||0),width=iw*v;if(width>0){const rect=svgEl('rect',{x,y,width,height:barH,class:cls});const title=svgEl('title');title.textContent=`${label} · ${name}: ${pct(v)}`;rect.appendChild(title);svg.appendChild(rect);if(width>55)svgText(svg,x+width/2,y+barH/2+4,pct(v));}x+=width;});});}
function horizontalPctChart(id,rows,valueKey){const h=Math.max(320,rows.length*34+45),svg=chartSvg(id,900,h),w=900,l=315,r=25,t=18,b=25,iw=w-l-r,gap=(h-t-b)/rows.length,bh=Math.min(10,gap*.32);for(let i=0;i<=4;i++){const x=l+iw*i/4;svg.appendChild(svgEl('line',{x1:x,y1:t,x2:x,y2:h-b,class:'gridline'}));svgText(svg,x,h-5,pct(i/4));}rows.forEach((row,i)=>{const cy=t+gap*(i+.5),label=String(row.label).length>43?String(row.label).slice(0,40)+'…':String(row.label);svgText(svg,l-10,cy+4,label,'end');[['dev','bar-dev',-bh*.7,'DEV'],['holdout','bar-holdout',bh*.7,'HOLDOUT']].forEach(([side,cls,off,sideLabel])=>{const v=Number(row[side]?.[valueKey]||0),rect=svgEl('rect',{x:l,y:cy+off-bh/2,width:iw*v,height:bh,rx:3,class:cls});const title=svgEl('title');title.textContent=`${row.label} · ${sideLabel}: ${pct(v)}`;rect.appendChild(title);svg.appendChild(rect);});});}
function renderGeneral(){if(!hasHoldout){$('generalContent').innerHTML='<div class="card empty">O comparativo GERAL é habilitado quando um run HOLDOUT é informado.</div>';return}const d=DATA.dev,h=DATA.holdout,dm=modelMap(d.models),hm=modelMap(h.models),dc=configMap(d.configs),hc=configMap(h.configs),modelChart=[...dm.keys()].map(k=>({label:k,dev:dm.get(k),holdout:hm.get(k)})),configChart=[...dc.keys()].map(k=>({label:dc.get(k).label,dev:dc.get(k),holdout:hc.get(k)})).filter(x=>x.holdout);$('generalContent').innerHTML=`<div class="grid"><div class="card span-6"><h2>DEV</h2><div class="sub">${d.question_counts.total} questões · ${d.question_counts.easy} easy · ${d.question_counts.medium} medium · ${d.question_counts.hard} hard</div><div class="kpis"><div class="kpi"><div class="v">${pct(d.overall.accuracy)}</div><div class="l">Accuracy</div></div><div class="kpi"><div class="v">${pct(d.overall.weighted_accuracy)}</div><div class="l">Acc. ponderada</div></div><div class="kpi"><div class="v">${pct(d.overall.precision)}</div><div class="l">Precisão</div></div><div class="kpi"><div class="v">${pct(d.overall.recall)}</div><div class="l">Recall</div></div></div></div><div class="card span-6"><h2>HOLDOUT</h2><div class="sub">${h.question_counts.total} questões · ${h.question_counts.easy} easy · ${h.question_counts.medium} medium · ${h.question_counts.hard} hard</div><div class="kpis"><div class="kpi"><div class="v">${pct(h.overall.accuracy)}</div><div class="l">Accuracy</div></div><div class="kpi"><div class="v">${pct(h.overall.weighted_accuracy)}</div><div class="l">Acc. ponderada</div></div><div class="kpi"><div class="v">${pct(h.overall.precision)}</div><div class="l">Precisão</div></div><div class="kpi"><div class="v">${pct(h.overall.recall)}</div><div class="l">Recall</div></div></div></div><div class="card span-6"><h2>Accuracy por modelo</h2><div class="sub">Comparação direta dos mesmos quatro geradores no DEV e HOLDOUT.</div><div class="chart" id="modelAccuracyChart"></div><div class="chart-legend"><span>DEV</span><span class="holdout">HOLDOUT</span></div></div><div class="card span-6"><h2>Distribuição das respostas</h2><div class="sub">Corretas, ambíguas, erradas e inválidas em cada split.</div><div class="chart" id="outcomeChart"></div><div class="chart-legend"><span class="correct">correta</span><span class="ambiguous">ambígua</span><span class="wrong">errada</span><span class="invalid">inválida</span></div></div><div class="card span-12"><h2>Retrievals · accuracy ponderada DEV × HOLDOUT</h2><div class="sub">Cada barra agrega os quatro LLMs; facilita ver estabilidade downstream das 10 configurações congeladas.</div><div class="chart" id="retrievalWeightedChart" style="height:390px"></div><div class="chart-legend"><span>DEV</span><span class="holdout">HOLDOUT</span></div></div><div class="card span-12"><h2>Modelos · DEV × HOLDOUT</h2><div class="sub">Mesmos geradores e mesmas 10 configurações; delta = HOLDOUT − DEV.</div><div class="table-wrap"><table><thead><tr><th>Modelo</th><th>DEV Acc.</th><th>HOLDOUT Acc.</th><th>Δ Acc.</th><th>DEV pond.</th><th>HOLDOUT pond.</th><th>Δ pond.</th><th>DEV Prec.</th><th>HOLDOUT Prec.</th><th>DEV Recall</th><th>HOLDOUT Recall</th></tr></thead><tbody>${[...dm.keys()].map(k=>{const a=dm.get(k),b=hm.get(k),da=delta(a.accuracy,b.accuracy),dw=delta(a.weighted_accuracy,b.weighted_accuracy);return `<tr><td><b>${esc(k)}</b></td><td>${pct(a.accuracy)}</td><td>${pct(b.accuracy)}</td><td class="${dclass(da)}">${dPct(da)}</td><td>${pct(a.weighted_accuracy)}</td><td>${pct(b.weighted_accuracy)}</td><td class="${dclass(dw)}">${dPct(dw)}</td><td>${pct(a.precision)}</td><td>${pct(b.precision)}</td><td>${pct(a.recall)}</td><td>${pct(b.recall)}</td></tr>`}).join('')}</tbody></table></div></div><div class="card span-12"><h2>Retrievals · impacto downstream</h2><div class="sub">Cada linha agrega os quatro LLMs. Comparação de estabilidade; não há nova seleção no HOLDOUT.</div><div class="table-wrap"><table><thead><tr><th>Configuração</th><th>DEV Acc.</th><th>HOLDOUT Acc.</th><th>Δ Acc.</th><th>DEV pond.</th><th>HOLDOUT pond.</th><th>Δ pond.</th></tr></thead><tbody>${[...dc.keys()].map(k=>{const a=dc.get(k),b=hc.get(k);if(!b)return '';const da=delta(a.accuracy,b.accuracy),dw=delta(a.weighted_accuracy,b.weighted_accuracy);return `<tr><td>${esc(a.label)}</td><td>${pct(a.accuracy)}</td><td>${pct(b.accuracy)}</td><td class="${dclass(da)}">${dPct(da)}</td><td>${pct(a.weighted_accuracy)}</td><td>${pct(b.weighted_accuracy)}</td><td class="${dclass(dw)}">${dPct(dw)}</td></tr>`}).join('')}</tbody></table></div></div></div>`;groupedPctChart('modelAccuracyChart',modelChart,'accuracy');stackedOutcomeChart('outcomeChart',d.overall,h.overall);horizontalPctChart('retrievalWeightedChart',configChart,'weighted_accuracy');}
$('btn-dev').onclick=()=>setView('dev');$('btn-holdout').onclick=()=>hasHoldout&&setView('holdout');$('btn-general').onclick=()=>hasHoldout&&setView('general');$('btn-holdout').disabled=!hasHoldout;$('btn-general').disabled=!hasHoldout;$('headerMeta').textContent=hasHoldout?`${DATA.dev.run_id} × ${DATA.holdout.run_id}`:`${DATA.dev.run_id} · HOLDOUT ainda não informado`;renderGeneral();setView(hasHoldout?'general':'dev');
</script></body></html>'''
    return template.replace("__COMPARISON__", safe_json(comparison)).replace("__DEV_B64__", dev_b64).replace("__HOLDOUT_B64__", holdout_b64)

def main() -> None:
    args = parse_args()
    config = ConfigManager.load()
    storage = RustFSManager.from_config(config.environment.rustfs)
    bucket = config.storage.benchmarks_bucket

    dev = load_mcq_run(storage, bucket, args.run_id)
    holdout: dict[str, Any] | None = None
    if args.holdout_run_id:
        holdout = load_mcq_run(storage, bucket, args.holdout_run_id)
        _validate_pair(dev, holdout)

    html = render_dual_html(dev=dev, holdout=holdout)
    if args.output is not None:
        output = args.output
    elif holdout is not None:
        output = Path("reports") / f"mcq_{dev['run_id']}__{holdout['run_id']}.html"
    else:
        output = Path("reports") / f"{dev['run_id']}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    if not args.no_upload:
        report_key = f"{dev['run_prefix']}/reports/{REPORT_KEY_NAME}"
        storage.upload_bytes(bucket=bucket, key=report_key, data=html.encode("utf-8"), content_type="text/html; charset=utf-8")
        print(f"rustfs: s3://{bucket}/{report_key}")

    print("=== MCQ DASHBOARD OK ===")
    print(f"dev: {dev['run_id']} | rows={len(dev['results'])}")
    if holdout is not None:
        print(f"holdout: {holdout['run_id']} | rows={len(holdout['results'])}")
        print("geral: enabled")
    else:
        print("holdout: ausente | seletor desabilitado")
        print("geral: disabled")
    print(f"local: {output.resolve()}")
    if args.open_browser:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
