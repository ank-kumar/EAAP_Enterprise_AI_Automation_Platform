"""
EAAP - Enterprise AI Automation Platform
Backend API: Amazon Bedrock Knowledge Base retrieval + Claude generation.

Two modes:
  grounded  Answer strictly from retrieved passages, with citations.
  design    Use the Enterprise AI Automation Framework (retrieved from the KB) as the
            reference architecture and apply it to a new problem or domain. Framework
            elements are cited [n]; Claude's own domain proposals are marked [proposed].

Endpoints:
  POST /api/query         single JSON response (curl / scripting)
  POST /api/query/stream  server-sent events; trace first, then streamed answer

Run:
  uvicorn app:app --port 8000
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

REGION = os.getenv("EAAP_REGION") or os.getenv("AWS_REGION") or "ap-southeast-2"
KB_ID = os.getenv("EAAP_KB_ID", "")
MODEL_ID = os.getenv("EAAP_MODEL_ID", "au.anthropic.claude-haiku-4-5-20251001-v1:0")
TOP_K = int(os.getenv("EAAP_TOP_K", "5"))
MAX_TOKENS = int(os.getenv("EAAP_MAX_TOKENS", "1200"))

# Design mode: optionally a stronger model, and room for a longer answer.
DESIGN_MODEL_ID = os.getenv("EAAP_DESIGN_MODEL_ID") or MODEL_ID
DESIGN_MAX_TOKENS = int(os.getenv("EAAP_DESIGN_MAX_TOKENS", "2500"))
DESIGN_CHUNK_CAP = 8
FRAMEWORK_QUERY = os.getenv(
    "EAAP_FRAMEWORK_QUERY",
    "Enterprise AI Automation Framework stages reference architecture governance controls",
)

_boto_cfg = Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=120)
agent_rt = boto3.client("bedrock-agent-runtime", region_name=REGION, config=_boto_cfg)
bedrock_rt = boto3.client("bedrock-runtime", region_name=REGION, config=_boto_cfg)

GROUNDED_PROMPT = """You are EAAP, the Enterprise AI Automation Platform assistant.

You answer strictly from the CONTEXT passages supplied below, which were retrieved
from an enterprise knowledge base of architecture and solution documents.

Rules:
- Use only the context. Never add outside knowledge or invent detail.
- Cite the passage number inline as [1], [2] after each claim it supports.
- If the context does not contain the answer, say exactly what is missing and
  what document would be needed. Do not guess.
- Do not state a count of items unless the context states that count. If the
  context names a total but only details some of them, say which are detailed
  and which are not present in the retrieved passages.
- Answer as an enterprise architect briefing a technical stakeholder:
  precise, structured, no filler. Use short headings and bullets where the
  content is genuinely a list.
"""

DESIGN_PROMPT = """You are EAAP in DESIGN mode.

The CONTEXT passages come from an enterprise knowledge base containing the
Enterprise AI Automation Framework and solution implementations built on it.
Treat the framework in the context as the reference architecture.

Task: apply the framework to the problem in the QUESTION. The problem may be in a
domain the documents do not cover; that is expected.

Rules:
- Structure the design stage by stage using the framework's stage names exactly as
  they appear in the context. Do not rename, merge or invent stages. Cite [n]
  whenever you use a stage, pattern, control or principle taken from the context.
- Everything domain-specific that you contribute yourself (data sources, models,
  tools, thresholds, integrations, regulations) is your own reasoning: mark each
  such item with the tag [proposed]. Never cite a passage for something it does
  not say.
- Do not claim the design has been built, deployed or measured. Invent no metrics,
  results, baselines or projected improvements anywhere, including in Assumptions.
- Do not assume a jurisdiction or name specific regulations; if regulation matters,
  list it under What to validate with the client.
- Respect latency: if the domain has a real-time decision path, keep LLM reasoning
  out of it unless the budget clearly allows, and state which deterministic checks
  sit in that path instead.
- Call out explicitly where the framework's governance controls (confidence
  thresholds, human oversight, audit trail) matter most for this domain.
- End with two short sections: Assumptions, and What to validate with the client.
- Format: a short heading per framework stage, then bullets. No tables.
  Concise; an enterprise architect briefing a technical stakeholder.
"""


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    mode: str = "grounded"


app = FastAPI(title="EAAP API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Access logging (IST timestamps, real visitor IP via Cloudflare header)
# --------------------------------------------------------------------------

IST = timezone(timedelta(hours=5, minutes=30))
LOG = logging.getLogger("eaap.access")
LOG.setLevel(logging.INFO)
LOG.propagate = False
_fh = logging.FileHandler(BASE_DIR / "eaap_access.log")
_fh.setFormatter(logging.Formatter("%(message)s"))
LOG.addHandler(_fh)
LOG.addHandler(logging.StreamHandler())


def _now() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def _ip(request: Request) -> str:
    return (request.headers.get("cf-connecting-ip")
            or (request.client.host if request.client else "-"))


@app.middleware("http")
async def access_log(request: Request, call_next):
    response = await call_next(request)
    ua = request.headers.get("user-agent", "-")
    LOG.info(f"{_now()} | {_ip(request)} | {request.method} {request.url.path} "
             f"| {response.status_code} | {ua}")
    return response


def _logged(stream, ip: str, question: str, mode: str):
    """Log every query's outcome, including when the visitor closes the page mid-answer."""
    started, status = time.perf_counter(), "incomplete (client disconnected)"
    try:
        for chunk in stream:
            if chunk.startswith("event: done"):
                status = "completed"
            elif chunk.startswith("event: error"):
                try:
                    status = "error: " + json.loads(chunk.split("data: ", 1)[1]).get("error", "")[:120]
                except Exception:
                    status = "error"
            yield chunk
    finally:
        ms = round((time.perf_counter() - started) * 1000)
        LOG.info(f"{_now()} | {ip} | QUERY [{mode}] {question[:200]!r} | {status} | {ms} ms")


# --------------------------------------------------------------------------
# Retrieval helpers
# --------------------------------------------------------------------------

def _source_uri(result: Dict[str, Any]) -> str:
    loc = result.get("location") or {}
    for key in (
        "s3Location", "webLocation", "confluenceLocation", "sharePointLocation",
        "salesforceLocation", "customDocumentLocation", "kendraDocumentLocation",
        "sqlLocation",
    ):
        block = loc.get(key)
        if isinstance(block, dict):
            for field in ("uri", "url", "id", "query"):
                if block.get(field):
                    return str(block[field])
    meta = result.get("metadata") or {}
    return str(meta.get("x-amz-bedrock-kb-source-uri", "unknown"))


def _doc_label(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1] or uri


def _text(result: Dict[str, Any]) -> str:
    return ((result.get("content") or {}).get("text") or "").strip()


def _retrieve(query_text: str, top_k: int) -> List[Dict[str, Any]]:
    """Standard retrieval. Managed KBs may cap the requested result count."""
    args = {
        "knowledgeBaseId": KB_ID,
        "retrievalQuery": {"text": query_text},
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {"numberOfResults": top_k}
        },
    }
    try:
        return agent_rt.retrieve(**args).get("retrievalResults", [])
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in ("ValidationException", "BadRequestException"):
            raise
        args.pop("retrievalConfiguration")
        return agent_rt.retrieve(**args).get("retrievalResults", [])[:top_k]


def _to_chunks(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "rank": i,
            "score": round(float(r.get("score", 0.0)), 4),
            "uri": _source_uri(r),
            "document": _doc_label(_source_uri(r)),
            "text": _text(r),
        }
        for i, r in enumerate(results, start=1)
    ]


def _dedupe(results):
    """Drop cross-format duplicates: the same passage parsed from both .docx and .pdf."""
    seen, out = set(), []
    for r in sorted(results, key=lambda r: float(r.get("score", 0.0)), reverse=True):
        key = " ".join(_text(r).lower().split())[:150]
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _plan(question: str, top_k: int, mode: str) -> Dict[str, Any]:
    """Retrieve context and choose prompt/model for the requested mode."""
    t0 = time.perf_counter()
    if mode == "design":
        # Two retrievals in parallel: the problem itself, plus the framework reference,
        # so the architecture is anchored even when the question names a new domain.
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_q = ex.submit(_retrieve, question, top_k)
            f_fw = ex.submit(_retrieve, FRAMEWORK_QUERY, top_k)
            raw = f_q.result() + f_fw.result()
        results = _dedupe(raw)[:DESIGN_CHUNK_CAP]
        plan = {
            "op": "bedrock-agent-runtime:Retrieve x2 (question + framework reference)",
            "system": DESIGN_PROMPT,
            "model": DESIGN_MODEL_ID,
            "max_tokens": DESIGN_MAX_TOKENS,
            "temperature": 0.4,
        }
    else:
        results = _dedupe(_retrieve(question, top_k))
        plan = {
            "op": "bedrock-agent-runtime:Retrieve",
            "system": GROUNDED_PROMPT,
            "model": MODEL_ID,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2,
        }
    plan["chunks"] = _to_chunks(results)
    plan["retrieval_ms"] = round((time.perf_counter() - t0) * 1000)
    return plan


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(
        f"[{c['rank']}] source: {c['document']}\n{c['text']}" for c in chunks
    )


def _user_turn(question: str, context: str, mode: str) -> List[Dict[str, Any]]:
    if mode == "design":
        tail = ("Design the architecture using the framework in the context as the "
                "reference. Cite [n] for framework elements and mark your own "
                "proposals [proposed].")
    else:
        tail = "Answer from the context above and cite passages."
    return [{
        "role": "user",
        "content": [{"text": f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\n{tail}"}],
    }]


def _dedupe_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen, out = set(), []
    for c in chunks:
        if c["uri"] not in seen:
            seen.add(c["uri"])
            out.append({"document": c["document"], "uri": c["uri"]})
    return out


def _stages(question, top_k, mode, retrieval_ms, n_chunks, generation_ms, stop, op):
    return [
        {"name": "USER QUERY", "ms": 0, "detail": f"{len(question)} chars"},
        {"name": "KNOWLEDGE ROUTING", "ms": 0, "detail": f"{mode} · top_k {top_k}"},
        {"name": "BEDROCK KNOWLEDGE BASE", "ms": retrieval_ms,
         "detail": "Retrieve x2" if mode == "design" else "Retrieve"},
        {"name": "RETRIEVED CHUNKS", "ms": 0, "detail": f"{n_chunks} passages"},
        {"name": "CLAUDE", "ms": generation_ms, "detail": op},
        {"name": "GROUNDED RESPONSE", "ms": 0, "detail": stop or "complete"},
    ]


def _mode(raw: str | None) -> str:
    return "design" if (raw or "").lower() == "design" else "grounded"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "region": REGION,
        "knowledge_base_id": KB_ID or "NOT SET",
        "model_id": MODEL_ID,
        "design_model_id": DESIGN_MODEL_ID,
        "top_k": TOP_K,
        "streaming": True,
        "modes": ["grounded", "design"],
    }


@app.post("/api/query")
def query(req: QueryRequest):
    """Non-streaming. Same pipeline, single JSON payload."""
    question = (req.question or "").strip()
    mode = _mode(req.mode)
    if not question:
        return JSONResponse({"error": "Enter a question to run the pipeline."}, 400)
    if not KB_ID:
        return JSONResponse({"error": "EAAP_KB_ID is not set."}, 500)

    top_k = req.top_k or TOP_K
    t_start = time.perf_counter()
    try:
        plan = _plan(question, top_k, mode)
    except ClientError as exc:
        return JSONResponse(
            {"error": exc.response["Error"]["Message"],
             "stage": "BEDROCK KNOWLEDGE BASE",
             "code": exc.response["Error"]["Code"]}, 502)

    chunks = plan["chunks"]
    if not chunks:
        return JSONResponse(
            {"error": "No passages matched. Rephrase, or check the data source sync.",
             "stage": "RETRIEVED CHUNKS", "code": "EmptyResult"}, 502)

    try:
        t1 = time.perf_counter()
        resp = bedrock_rt.converse(
            modelId=plan["model"],
            system=[{"text": plan["system"]}],
            messages=_user_turn(question, _build_context(chunks), mode),
            inferenceConfig={"maxTokens": plan["max_tokens"],
                             "temperature": plan["temperature"]},
        )
        generation_ms = round((time.perf_counter() - t1) * 1000)
    except ClientError as exc:
        return JSONResponse(
            {"error": exc.response["Error"]["Message"], "stage": "CLAUDE",
             "code": exc.response["Error"]["Code"]}, 502)

    text = "".join(
        b.get("text", "") for b in resp["output"]["message"]["content"] if "text" in b
    ).strip()
    usage = resp.get("usage", {})
    total_ms = round((time.perf_counter() - t_start) * 1000)

    return {
        "answer": text,
        "sources": _dedupe_sources(chunks),
        "trace": {
            "mode": mode,
            "knowledge_base_id": KB_ID, "region": REGION, "model_id": plan["model"],
            "retrieval_operation": plan["op"],
            "generation_operation": "bedrock-runtime:Converse",
            "top_k": top_k, "chunks": chunks,
            "stages": _stages(question, top_k, mode, plan["retrieval_ms"], len(chunks),
                              generation_ms, resp.get("stopReason"), "Converse"),
            "tokens": {"input": usage.get("inputTokens"),
                       "output": usage.get("outputTokens")},
            "latency": {"retrieval_ms": plan["retrieval_ms"],
                        "generation_ms": generation_ms, "total_ms": total_ms},
            "status": "200 OK",
        },
    }


def _sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.post("/api/query/stream")
def query_stream(req: QueryRequest, request: Request):
    """Server-sent events: trace first, then the answer token by token."""
    question = (req.question or "").strip()
    mode = _mode(req.mode)
    top_k = req.top_k or TOP_K

    def gen() -> Generator[str, None, None]:
        if not question:
            yield _sse("error", {"error": "Enter a question to run the pipeline."})
            return
        if not KB_ID:
            yield _sse("error", {"error": "EAAP_KB_ID is not set."})
            return

        t_start = time.perf_counter()

        try:
            plan = _plan(question, top_k, mode)
        except ClientError as exc:
            yield _sse("error", {
                "error": exc.response["Error"]["Message"],
                "stage": "BEDROCK KNOWLEDGE BASE",
                "code": exc.response["Error"]["Code"]})
            return

        chunks = plan["chunks"]
        if not chunks:
            yield _sse("error", {
                "error": "No passages matched. Rephrase, or check the data source sync.",
                "stage": "RETRIEVED CHUNKS", "code": "EmptyResult"})
            return

        yield _sse("retrieval", {
            "mode": mode,
            "chunks": chunks,
            "sources": _dedupe_sources(chunks),
            "knowledge_base_id": KB_ID,
            "region": REGION,
            "model_id": plan["model"],
            "retrieval_operation": plan["op"],
            "generation_operation": "bedrock-runtime:ConverseStream",
            "top_k": top_k,
            "question_chars": len(question),
            "retrieval_ms": plan["retrieval_ms"],
        })

        stop_reason, usage, first_token_ms = None, {}, None
        t1 = time.perf_counter()
        try:
            stream = bedrock_rt.converse_stream(
                modelId=plan["model"],
                system=[{"text": plan["system"]}],
                messages=_user_turn(question, _build_context(chunks), mode),
                inferenceConfig={"maxTokens": plan["max_tokens"],
                                 "temperature": plan["temperature"]},
            )["stream"]
            for event in stream:
                if "contentBlockDelta" in event:
                    piece = event["contentBlockDelta"]["delta"].get("text", "")
                    if piece:
                        if first_token_ms is None:
                            first_token_ms = round((time.perf_counter() - t1) * 1000)
                        yield _sse("delta", {"text": piece})
                elif "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason")
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {}) or {}
        except ClientError as exc:
            yield _sse("error", {
                "error": exc.response["Error"]["Message"], "stage": "CLAUDE",
                "code": exc.response["Error"]["Code"]})
            return

        generation_ms = round((time.perf_counter() - t1) * 1000)
        total_ms = round((time.perf_counter() - t_start) * 1000)

        yield _sse("done", {
            "trace": {
                "mode": mode,
                "knowledge_base_id": KB_ID, "region": REGION, "model_id": plan["model"],
                "retrieval_operation": plan["op"],
                "generation_operation": "bedrock-runtime:ConverseStream",
                "top_k": top_k, "chunks": chunks,
                "stages": _stages(question, top_k, mode, plan["retrieval_ms"],
                                  len(chunks), generation_ms, stop_reason,
                                  "ConverseStream"),
                "tokens": {"input": usage.get("inputTokens"),
                           "output": usage.get("outputTokens")},
                "latency": {"retrieval_ms": plan["retrieval_ms"],
                            "generation_ms": generation_ms,
                            "first_token_ms": first_token_ms,
                            "total_ms": total_ms},
                "status": "200 OK",
            }
        })

    return StreamingResponse(
        _logged(gen(), _ip(request), question, mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
