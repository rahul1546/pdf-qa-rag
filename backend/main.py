from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pypdf import PdfReader
from pathlib import Path
from pydantic import BaseModel
import os
import re
import uuid
import json
import numpy as np
import faiss
from openai import OpenAI

import fitz 

load_dotenv()

BASE_DIR = Path.cwd()

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DOCS_DIR = BASE_DIR / "data" / "docs"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
INDEX_DIR = BASE_DIR / "data" / "indexes"
META_DIR = BASE_DIR / "data" / "meta"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


# OpenAI client (reuse)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("OPENAI_API_KEY is missing. Add it to backend/.env and restart the server.")
client = OpenAI(api_key=api_key) if api_key else None


def split_text_with_overlap(text: str, chunk_size: int = 1000, overlap: int = 150):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

        if end == n:
            break

    return chunks


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    if not api_key or not client:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to backend/.env and restart the server.")
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


#Evidence Extractor
def extract_evidence_snippets(question: str, retrieved: list[dict], max_snippets: int = 3):
    snippets = []
    for r in retrieved[:max_snippets]:
        text = (r.get("text") or "").strip()
        if not text:
            continue

        snippet = text[:280].replace("\n", " ").strip()
        if len(text) > 280:
            snippet += "…"

        out = {
            "page_number": r["page_number"],
            "chunk_id": r["chunk_id"],
            "score": r["score"],
            "snippet": snippet
        }

        # include doc_id when present (multi-doc)
        if "doc_id" in r:
            out["doc_id"] = r["doc_id"]

        snippets.append(out)

    return snippets


# Highlight helpers
def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _candidate_phrases(text: str) -> list[str]:
    """
    Create multiple search phrases to make highlighting robust.
    We try:
    - full text (trimmed)
    - first ~12 words
    - a few sliding windows of ~8 words
    """
    t = _normalize_space(text)
    if not t:
        return []

    words = t.split(" ")
    phrases = []

    # full
    phrases.append(t)

    # first 12 words
    if len(words) > 12:
        phrases.append(" ".join(words[:12]))

    # sliding windows of 8 words (up to 5 windows)
    win = 8
    if len(words) >= win:
        step = max(1, (len(words) - win) // 4) 
        count = 0
        for i in range(0, len(words) - win + 1, step):
            phrases.append(" ".join(words[i:i+win]))
            count += 1
            if count >= 5:
                break

    # de-dupe preserving order
    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _find_pdf_path(doc_id: str) -> str | None:
    """
    Locate the stored PDF for a doc_id.
    Prefer DOCS manifest stored_path; fallback to uploads folder scan.
    """
    manifest_path = DOCS_DIR / f"{doc_id}.json"
    if manifest_path.exists():
        try:
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
            stored_path = doc.get("stored_path")
            if stored_path and Path(stored_path).exists():
                return stored_path
        except Exception:
            pass

    if UPLOAD_DIR.exists():
        candidates = list(UPLOAD_DIR.glob(f"{doc_id}_*.pdf"))
        if candidates:
            return str(candidates[0])

    return None


def _rects_for_text(doc_id: str, page_number: int, text: str, max_rects: int = 20):
    pdf_path = _find_pdf_path(doc_id)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF file not found for this doc_id.")

    if page_number <= 0:
        raise HTTPException(status_code=400, detail="page_number must be >= 1.")

    phrases = _candidate_phrases(text)
    if not phrases:
        return []

    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open PDF for highlighting: {str(e)}")

    if page_number > pdf.page_count:
        pdf.close()
        raise HTTPException(status_code=400, detail=f"page_number out of range (1..{pdf.page_count}).")

    page = pdf.load_page(page_number - 1)
    page_rect = page.rect
    w, h = float(page_rect.width), float(page_rect.height)

    found = []
    seen = set()

    for phrase in phrases:
        try:
            rects = page.search_for(phrase)
        except Exception:
            rects = []

        for r in rects:
            # Normalize
            x0 = max(0.0, min(1.0, float(r.x0) / w))
            y0 = max(0.0, min(1.0, float(r.y0) / h))
            x1 = max(0.0, min(1.0, float(r.x1) / w))
            y1 = max(0.0, min(1.0, float(r.y1) / h))

            key = (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))
            if key in seen:
                continue
            seen.add(key)

            found.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
            if len(found) >= max_rects:
                break

        if len(found) >= max_rects:
            break

        if found:
            break

    pdf.close()
    return found


# Request Models
class ChunkRequest(BaseModel):
    chunk_size: int = 1000
    overlap: int = 150


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskManyRequest(BaseModel):
    doc_ids: list[str]
    question: str
    top_k_per_doc: int = 3 


#highlight request
class HighlightRequest(BaseModel):
    doc_id: str
    page_number: int
    text: str
    max_rects: int = 20

app = FastAPI(title="PDF QA (RAG) Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/pdf/{doc_id}")
def get_pdf(doc_id: str):
    manifest_path = DOCS_DIR / f"{doc_id}.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="doc_id not found. Upload first.")

    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_path = doc.get("stored_path")
    if not stored_path:
        raise HTTPException(status_code=500, detail="stored_path missing in manifest.")

    pdf_path = Path(stored_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    filename = doc.get("original_filename") or pdf_path.name
    headers = {"Content-Disposition": f'inline; filename="{filename}"'}

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        headers=headers
    )

from fastapi.responses import Response
import urllib.parse

@app.get("/pdf_highlight/{doc_id}")
def pdf_highlight(doc_id: str, page: int, text: str):
    pdf_path = _find_pdf_path(doc_id)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF file not found for this doc_id.")

    if page <= 0:
        raise HTTPException(status_code=400, detail="page must be >= 1")

    if not text.strip():
        # return original pdf if no text
        return get_pdf(doc_id)

    rects = _rects_for_text(doc_id, page, text, max_rects=50)

    # If nothing matched, return original PDF (so UI doesn’t break)
    if not rects:
        return get_pdf(doc_id)

    pdf = fitz.open(pdf_path)
    if page > pdf.page_count:
        pdf.close()
        raise HTTPException(status_code=400, detail=f"page out of range (1..{pdf.page_count})")

    p = pdf.load_page(page - 1)
    page_rect = p.rect
    w, h = float(page_rect.width), float(page_rect.height)

    # highlight annotations (yellow highlight)
    for r in rects:
        x0 = r["x0"] * w
        y0 = r["y0"] * h
        x1 = r["x1"] * w
        y1 = r["y1"] * h
        rr = fitz.Rect(x0, y0, x1, y1)
        annot = p.add_highlight_annot(rr)
        annot.update()

    pdf_bytes = pdf.tobytes() 
    pdf.close()

    return Response(content=pdf_bytes, media_type="application/pdf")


@app.get("/status/{doc_id}")
def doc_status(doc_id: str):
    manifest_path = DOCS_DIR / f"{doc_id}.json"
    chunks_path = CHUNKS_DIR / f"{doc_id}.json"
    index_path = INDEX_DIR / f"{doc_id}.faiss"
    meta_path = META_DIR / f"{doc_id}.json"

    pdf_path = _find_pdf_path(doc_id)

    has_pdf = pdf_path is not None
    has_manifest = manifest_path.exists()
    has_chunks = chunks_path.exists()
    has_index = index_path.exists() and meta_path.exists()

    num_pages = None
    original_filename = None
    total_extracted_chars = None

    if has_manifest:
        try:
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
            num_pages = doc.get("num_pages")
            original_filename = doc.get("original_filename")
            total_extracted_chars = doc.get("total_extracted_chars")
        except Exception:
            pass

    num_chunks = None
    if has_chunks:
        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            num_chunks = len(chunks)
        except Exception:
            pass

    return {
        "doc_id": doc_id,
        "has_pdf": has_pdf,
        "has_manifest": has_manifest,
        "has_chunks": has_chunks,
        "has_index": has_index,
        "pdf_path": pdf_path,  # for debugging
        "original_filename": original_filename,
        "num_pages": num_pages,
        "total_extracted_chars": total_extracted_chars,
        "num_chunks": num_chunks,
    }

@app.get("/page_text/{doc_id}/{page_number}")
def page_text(doc_id: str, page_number: int):
    manifest_path = DOCS_DIR / f"{doc_id}.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="doc_id not found. Upload first.")

    if page_number <= 0:
        raise HTTPException(status_code=400, detail="page_number must be >= 1")

    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = doc.get("pages") or []
    if not pages:
        raise HTTPException(status_code=404, detail="No pages found in manifest. Re-upload the PDF.")

    # pages are stored as [{page_number: 1, text: "..."} ...]
    if page_number > len(pages):
        raise HTTPException(status_code=400, detail=f"page_number out of range (1..{len(pages)})")

    page_obj = pages[page_number - 1]
    return {
        "doc_id": doc_id,
        "page_number": page_number,
        "text": page_obj.get("text", "") or ""
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    original_name = file.filename or "document.pdf"
    safe_name = original_name.replace(" ", "_")
    save_path = UPLOAD_DIR / f"{doc_id}_{safe_name}"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    save_path.write_bytes(content)

    try:
        reader = PdfReader(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    pages_preview = []
    pages_full = []
    total_chars = 0

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        char_count = len(text)
        total_chars += char_count

        pages_preview.append({
            "page_number": i + 1,
            "char_count": char_count,
            "text_preview": text[:400]
        })

        pages_full.append({
            "page_number": i + 1,
            "char_count": char_count,
            "text": text
        })

    doc_manifest = {
        "doc_id": doc_id,
        "original_filename": original_name,
        "stored_filename": save_path.name,
        "stored_path": str(save_path),
        "num_pages": len(reader.pages),
        "total_extracted_chars": total_chars,
        "pages": pages_full
    }

    (DOCS_DIR / f"{doc_id}.json").write_text(
        json.dumps(doc_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {
        "doc_id": doc_id,
        "filename": original_name,
        "saved_as": save_path.name,
        "num_pages": len(reader.pages),
        "total_extracted_chars": total_chars,
        "pages": pages_preview
    }


@app.post("/chunk/{doc_id}")
def chunk_document(doc_id: str, req: ChunkRequest):
    manifest_path = DOCS_DIR / f"{doc_id}.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="doc_id not found. Upload first.")

    doc = json.loads(manifest_path.read_text(encoding="utf-8"))

    chunk_size = req.chunk_size
    overlap = req.overlap

    all_chunks = []

    for page in doc["pages"]:
        page_num = page["page_number"]
        text = (page.get("text") or "").strip()
        if not text:
            continue

        page_chunks = split_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)

        for idx, ch in enumerate(page_chunks):
            clean = ch.replace("_", "").replace("\n", " ").strip()
            if len(clean) < 200:
                continue

            all_chunks.append({
                "chunk_id": f"{doc_id}_p{page_num}_c{idx+1}",
                "doc_id": doc_id,
                "page_number": page_num,
                "chunk_index_on_page": idx + 1,
                "text": ch,
                "char_count": len(ch)
            })

    chunks_path = CHUNKS_DIR / f"{doc_id}.json"
    chunks_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {
        "doc_id": doc_id,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_chunks": len(all_chunks),
        "saved_to": str(chunks_path),
        "sample_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "page_number": c["page_number"],
                "char_count": c["char_count"],
                "preview": c["text"][:250]
            }
            for c in all_chunks[:3]
        ]
    }


@app.post("/index/{doc_id}")
def build_faiss_index(doc_id: str):
    chunks_path = CHUNKS_DIR / f"{doc_id}.json"
    if not chunks_path.exists():
        raise HTTPException(status_code=404, detail="Chunks not found. Run /chunk/{doc_id} first.")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found in file.")

    texts = [c["text"] for c in chunks]

    embeddings = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(embed_texts(batch))

    X = np.array(embeddings, dtype="float32")
    dim = X.shape[1]

    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    index_path = INDEX_DIR / f"{doc_id}.faiss"
    meta_path = META_DIR / f"{doc_id}.json"

    faiss.write_index(index, str(index_path))

    meta = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": doc_id,
            "page_number": c["page_number"],
            "char_count": c["char_count"],
            "text": c["text"]
        }
        for c in chunks
    ]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "doc_id": doc_id,
        "num_vectors": len(chunks),
        "embedding_dim": dim,
        "index_path": str(index_path),
        "meta_path": str(meta_path)
    }


@app.post("/retrieve/{doc_id}")
def retrieve(doc_id: str, req: RetrieveRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    top_k = req.top_k
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0.")
    if top_k > 20:
        raise HTTPException(status_code=400, detail="top_k too large. Use <= 20 for now.")

    index_path = INDEX_DIR / f"{doc_id}.faiss"
    meta_path = META_DIR / f"{doc_id}.json"

    if not index_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=404, detail="Index not found. Run /index/{doc_id} first.")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    q_emb = embed_texts([query])[0]
    q = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx == -1:
            continue
        chunk_info = meta[idx]
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": chunk_info["chunk_id"],
            "doc_id": chunk_info.get("doc_id", doc_id),
            "page_number": chunk_info["page_number"],
            "char_count": chunk_info["char_count"],
            "preview": (chunk_info["text"][:300] + ("..." if len(chunk_info["text"]) > 300 else "")),
            "text": chunk_info["text"]
        })

    return {
        "doc_id": doc_id,
        "query": query,
        "top_k": top_k,
        "results": results
    }

# Helpers for multi-doc retrieve
def _load_index_and_meta(doc_id: str):
    index_path = INDEX_DIR / f"{doc_id}.faiss"
    meta_path = META_DIR / f"{doc_id}.json"
    if not index_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Index not found for doc_id={doc_id}. Run /index/{doc_id} first.")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, meta


def _retrieve_for_doc(doc_id: str, question: str, top_k: int) -> list[dict]:
    index, meta = _load_index_and_meta(doc_id)

    q_emb = embed_texts([question])[0]
    q = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k)

    retrieved = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        retrieved.append({
            "doc_id": doc_id,
            "score": float(score),
            "chunk_id": meta[idx]["chunk_id"],
            "page_number": meta[idx]["page_number"],
            "text": meta[idx]["text"]
        })

    return retrieved


# Highlight endpoint
@app.post("/highlight")
def highlight(req: HighlightRequest):
    doc_id = (req.doc_id or "").strip()
    text = (req.text or "").strip()
    page_number = req.page_number

    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required.")
    if not text:
        raise HTTPException(status_code=400, detail="text is required.")
    if req.max_rects <= 0 or req.max_rects > 200:
        raise HTTPException(status_code=400, detail="max_rects must be between 1 and 200.")

    rects = _rects_for_text(doc_id, page_number, text, max_rects=req.max_rects)

    return {
        "doc_id": doc_id,
        "page_number": page_number,
        "rects": rects,  # normalized rects in 0..1
        "matched": bool(rects)
    }



# Single-doc ask
@app.post("/ask/{doc_id}")
def ask(doc_id: str, req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    top_k = req.top_k
    if top_k <= 0 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20.")

    index_path = INDEX_DIR / f"{doc_id}.faiss"
    meta_path = META_DIR / f"{doc_id}.json"
    if not index_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=404, detail="Index not found. Run /index/{doc_id} first.")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    q_emb = embed_texts([question])[0]
    q = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k)

    retrieved = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        retrieved.append({
            "score": float(score),
            "chunk_id": meta[idx]["chunk_id"],
            "page_number": meta[idx]["page_number"],
            "text": meta[idx]["text"]
        })

    if not retrieved or retrieved[0]["score"] < 0.35:
        return {
            "doc_id": doc_id,
            "question": question,
            "answer": "I do not have enough information in the document.",
            "answer_citations": [],
            "retrieval_hits": retrieved,
            "retrieved_context": [
                {"page_number": r["page_number"], "chunk_id": r["chunk_id"], "score": r["score"]}
                for r in retrieved
            ],
            "evidence": []
        }

    evidence = extract_evidence_snippets(question, retrieved, max_snippets=3)

    context_blocks = []
    citations = []
    for r in retrieved:
        citations.append({
            "page_number": r["page_number"],
            "chunk_id": r["chunk_id"],
            "score": r["score"],
            "preview": (r["text"][:220] + ("..." if len(r["text"]) > 220 else "")),
            "text": r["text"]
        })
        context_blocks.append(f"[Page {r['page_number']}] {r['text']}")

    context = "\n\n".join(context_blocks)

    system_msg = (
        "You are a precise question-answering assistant. "
        "Use ONLY the provided context to answer. Do not use outside knowledge. "
        "Do not add interpretations or assumptions; only restate what the document says. "
        "If the answer is not explicitly supported by the context, reply exactly: "
        "'I do not have enough information in the document.' "
        "Every sentence MUST include at least one citation in the form [pX]."
    )

    user_msg = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer in English."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()
    used_pages = sorted(set(int(x) for x in re.findall(r"\[p(\d+)\]", answer)))

    return {
        "doc_id": doc_id,
        "question": question,
        "answer": answer,
        "answer_citations": used_pages,
        "retrieval_hits": citations,
        "evidence": evidence,
        "retrieved_context": [
            {"page_number": r["page_number"], "chunk_id": r["chunk_id"], "score": r["score"]}
            for r in retrieved
        ]
    }

# Multi-doc ask endpoint

@app.post("/ask_many")
def ask_many(req: AskManyRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    doc_ids = [d.strip() for d in (req.doc_ids or []) if d.strip()]
    doc_labels = {f"D{i+1}": doc_id for i, doc_id in enumerate(doc_ids)}
    label_by_doc = {doc_id: label for label, doc_id in doc_labels.items()}
    if not doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids cannot be empty.")

    if len(doc_ids) > 10:
        raise HTTPException(status_code=400, detail="Too many doc_ids. Use <= 10.")

    top_k_per_doc = req.top_k_per_doc
    if top_k_per_doc <= 0 or top_k_per_doc > 10:
        raise HTTPException(status_code=400, detail="top_k_per_doc must be between 1 and 10.")

    merged: list[dict] = []
    for doc_id in doc_ids:
        hits = _retrieve_for_doc(doc_id, question, top_k_per_doc)
        merged.extend(hits)

    merged.sort(key=lambda x: x["score"], reverse=True)

    if not merged or merged[0]["score"] < 0.35:
        return {
            "question": question,
            "answer": "I do not have enough information in the document.",
            "answer_citations": [],
            "answer_doc_citations": [],
            "retrieval_hits": merged,
            "retrieved_context": [
                {"doc_id": r["doc_id"], "page_number": r["page_number"], "chunk_id": r["chunk_id"], "score": r["score"]}
                for r in merged
            ],
            "evidence": []
        }

    evidence = extract_evidence_snippets(question, merged, max_snippets=5)

    context_blocks = []
    retrieval_hits = []
    for r in merged[: min(len(merged), 20)]:
        retrieval_hits.append({
            "doc_id": r["doc_id"],
            "page_number": r["page_number"],
            "chunk_id": r["chunk_id"],
            "score": r["score"],
            "preview": (r["text"][:220] + ("..." if len(r["text"]) > 220 else "")),
            "text": r["text"]
        })
        label = label_by_doc.get(r["doc_id"], "D?")
        context_blocks.append(f"[{label} p{r['page_number']}] {r['text']}")

    context = "\n\n".join(context_blocks)

    system_msg = (
        "You are a precise question-answering assistant. "
        "Use ONLY the provided context to answer. Do not use outside knowledge. "
        "Do not add interpretations or assumptions; only restate what the document says. "
        "If the answer is not explicitly supported by the context, reply exactly: "
        "'I do not have enough information in the document.' "
        "Every sentence MUST include at least one citation in the form [D# p#] (example: [D1 p3])."
    )

    user_msg = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer in English."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()

    used_pages = sorted(set(int(x) for x in re.findall(r"\[p(\d+)\]", answer)))

    seen = set()
    answer_doc_citations = []
    for r in merged:
        key = (r["doc_id"], r["page_number"])
        if key in seen:
            continue
        seen.add(key)
        answer_doc_citations.append({"doc_id": r["doc_id"], "page": r["page_number"]})
        if len(answer_doc_citations) >= 8:
            break

    return {
        "question": question,
        "answer": answer,
        "answer_citations": used_pages,
        "doc_labels": doc_labels,
        "answer_doc_citations": answer_doc_citations,
        "retrieval_hits": retrieval_hits,
        "evidence": evidence,
        "retrieved_context": [
            {"doc_id": r["doc_id"], "page_number": r["page_number"], "chunk_id": r["chunk_id"], "score": r["score"]}
            for r in merged
        ]
    }
