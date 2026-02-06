"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  uploadPDF,
  chunkDoc,
  indexDoc,
  askDoc,
  askManyDocs,
  type AskResponse,
} from "./lib/api";
import { highlightText, pickHighlightText } from "./lib/api";

const PdfPreview = dynamic(() => import("./components/PdfPreview"), {
  ssr: false,
});

type Stage =
  | "idle"
  | "uploading"
  | "chunking"
  | "indexing"
  | "ready"
  | "asking"
  | "error";

type EvidenceItem = {
  doc_id?: string;
  page_number: number;
  chunk_id: string;
  score: number;
  snippet: string;
};

type HistoryItem = {
  id: string;
  doc_id: string;
  question: string;
  answer: string;
  answer_citations: number[];
  evidence?: EvidenceItem[];
  doc_labels?: Record<string, string>;
  created_at: number;
};

type DocItem = {
  doc_id: string;
  filename: string;
  saved_as?: string;
  num_pages?: number;
  uploaded_at: number;
};

type DocStatus = {
  doc_id: string;
  has_pdf: boolean;
  has_manifest: boolean;
  has_chunks: boolean;
  has_index: boolean;
  num_pages?: number | null;
  num_chunks?: number | null;
};

type CitationCtx = {
  answer?: string; 
  evidence?: EvidenceItem[];
  doc_labels?: Record<string, string>;
};


export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [docId, setDocId] = useState<string>("");

  const [stage, setStage] = useState<Stage>("idle");
  const [status, setStatus] = useState<string>("Idle");
  const [error, setError] = useState<string>("");

  const [chunkSize, setChunkSize] = useState<number>(1000);
  const [overlap, setOverlap] = useState<number>(150);

  const [selectedPage, setSelectedPage] = useState<number>(1);
  const pdfUrl = docId
    ? `${process.env.NEXT_PUBLIC_API_BASE_URL}/pdf/${docId}`
    : "";

  const [question, setQuestion] = useState<string>("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [showDebug, setShowDebug] = useState<boolean>(false);

  const [activeSourcePage, setActiveSourcePage] = useState<number | null>(null);
  const [docStatusMap, setDocStatusMap] = useState<Record<string, DocStatus>>(
    {}
  );

  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [docs, setDocs] = useState<DocItem[]>([]);

  // highlight state
  const [highlightRects, setHighlightRects] = useState<any[]>([]);
  const [highlightDocId, setHighlightDocId] = useState<string>("");
  const [highlightPage, setHighlightPage] = useState<number>(1);

  // multi-select docs
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  // chat scroll
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // split-pane sizing (chat vs pdf)
  const splitWrapRef = useRef<HTMLDivElement | null>(null);
  const [chatPanePct, setChatPanePct] = useState<number>(58);
  const draggingRef = useRef(false);

  // Citation hover UI
  const [hoveredCitation, setHoveredCitation] = useState<{
    key: string;
    snippet: string;
    page: number;
    citedDocId?: string;
    docLabel?: string;
  } | null>(null);

  // Setup dropdown state
  const [showSetup, setShowSetup] = useState<boolean>(false);

  const effectiveDocIds = useMemo(() => {
    if (selectedDocIds.length > 0) return selectedDocIds;
    return docId ? [docId] : [];
  }, [selectedDocIds, docId]);

  const canAsk = useMemo(() => {
    if (stage !== "ready") return false;
    if (effectiveDocIds.length === 0) return false;

    const anyNotIndexed = effectiveDocIds.some((id) => {
      const s = docStatusMap[id];
      return s ? !s.has_index : false;
    });

    return !anyNotIndexed;
  }, [stage, effectiveDocIds, docStatusMap]);

  // localStorage: history 
  useEffect(() => {
    try {
      const raw = localStorage.getItem("pdfqa_history");
      if (raw) setHistory(JSON.parse(raw));
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("pdfqa_history", JSON.stringify(history));
    } catch {}
  }, [history]);

  // localStorage: docs 
  useEffect(() => {
    try {
      const raw = localStorage.getItem("pdfqa_docs");
      if (raw) setDocs(JSON.parse(raw));
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("pdfqa_docs", JSON.stringify(docs));
    } catch {}
  }, [docs]);

  // after refresh: if we already have docs, allow asking
  useEffect(() => {
    if (docs.length > 0 && stage === "idle") {
      setStage("ready");
      setStatus("Ready ✅ Select a document (or just ask if one is open).");
    }
  }, [docs, stage]);

  // if no doc open, auto-open most recent
  useEffect(() => {
    if (!docId && docs.length > 0) {
      const mostRecent = docs[0];
      setDocId(mostRecent.doc_id);
      setSelectedPage(1);
      setStage("ready");
      setStatus(`Selected: ${mostRecent.filename}`);

      setHighlightRects([]);
      setHighlightDocId(mostRecent.doc_id);
      setHighlightPage(1);
    }
  }, [docs, docId]);

  // load statuses
  useEffect(() => {
    let cancelled = false;

    async function loadStatuses() {
      const base = process.env.NEXT_PUBLIC_API_BASE_URL;
      if (!base) return;

      const ids = docs.map((d) => d.doc_id).filter(Boolean);
      if (!ids.length) return;

      try {
        const results = await Promise.all(
          ids.map(async (id) => {
            const res = await fetch(`${base}/status/${id}`);
            if (!res.ok) throw new Error(`status ${id} failed`);
            return (await res.json()) as DocStatus;
          })
        );

        if (cancelled) return;

        setDocStatusMap((prev) => {
          const next = { ...prev };
          for (const s of results) next[s.doc_id] = s;
          return next;
        });
      } catch {
        // ignore
      }
    }

    loadStatuses();
    return () => {
      cancelled = true;
    };
  }, [docs]);

  // active doc
  const activeDoc = useMemo(() => {
    if (!docId) return null;
    return docs.find((d) => d.doc_id === docId) ?? null;
  }, [docs, docId]);

  // chat for active doc
  const activeDocChat = useMemo(() => {
    if (!docId) return [];
    return history
      .filter((h) => h.doc_id === docId)
      .sort((a, b) => a.created_at - b.created_at);
  }, [history, docId]);

  // auto-scroll chat when new message arrives
  useEffect(() => {
    if (!docId) return;
    requestAnimationFrame(() => {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    });
  }, [activeDocChat.length, docId, stage]);

  function selectDoc(d: DocItem) {
    setDocId(d.doc_id);
    setSelectedPage(1);
    setQuestion("");
    setResult(null);
    setShowDebug(false);
    setActiveSourcePage(null);
    setError("");
    setStage("ready");
    setStatus(`Selected: ${d.filename}`);

    setHighlightRects([]);
    setHighlightDocId(d.doc_id);
    setHighlightPage(1);
  }

  function toggleSelectDoc(id: string) {
    setSelectedDocIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      return [...prev, id];
    });
  }

  function statusBadge(s?: DocStatus) {
    if (!s) {
      return (
        <span className="text-[11px] rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-zinc-300">
          Checking…
        </span>
      );
    }

    if (s.has_index) {
      return (
        <span className="text-[11px] rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-emerald-300">
          Ready
        </span>
      );
    }
    if (s.has_chunks) {
      return (
        <span className="text-[11px] rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-amber-300">
          Chunked
        </span>
      );
    }
    if (s.has_pdf || s.has_manifest) {
      return (
        <span className="text-[11px] rounded-full border border-sky-500/30 bg-sky-500/10 px-2 py-0.5 text-sky-300">
          Uploaded
        </span>
      );
    }
    return (
      <span className="text-[11px] rounded-full border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-rose-300">
        Missing
      </span>
    );
  }

  function clearChatForThisDoc() {
    if (!docId) return;
    setHistory((prev) => prev.filter((h) => h.doc_id !== docId));
    setResult(null);
    setShowDebug(false);
    setHighlightRects([]);
    setActiveSourcePage(null);
  }

  async function copyToClipboard(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setStatus("Copied ✅");
      setTimeout(() => setStatus("Ready ✅"), 900);
    } catch {
      // ignore
    }
  }

  // split-pane drag handlers
  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!draggingRef.current) return;
      const wrap = splitWrapRef.current;
      if (!wrap) return;

      const rect = wrap.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const pct = (x / rect.width) * 100;

      const clamped = Math.max(38, Math.min(76, pct));
      setChatPanePct(clamped);
    }

    function onUp() {
      draggingRef.current = false;
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  async function refreshOneStatus(id: string) {
    const base = process.env.NEXT_PUBLIC_API_BASE_URL;
    if (!base || !id) return;

    try {
      const res = await fetch(`${base}/status/${id}`);
      if (!res.ok) return;
      const s = (await res.json()) as DocStatus;

      setDocStatusMap((prev) => ({ ...prev, [s.doc_id]: s }));
    } catch {
      // ignore
    }
  }


  // navigation + highlight
async function goToCitation(
  page: number,
  citedDocId?: string,
  docLabel?: string,
  ctx?: CitationCtx
) {
  const targetDocId = citedDocId || docId;
  if (!targetDocId) return;

  if (targetDocId !== docId) setDocId(targetDocId);

  setSelectedPage(page);
  setActiveSourcePage(page);

  const ctxResult: AskResponse | null = ctx?.answer
    ? ({
        doc_id: targetDocId,
        question: "",
        answer: ctx.answer,
        evidence: ctx.evidence ?? [],
        doc_labels: ctx.doc_labels ?? undefined,
      } as any)
    : result;

  const textToHighlight = pickHighlightText(
    ctxResult,
    page,
    targetDocId,
    docLabel
  );

  if (!textToHighlight) {
    setHighlightRects([]);
    setHighlightDocId(targetDocId);
    setHighlightPage(page);
    return;
  }

  try {
    const h = await highlightText({
      doc_id: targetDocId,
      page_number: page,
      text: textToHighlight,
      max_rects: 20,
    });

    setHighlightDocId(targetDocId);
    setHighlightPage(page);
    setHighlightRects(h.rects ?? []);
  } catch {
    setHighlightDocId(targetDocId);
    setHighlightPage(page);
    setHighlightRects([]);
  }
}


  function getCitationSnippet(
    ctx: CitationCtx | undefined,
    page: number,
    citedDocId?: string
  ) {
    const ev = (ctx?.evidence ?? []).find((e) => {
      if (e.page_number !== page) return false;
      if (!citedDocId) return true;
      if (!e.doc_id) return true;
      return e.doc_id === citedDocId;
    });
    return ev?.snippet ?? "";
  }

  function renderAnswerWithClickableCitations(text: string, ctx?: CitationCtx) {
    const parts = text.split(/(\[(?:D\d+\s*)?p\d+\])/g);

    return (
      <>
        {parts.map((part, i) => {
          let m = part.match(/^\[p(\d+)\]$/);
          if (m) {
            const p = Number(m[1]);
            const snippet = getCitationSnippet(ctx, p, docId);
            const key = `p${p}-${i}`;

            return (
              <span key={key} className="relative inline-block">
                <button
                  onClick={() =>
                    goToCitation(p, undefined, undefined, {
                      answer: text,
                      evidence: ctx?.evidence ?? [],
                      doc_labels: ctx?.doc_labels,
                    })
                  }
                  onMouseEnter={() =>
                    setHoveredCitation({
                      key,
                      snippet,
                      page: p,
                      citedDocId: docId || undefined,
                    })
                  }
                  onMouseLeave={() =>
                    setHoveredCitation((prev) => (prev?.key === key ? null : prev))
                  }
                  className="mx-1 inline-flex items-center rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-zinc-200 hover:bg-white/10"
                  type="button"
                >
                  p{p}
                </button>

                {hoveredCitation?.key === key && snippet && (
                  <div className="absolute z-[999] mt-2 w-[320px] rounded-xl border border-white/10 bg-[#0b0f14] p-3 shadow-xl">
                    <div className="text-[11px] uppercase tracking-wide text-zinc-400">
                      Citation • Page {p}
                    </div>
                    <div className="mt-2 text-sm leading-6 text-zinc-200 line-clamp-6">
                      {snippet}
                    </div>
                    
                  </div>
                )}
              </span>
            );
          }

          m = part.match(/^\[(D\d+)\s*p(\d+)\]$/);
          if (m) {
            const label = m[1];
            const p = Number(m[2]);
            const citedDocId = ctx?.doc_labels?.[label];
            const snippet = getCitationSnippet(ctx, p, citedDocId);
            const key = `${label}-p${p}-${i}`;

            return (
              <span key={key} className="relative inline-block">
                <button
                  onClick={() =>
                    goToCitation(p, citedDocId, label, {
                      answer: text,
                      evidence: ctx?.evidence ?? [],
                      doc_labels: ctx?.doc_labels,
                    })
                  }
                  onMouseEnter={() =>
                    setHoveredCitation({
                      key,
                      snippet,
                      page: p,
                      citedDocId,
                      docLabel: label,
                    })
                  }
                  onMouseLeave={() =>
                    setHoveredCitation((prev) => (prev?.key === key ? null : prev))
                  }
                  className="mx-1 inline-flex items-center rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-zinc-200 hover:bg-white/10"
                  type="button"
                  title={citedDocId ? `Open ${label}` : label}
                >
                  {label} p{p}
                </button>

                {hoveredCitation?.key === key && snippet && (
                  <div className="absolute z-[999] mt-2 w-[340px] rounded-xl border border-white/10 bg-[#0b0f14] p-3 shadow-xl">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-[11px] uppercase tracking-wide text-zinc-400">
                        Citation • {label} • Page {p}
                      </div>
                      {citedDocId ? (
                        <div className="text-[10px] text-zinc-500 font-mono truncate max-w-[140px]">
                          {citedDocId}
                        </div>
                      ) : null}
                    </div>

                    <div className="mt-2 text-sm leading-6 text-zinc-200 line-clamp-6">
                      {snippet}
                    </div>

            
                  </div>
                )}
              </span>
            );
          }

          return <span key={`${part}-${i}`}>{part}</span>;
        })}
      </>
    );
  }

  async function handleUploadPipeline() {
    if (!file) return;

    setError("");
    setResult(null);
    setShowDebug(false);
    setActiveSourcePage(null);

    setHighlightRects([]);
    setHighlightDocId(docId);
    setHighlightPage(1);

    try {
      setStage("uploading");
      setStatus("Uploading PDF…");
      const up = await uploadPDF(file);
      setDocId(up.doc_id);
      setSelectedPage(1);

      setDocs((prev) => {
        const item: DocItem = {
          doc_id: up.doc_id,
          filename: up.filename ?? file.name,
          saved_as: up.saved_as,
          num_pages: up.num_pages,
          uploaded_at: Date.now(),
        };
        const next = [item, ...prev.filter((d) => d.doc_id !== up.doc_id)];
        return next.slice(0, 30);
      });

      setStage("chunking");
      setStatus("Chunking document…");
      await chunkDoc(up.doc_id, chunkSize, overlap);

      setStage("indexing");
      setStatus("Building FAISS index…");
      await indexDoc(up.doc_id);
      await refreshOneStatus(up.doc_id);
      setStage("ready");
      setStatus("Ready!! You can ask questions now.");
      setShowSetup(false); 
    } catch (e: any) {
      setStage("error");
      setStatus("Error");
      setError(e?.message ?? String(e));
    }
  }

  async function handleAsk() {
    if (!canAsk || !question.trim()) return;

    setError("");
    setResult(null);
    setShowDebug(false);
    setActiveSourcePage(null);

    setHighlightRects([]);
    setHighlightDocId(docId);
    setHighlightPage(selectedPage);

    try {
      setStage("asking");
      setStatus("Thinking…");

      const q = question.trim();
      let res: AskResponse;

      if (selectedDocIds.length > 0) {
        res = await askManyDocs(selectedDocIds, q, 3);
      } else {
        res = await askDoc(docId, q, 5);
      }

      setResult(res);

      const persistDocId = docId || selectedDocIds[0] || "";

      setHistory((prev) => {
        const item: HistoryItem = {
          id: crypto.randomUUID(),
          doc_id: persistDocId,
          question: q,
          answer: res.answer,
          answer_citations: res.answer_citations ?? [],
          evidence: (res as any).evidence ?? [],
          doc_labels: (res as any).doc_labels ?? undefined,
          created_at: Date.now(),
        };

        const next = [...prev, item].slice(-250);
        return next;
      });

      setQuestion("");
      setStage("ready");
      setStatus("Done ✅");
    } catch (e: any) {
      setStage("error");
      setStatus("Error");
      setError(e?.message ?? String(e));
    }
  }

  function onQuestionKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canAsk && question.trim() && stage !== "asking") {
        handleAsk();
      }
    }
  }

  function TypingIndicator() {
    return (
      <div className="flex justify-start gap-3">
        <div className="h-9 w-9 shrink-0 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-sm font-semibold text-zinc-200">
          A
        </div>
        <div className="max-w-[70%] rounded-2xl rounded-bl-sm bg-white/5 border border-white/10 px-4 py-3">
          <div className="flex items-center gap-2 text-sm text-zinc-300">
            <span>Thinking</span>
            <span className="inline-flex gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-bounce [animation-delay:-0.2s]" />
              <span className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-bounce [animation-delay:-0.1s]" />
              <span className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-bounce" />
            </span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <main className="h-screen overflow-hidden bg-[#0b0f14] text-zinc-100">
      <div className="pointer-events-none fixed inset-0 opacity-50">
        <div className="absolute -top-24 left-1/3 h-80 w-80 rounded-full bg-sky-500/10 blur-3xl" />
        <div className="absolute top-1/2 -right-24 h-80 w-80 rounded-full bg-emerald-500/10 blur-3xl" />
      </div>

      <div className="relative h-full w-full px-3 sm:px-4 lg:px-6 py-4 flex flex-col gap-3">
        <header className="shrink-0 flex items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">
              PDF Q&A <span className="text-zinc-400 font-medium">(Upload & Ask)</span>
            </h1>
          </div>

          <div className="text-xs rounded-full border border-white/10 bg-white/5 px-3 py-1 text-zinc-300">
            {stage.toUpperCase()}
          </div>
        </header>

        <div className="flex-1 min-h-0">
          <div className="grid h-full min-h-0 gap-4 lg:grid-cols-[330px_minmax(0,1fr)]">
            {/* LEFT: Documents */}
            <aside className="rounded-2xl border border-white/10 bg-[#0f1520] shadow-sm overflow-hidden flex flex-col min-h-0">
              <div className="shrink-0 p-4 pb-3 flex items-center justify-between">
                <div className="font-medium">Documents</div>
                <button
                  type="button"
                  onClick={() => {
                    setDocs([]);
                    setSelectedDocIds([]);
                    setDocId("");
                    setQuestion("");
                    setResult(null);
                    setSelectedPage(1);
                    setActiveSourcePage(null);
                    setShowDebug(false);
                    setStage("idle");
                    setStatus("Idle");
                    setError("");

                    setHighlightRects([]);
                    setHighlightDocId("");
                    setHighlightPage(1);
                  }}
                  className="text-xs rounded-full border border-white/10 bg-white/5 px-3 py-1 text-zinc-200 hover:bg-white/10"
                >
                  Clear
                </button>
              </div>

              <div className="min-h-0 overflow-y-auto px-4 pb-4 pr-5">
                <div className="grid gap-2">
                  {docs.length ? (
                    docs.map((d) => {
                      const active = d.doc_id === docId;
                      const s = docStatusMap[d.doc_id];
                      const isSelected = selectedDocIds.includes(d.doc_id);

                      const pagesLabel =
                        s?.num_pages != null
                          ? `${s.num_pages} pages`
                          : d.num_pages
                          ? `${d.num_pages} pages`
                          : "PDF";

                      const chunksLabel =
                        s?.num_chunks != null ? ` • ${s.num_chunks} chunks` : "";

                      return (
                        <div
                          key={d.doc_id}
                          className={`w-full rounded-xl border p-3 transition ${
                            active
                              ? "bg-white/5 border-white/15"
                              : "bg-transparent border-white/10 hover:bg-white/5"
                          }`}
                        >
                          <button
                            type="button"
                            onClick={() => selectDoc(d)}
                            className="w-full text-left"
                          >
                            <div className="flex items-start justify-between gap-2 min-w-0">
                              <div className="min-w-0 text-sm font-medium line-clamp-1">
                                {d.filename}
                              </div>
                              <div className="shrink-0">{statusBadge(s)}</div>
                            </div>

                            <div className="mt-1 text-xs text-zinc-400">
                              {pagesLabel}
                              {chunksLabel} •{" "}
                              {new Date(d.uploaded_at).toLocaleDateString()}
                            </div>

                            <div className="mt-2 font-mono text-[10px] text-zinc-500 break-all">
                              {d.doc_id}
                            </div>
                          </button>

                          <div className="mt-3 flex gap-2">
                            <button
                              type="button"
                              onClick={() => selectDoc(d)}
                              className="text-xs rounded-full bg-white text-black px-3 py-1 hover:opacity-90"
                            >
                              Open
                            </button>

                            <button
                              type="button"
                              onClick={() => toggleSelectDoc(d.doc_id)}
                              className={`text-xs rounded-full border px-3 py-1 ${
                                isSelected
                                  ? "bg-sky-500/20 text-sky-200 border-sky-500/30"
                                  : "text-zinc-200 bg-white/5 border-white/10 hover:bg-white/10"
                              }`}
                            >
                              {isSelected ? "Selected" : "Select"}
                            </button>

                            <button
                              type="button"
                              onClick={() => {
                                setDocs((prev) =>
                                  prev.filter((x) => x.doc_id !== d.doc_id)
                                );
                                setSelectedDocIds((prev) =>
                                  prev.filter((id) => id !== d.doc_id)
                                );

                                if (docId === d.doc_id) {
                                  setDocId("");
                                  setResult(null);
                                  setQuestion("");
                                  setStage("idle");
                                  setStatus("Idle");
                                  setSelectedPage(1);
                                  setActiveSourcePage(null);
                                  setShowDebug(false);
                                  setError("");

                                  setHighlightRects([]);
                                  setHighlightDocId("");
                                  setHighlightPage(1);
                                }
                              }}
                              className="text-xs rounded-full border border-white/10 bg-white/5 px-3 py-1 text-zinc-200 hover:bg-white/10"
                            >
                              Remove
                            </button>
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-sm text-zinc-400">
                      No documents yet. Upload a PDF to start.
                    </div>
                  )}
                </div>
              </div>
            </aside>

            {/* RIGHT: Split Pane (Chat | PDF) */}
            <div ref={splitWrapRef} className="min-w-0 h-full min-h-0 rounded-2xl">
              <div className="h-full min-h-0 flex gap-0">
                {/* Chat pane */}
                <section
                  className="min-w-0 rounded-2xl border border-white/10 bg-[#0f1520] shadow-sm overflow-hidden flex flex-col min-h-0"
                  style={{ width: `${chatPanePct}%` }}
                >
                  {/* Chat header */}
                  <div className="shrink-0 border-b border-white/10 px-4 py-3">
                    <div className="flex items-center justify-between gap-3">
                      {/* LEFT: selected doc pill replaces name/id block */}
                      <div className="min-w-0 flex items-center gap-2">
                        <span className="text-[11px] rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-zinc-300">
                          {docId ? "Selected" : "No doc"}
                        </span>

                        <div className="min-w-0 truncate text-xs text-zinc-300">
                          {activeDoc?.filename ?? "Select or upload a document"}
                        </div>
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        {/* Setup dropdown toggle */}
                        <button
                          type="button"
                          onClick={() => setShowSetup((v) => !v)}
                          className="text-xs rounded-full border border-white/10 bg-white/5 px-3 py-1 text-zinc-200 hover:bg-white/10"
                        >
                          {showSetup ? "Hide" : "Add File"}
                        </button>

                        <button
                          type="button"
                          onClick={clearChatForThisDoc}
                          disabled={!docId}
                          className="text-xs rounded-full border border-white/10 bg-white/5 px-3 py-1 text-zinc-200 hover:bg-white/10 disabled:opacity-50"
                        >
                          Clear chat
                        </button>
                      </div>
                    </div>

                    {/* Collapsible setup panel */}
                    {showSetup && (
                      <div className="mt-3 rounded-2xl border border-white/10 bg-white/[0.03] p-3">
                        {/* One-row controls */}
                        <div className="flex items-center gap-3">
                          {/* Browse + filename underneath */}
                          <div className="min-w-[80px] max-w-[100px]">
                            <label className="inline-flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-zinc-200 hover:bg-white/10">
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 24 24"
                                fill="none"
                                className="opacity-80"
                              >
                                <path
                                  d="M21 11.5l-8.8 8.8a6 6 0 01-8.5-8.5l9.2-9.2a4.5 4.5 0 016.4 6.4l-9.2 9.2a3 3 0 01-4.2-4.2l8.8-8.8"
                                  stroke="currentColor"
                                  strokeWidth="2"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                              <span>Browse</span>
                              <input
                                type="file"
                                accept="application/pdf"
                                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                                className="hidden"
                              />
                            </label>

                            {/* filename under Browse */}
                            <div className="mt-1 text-[11px] text-zinc-400 truncate">
                              {file ? file.name : "No file selected"}
                            </div>
                          </div>

                          {/* Chunk */}
                          <label className="flex items-center -mt-5 gap-2 text-xs text-zinc-400">
                            Chunk
                            <input
                              type="number"
                              value={chunkSize}
                              onChange={(e) => setChunkSize(Number(e.target.value))}
                              className="w-24 rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-zinc-100"
                            />
                          </label>

                          {/* Overlap */}
                          <label className="flex items-center -mt-5 gap-2 text-xs text-zinc-400">
                            Overlap
                            <input
                              type="number"
                              value={overlap}
                              onChange={(e) => setOverlap(Number(e.target.value))}
                              className="w-24 rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-zinc-100"
                            />
                          </label>

                          {/* Upload (right aligned) */}
                          <div className="ml-auto -mt-5">
                            <button
                              onClick={handleUploadPipeline}
                              disabled={
                                !file ||
                                stage === "uploading" ||
                                stage === "chunking" ||
                                stage === "indexing" ||
                                stage === "asking"
                              }
                              className="rounded-xl bg-white px-5 py-2 text-sm font-medium text-black disabled:opacity-50 hover:opacity-90"
                            >
                              Upload
                            </button>
                          </div>
                        </div>

                        {error && (
                          <div className="mt-3 rounded-xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-rose-200 text-sm">
                            {error}
                          </div>
                        )}
                      </div>
                    )}

                  </div>

                  {/* Chat stream */}
                  <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 space-y-5">
                    <div className="text-xs text-zinc-400">
                      {selectedDocIds.length > 0 ? (
                        <>
                          Asking across{" "}
                          <span className="text-zinc-200 font-medium">
                            {selectedDocIds.length}
                          </span>{" "}
                          selected documents.{" "}
                          <button
                            type="button"
                            onClick={() => setSelectedDocIds([])}
                            className="underline hover:text-zinc-200"
                          >
                            clear selection
                          </button>
                        </>
                      ) : (
                        <>Asking only the opened document.</>
                      )}
                    </div>

                    {docId ? (
                      activeDocChat.length ? (
                        activeDocChat.map((m) => (
                          <div key={m.id} className="space-y-4">
                            {/* user row */}
                            <div className="flex justify-end gap-3">
                              <div className="max-w-[78%] rounded-2xl rounded-br-sm bg-sky-500/20 border border-sky-500/25 px-4 py-3">
                                <div className="text-[15px] leading-7 text-zinc-100 whitespace-pre-wrap">
                                  {m.question}
                                </div>
                                <div className="mt-2 text-[11px] text-zinc-400">
                                  {new Date(m.created_at).toLocaleString()}
                                </div>
                              </div>

                              <div className="h-9 w-9 shrink-0 rounded-full bg-sky-500/20 border border-sky-500/25 flex items-center justify-center text-sm font-semibold text-sky-200">
                                U
                              </div>
                            </div>

                            {/* assistant row */}
                            <div className="flex justify-start gap-3">
                              <div className="h-9 w-9 shrink-0 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-sm font-semibold text-zinc-200">
                                A
                              </div>

                              <div className="max-w-[82%] rounded-2xl rounded-bl-sm bg-white/5 border border-white/10 px-4 py-3">
                                <div className="text-[15px] leading-7 text-zinc-100 whitespace-pre-wrap">
                                  {renderAnswerWithClickableCitations(m.answer, {
                                    evidence: m.evidence ?? [],
                                    doc_labels: m.doc_labels ?? undefined,
                                  })}
                                </div>

                                <div className="mt-3 flex flex-wrap items-center gap-3">
                                  <button
                                    type="button"
                                    onClick={() => copyToClipboard(m.answer)}
                                    className="text-xs rounded-full border border-white/10 bg-white/5 px-3 py-1 text-zinc-200 hover:bg-white/10"
                                  >
                                    Copy
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => setShowDebug((s) => !s)}
                                    className="text-xs text-zinc-300 underline hover:text-zinc-100"
                                  >
                                    {showDebug ? "Hide debug" : "Show debug"}
                                  </button>
                                </div>

                                {showDebug && (
                                  <pre className="mt-3 overflow-auto rounded-xl border border-white/10 bg-black/30 p-3 text-xs text-zinc-200">
                                    {JSON.stringify(
                                      {
                                        message_doc_labels: m.doc_labels,
                                        message_evidence_count:
                                          (m.evidence ?? []).length,
                                      },
                                      null,
                                      2
                                    )}
                                  </pre>
                                )}
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-sm text-zinc-400">
                          No chat yet for this document. Ask your first question below.
                        </div>
                      )
                    ) : (
                      <div className="text-sm text-zinc-400">
                        Select or upload a document to begin.
                      </div>
                    )}

                    {stage === "asking" ? <TypingIndicator /> : null}

                    <div ref={chatEndRef} />
                  </div>

                  {/* Composer */}
                  <div className="shrink-0 border-t border-white/10 p-2 bg-[#0f1520]">
                    <div className="flex items-end gap-3">
                      <textarea
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        onKeyDown={onQuestionKeyDown}
                        placeholder="Ask something…"
                        className="min-h-[54px] max-h-[170px] flex-1 resize-none rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-[15px] leading-6 text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-sky-500/30"
                        disabled={!canAsk && stage !== "asking"}
                      />

                      <button
                        onClick={handleAsk}
                        disabled={!canAsk || !question.trim() || stage === "asking"}
                        className="shrink-0 -translate-y-4 rounded-xl bg-sky-500 px-5 py-3 text-sm font-semibold text-white disabled:opacity-50 hover:opacity-90"
                      >
                        {stage === "asking" ? "Asking…" : "Send"}
                      </button>
                    </div>
                  </div>
                </section>

                {/* draggable divider */}
                <div
                  className="group relative w-[10px] cursor-col-resize flex items-stretch justify-center"
                  onMouseDown={() => (draggingRef.current = true)}
                  title="Drag to resize"
                >
                  <div className="absolute inset-y-0 w-[2px] rounded-full bg-white/10 group-hover:bg-white/20 transition" />
                  <div className="absolute inset-y-0 w-[10px]" />
                </div>

                {/* PDF pane */}
                <section
                  className="min-w-0 rounded-2xl border border-white/10 bg-[#0f1520] shadow-sm overflow-hidden flex flex-col min-h-0"
                  style={{ width: `${100 - chatPanePct}%` }}
                >
                  <div className="shrink-0 flex items-center justify-between border-b border-white/10 px-4 py-3">
                    <div className="text-sm font-medium">PDF Preview</div>
                    <div className="text-xs text-zinc-400">
                      {docId ? `Page ${selectedPage}` : "Upload a PDF to preview"}
                    </div>
                  </div>

                  <div className="min-h-0 flex-1 bg-black/20">
                    {docId ? (
                      <PdfPreview
                        fileUrl={pdfUrl}
                        pageNumber={selectedPage}
                        highlightRects={highlightRects}
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center text-sm text-zinc-400">
                        No document loaded
                      </div>
                    )}
                  </div>
                </section>
              </div>
            </div>
            {/* end right */}
          </div>
        </div>
      </div>
    </main>
  );
}
