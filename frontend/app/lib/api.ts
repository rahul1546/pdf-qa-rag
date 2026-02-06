const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL;

if (!API_BASE) {
  throw new Error("NEXT_PUBLIC_API_BASE_URL is missing in .env.local");
}

export type UploadResponse = {
  doc_id: string;
  filename: string;
  saved_as: string;
  num_pages: number;
  total_extracted_chars: number;
};

export type ChunkResponse = {
  doc_id: string;
  num_chunks: number;
  saved_to: string;
};

export type IndexResponse = {
  doc_id: string;
  num_vectors: number;
  embedding_dim: number;
  index_path: string;
  meta_path: string;
};

export type RetrievalHit = {
  doc_id?: string;
  page_number: number;
  chunk_id: string;
  score: number;
  preview?: string;
  text?: string;
};

export type EvidenceItem = {
  doc_id?: string;
  page_number: number;
  chunk_id: string;
  score: number;
  snippet: string;
};

export type AskResponse = {
  doc_id?: string;
  question: string;
  answer: string;
  answer_citations?: any;
  doc_labels?: Record<string, string>; // ✅ NEW
  retrieval_hits?: RetrievalHit[];
  retrieved_context?: RetrievalHit[];
  evidence?: EvidenceItem[];
};

export function pickHighlightText(
  result: AskResponse | null,
  page: number,
  docId?: string,
  docLabel?: string 
): string {
  if (!result) return "";

  
  const ans = result.answer ?? "";

  const marker = docLabel ? `[${docLabel} p${page}]` : `[p${page}]`;
  const idx = ans.indexOf(marker);

  if (idx !== -1) {
    const start = Math.max(0, idx - 220);
    const windowText = ans.slice(start, idx);

    const dq = windowText.match(/"([^"]{2,80})"\s*$/);
    if (dq?.[1]) return dq[1].trim();

    const sq = windowText.match(/'([^']{2,80})'\s*$/);
    if (sq?.[1]) return sq[1].trim();

    const cleaned = windowText.replace(/\s+/g, " ").trim();
    const parts = cleaned.split(" ");
    const tail = parts.slice(Math.max(0, parts.length - 4)).join(" ").trim();
    if (tail) return tail;
  }

  const ev = (result.evidence ?? []).find(
    (e) => e.page_number === page && (!docId || !e.doc_id || e.doc_id === docId)
  );
  if (ev?.snippet) return ev.snippet;

  const hit = (result.retrieval_hits ?? []).find(
    (h) => h.page_number === page && (!docId || !h.doc_id || h.doc_id === docId)
  );
  if (hit?.text) return hit.text;
  if (hit?.preview) return hit.preview;

  return "";
}



export async function uploadPDF(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function chunkDoc(docId: string, chunk_size = 1000, overlap = 150): Promise<ChunkResponse> {
  const res = await fetch(`${API_BASE}/chunk/${docId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk_size, overlap }),
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function indexDoc(docId: string): Promise<IndexResponse> {
  const res = await fetch(`${API_BASE}/index/${docId}`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function askDoc(docId: string, question: string, top_k = 5): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask/${docId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k }),
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
export async function askManyDocs(
  doc_ids: string[],
  question: string,
  top_k_per_doc = 3
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask_many`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_ids, question, top_k_per_doc }),
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export type HighlightRect = {
  x0: number; // 0..1
  y0: number; // 0..1
  x1: number; // 0..1
  y1: number; // 0..1
};

export type HighlightResponse = {
  doc_id: string;
  page_number: number;
  rects: HighlightRect[];
  matched: boolean;
};

export async function highlightText(payload: {
  doc_id: string;
  page_number: number;
  text: string;
  max_rects?: number;
}): Promise<HighlightResponse> {
  const res = await fetch(`${API_BASE}/highlight`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      doc_id: payload.doc_id,
      page_number: payload.page_number,
      text: payload.text,
      max_rects: payload.max_rects ?? 20,
    }),
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

