"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

type Rect = { x0: number; y0: number; x1: number; y1: number };

export default function PdfPreview({
  fileUrl,
  pageNumber,
  highlightRects = [],
  highlightFlashKey = 0,
  autoScrollToHighlight = false,
}: {
  fileUrl: string;
  pageNumber: number;
  highlightRects?: Rect[];
  highlightFlashKey?: number;
  autoScrollToHighlight?: boolean;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const [numPages, setNumPages] = useState<number>(0);
  const [containerWidth, setContainerWidth] = useState<number>(800);

  const [pageViewport, setPageViewport] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    setNumPages(0);
    setPageViewport(null);
  }, [fileUrl]);

  useEffect(() => {
    if (!wrapRef.current) return;
    const el = wrapRef.current;

    const measure = () => {
      const w = el.clientWidth;
      if (w) setContainerWidth(w);
    };

    measure();

    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const clampedPage = useMemo(() => {
    if (!numPages) return pageNumber;
    return Math.max(1, Math.min(pageNumber, numPages));
  }, [pageNumber, numPages]);

  const renderedHeight = useMemo(() => {
    if (!pageViewport) return 0;
    const scale = containerWidth / pageViewport.w;
    return pageViewport.h * scale;
  }, [pageViewport, containerWidth]);

  useEffect(() => {
    if (!autoScrollToHighlight) return;
    if (!wrapRef.current) return;
    if (!highlightRects || highlightRects.length === 0) return;
    if (!renderedHeight) return;

    const el = wrapRef.current;
    const first = highlightRects[0];

    const yCenter = (first.y0 + first.y1) / 2; // 0..1
    const targetY = yCenter * renderedHeight;

    const viewH = el.clientHeight || 0;
    const top = Math.max(0, targetY - viewH * 0.35);

    el.scrollTo({ top, behavior: "smooth" });
  }, [highlightFlashKey, autoScrollToHighlight, highlightRects, renderedHeight]);

  return (
    <div ref={wrapRef} className="h-full w-full overflow-auto">
      <style jsx>{`
        @keyframes pdfPulse {
          0% { transform: scale(1); opacity: 0.55; }
          45% { transform: scale(1.01); opacity: 0.9; }
          100% { transform: scale(1); opacity: 0.55; }
        }
      `}</style>

      <div className="relative inline-block">
        <Document
          key={fileUrl} 
          file={fileUrl}
          onLoadSuccess={(doc) => setNumPages(doc.numPages)}
          loading={<div className="p-4">Loading PDF…</div>}
          error={<div className="p-4 text-red-600">Failed to load PDF.</div>}
        >
          <Page
            key={`${fileUrl}-${clampedPage}`} 
            pageNumber={clampedPage}
            width={containerWidth}
            renderTextLayer={false}
            renderAnnotationLayer={false}
            loading={<div className="p-4">Loading page…</div>}
            onLoadSuccess={(page) => {
              const vp = page.getViewport({ scale: 1 });
              setPageViewport({ w: vp.width, h: vp.height });
            }}
          />
        </Document>

        {highlightRects.length > 0 && renderedHeight > 0 && (
          <div
            className="absolute left-0 top-0 pointer-events-none"
            style={{
              width: containerWidth,
              height: renderedHeight,
              zIndex: 50,
            }}
          >
            {highlightRects.map((r, idx) => {
              const left = `${r.x0 * 100}%`;
              const top = `${r.y0 * 100}%`;
              const width = `${(r.x1 - r.x0) * 100}%`;
              const height = `${(r.y1 - r.y0) * 100}%`;

              return (
                <div
                  key={`${highlightFlashKey}-${idx}`}
                  style={{
                    position: "absolute",
                    left,
                    top,
                    width,
                    height,
                    background: "rgba(255, 235, 59, 0.55)",
                    border: "1px solid rgba(255, 193, 7, 0.8)",
                    borderRadius: 4,
                    animation: "pdfPulse 900ms ease-out",
                  }}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
