PDF QA — Retrieval-Augmented Generation (RAG)
    A modern PDF Question Answering web application that enables users to upload documents, ask natural language questions, and receive grounded answers with page-level citations and visual highlights inside the PDF.
    This project is a practical, UI-focused implementation of Retrieval-Augmented Generation (RAG), inspired by foundational research and adapted into a real-world, interactive web experience.

> Research Inspiration:
- This project is inspired by the paper:
    Lewis, Patrick, et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    Authors include Patrick Lewis and Ethan Perez, among others.

- The paper introduces the core idea of combining:
    - Dense retrieval over an external knowledge source
    - With generation models to produce grounded, factual responses

- How this project applies the paper’s ideas
    - Uses retrieval over document chunks before answering
    - Answers are generated only from retrieved context
    -  Maintains a clear separation between:
        - Retrieval
        - Generation
        - Source attribution

- How this project goes beyond the paper:
    While the paper focuses on model architecture, this project extends the concept into a user-facing system with:
    - Page-level citations ([p3], [D1 p5])
    - Clickable citations that navigate the original PDF
    - Visual text highlighting inside the document
    - A ChatGPT-style conversational interface
    - Multi-document querying in a single session
    
    The goal was to translate research concepts into a usable product, not just a backend pipeline.

- Features:
    - Upload PDFs with automatic:
        - Text extraction
        - Chunking with overlap
        - Embedding generation
        - Vector indexing
    - Chat-style question answering
    - Page-level citations in answers
    - Click-to-jump citations with - text highlighting
    - Multi-document querying
    - Modern dark UI - (ChatGPT-inspired)
    - Persistent document state and chat history

> System Architecture (High-Level):
    - PDF Ingestion
        Text extracted with page mapping
    - Chunking
        Overlapping chunks preserve context
    - Embedding & Indexing
        Chunks embedded and stored in a vector index
    - Retrieval
        Top-k relevant chunks selected per query
    - Answer Generation
        Response generated using retrieved context
    - Citation & Highlighting
        - Citations map answers back to exact PDF pages
        - Highlight rectangles are computed from matched text

> UI Overview:
    - Left Panel
        - Document list
        - Indexing status
        - Multi-document selection
    - Center Panel
        - Chat-style conversation
        - Clickable citations
        - Citation hover previews
    - Right Panel
        - PDF preview
        - Automatic page navigation
        - Visual highlights of referenced text

> Example Use Cases:
    - Research paper analysis
    - Legal or policy documents
    - Academic study material
    - Business reports
    - Long PDFs where keyword search is insufficient

> Tech Stack:
    - Frontend
        - Next.js (App Router)
        - React + TypeScript
        - Tailwind CSS
        - react-pdf / pdfjs


    - Backend:
        - PDF parsing and page mapping
        - Chunking with overlap
        - Embedding generation
        - Vector similarity search (FAISS)
        - RAG-style answer generation
        - Text-to-coordinate highlighting

> Setup:
    npm install
    npm run dev

> Create .env.local:
    NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

> Design Notes:
    - Highlighting is text-based, not page-based, for precision
    - Citations are first-class UI elements, not metadata
    - Chat history is scoped per document
    - UI prioritizes reading flow over control-heavy layouts

> Future Improvements:
    -  Streaming responses
    -  Export answers with citations
    -  Authentication and cloud storage
    -  PDF page navigation controls
