import os
import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List

API_URL = os.getenv("HYBRID_RAG_API_URL", "http://localhost:8000").rstrip("/")

app = FastAPI(title="HybridRAG UI")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HybridRAG PDF Assistant</title>
  <style>
    :root { --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #f1f5f9; --muted: #94a3b8; --accent: #3b82f6; --accent-hover: #2563eb; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); }
    .container { max-width: 960px; margin: 32px auto; padding: 0 16px; }
    h1 { margin: 0 0 6px; font-size: 1.6rem; }
    .subtitle { color: var(--muted); margin-bottom: 24px; font-size: 0.92rem; }
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
    .tab-btn { background: var(--card); color: var(--muted); border: 1px solid var(--border); padding: 10px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    .tab-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
    .hidden { display: none; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
    .grow { flex: 1 1 400px; }
    label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }
    textarea, select, input[type="number"], input[type="file"] {
      width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 12px; font-size: 0.95rem;
    }
    button.primary {
      background: var(--accent); color: #fff; border: none; padding: 11px 22px;
      border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.95rem; margin-top: 12px;
    }
    button.primary:hover { background: var(--accent-hover); }
    button.primary:disabled { opacity: 0.6; cursor: wait; }
    .answer-box { white-space: pre-wrap; line-height: 1.6; font-size: 0.98rem; }
    .meta { margin-top: 12px; font-size: 0.82rem; color: var(--muted); }
    .item { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-top: 10px; }
    .item-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 6px; color: #93c5fd; }
    .snippet { font-size: 0.88rem; color: #cbd5e1; white-space: pre-wrap; }
    details summary { cursor: pointer; font-weight: 600; color: #cbd5e1; }
  </style>
</head>
<body>
  <div class="container">
    <h1>HybridRAG PDF Assistant</h1>
    <div class="subtitle">Dense (<code>BAAI/bge-small-en-v1.5</code>) + BM25 + SPLADE &bull; RRF (<code>k = 10</code>) &bull; OpenRouter</div>

    <div class="tabs">
      <button class="tab-btn active" onclick="showTab('ask', this)">Ask Questions</button>
      <button class="tab-btn" onclick="showTab('upload', this)">Upload PDFs</button>
    </div>

    <div id="tab-ask">
      <div class="card">
        <div class="row">
          <div class="grow">
            <label for="query">Question</label>
            <textarea id="query" rows="3" placeholder="Ask anything about your uploaded PDFs..."></textarea>
          </div>
          <div style="width: 170px;">
            <label for="mode">Retrieval Mode</label>
            <select id="mode">
              <option value="hybrid" selected>hybrid</option>
              <option value="dense">dense</option>
              <option value="bm25">bm25</option>
              <option value="splade">splade</option>
              <option value="bm25_dense">bm25_dense</option>
              <option value="bm25_splade">bm25_splade</option>
            </select>
          </div>
          <div style="width: 100px;">
            <label for="top_k">Top K</label>
            <input id="top_k" type="number" min="1" max="15" value="5" />
          </div>
        </div>
        <button id="askBtn" class="primary" onclick="submitAsk()">Ask</button>
      </div>

      <div id="askResult" class="hidden">
        <div class="card">
          <h3 style="margin-top:0;">Answer</h3>
          <div id="answerText" class="answer-box"></div>
          <div id="answerMeta" class="meta"></div>
        </div>

        <div class="card">
          <details open>
            <summary>Citations</summary>
            <div id="citationsList"></div>
          </details>
        </div>

        <div class="card">
          <details>
            <summary>Retrieved Chunks (RRF)</summary>
            <div id="chunksList"></div>
          </details>
        </div>
      </div>
    </div>

    <div id="tab-upload" class="hidden">
      <div class="card">
        <label for="pdfFiles">Select PDF file(s)</label>
        <input id="pdfFiles" type="file" accept=".pdf" multiple />
        <button id="uploadBtn" class="primary" onclick="submitUpload()">Upload &amp; Index</button>
        <div id="uploadStatus" style="margin-top: 14px;"></div>
      </div>
    </div>
  </div>

  <script>
    function showTab(name, btn) {
      document.getElementById('tab-ask').classList.toggle('hidden', name !== 'ask');
      document.getElementById('tab-upload').classList.toggle('hidden', name !== 'upload');
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }

    function escapeHtml(str) {
      return String(str ?? '').replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
    }

    async function submitAsk() {
      const query = document.getElementById('query').value.trim();
      const mode = document.getElementById('mode').value;
      const top_k = parseInt(document.getElementById('top_k').value || '5', 10);
      if (!query) { alert('Please enter a question.'); return; }

      const btn = document.getElementById('askBtn');
      btn.disabled = true;
      btn.textContent = 'Thinking...';
      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ query, mode, top_k, rerank: false })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || JSON.stringify(data));

        document.getElementById('askResult').classList.remove('hidden');
        document.getElementById('answerText').textContent = data.answer || 'No answer returned.';
        document.getElementById('answerMeta').textContent =
          `Latency: ${data.latency_ms ?? 0} ms | Provider: ${data.llm_provider ?? 'unknown'} | Mode: ${mode}`;

        const cits = data.citations || [];
        document.getElementById('citationsList').innerHTML = cits.length
          ? cits.map((c, i) => `
              <div class="item">
                <div class="item-title">[${i + 1}] ${escapeHtml(c.document_name)} — Page ${escapeHtml(c.page_number)}${c.section_title ? ' | ' + escapeHtml(c.section_title) : ''}</div>
                <div class="snippet">${escapeHtml(c.snippet)}</div>
              </div>`).join('')
          : '<div class="meta">No citations returned.</div>';

        const chunks = data.retrieved_chunks || [];
        document.getElementById('chunksList').innerHTML = chunks.length
          ? chunks.map((ch, i) => `
              <div class="item">
                <div class="item-title">Chunk ${i + 1} — ${escapeHtml(ch.document_name)} (Page ${escapeHtml(ch.page_number)}) | Score: ${escapeHtml(ch.score)} | ${escapeHtml(ch.retrieval_method)} | Tokens: ${escapeHtml(ch.token_count)}</div>
                <div class="snippet">${escapeHtml(ch.chunk_text)}</div>
              </div>`).join('')
          : '<div class="meta">No chunks retrieved.</div>';
      } catch (err) {
        document.getElementById('askResult').classList.remove('hidden');
        document.getElementById('answerText').textContent = 'Error: ' + err.message;
        document.getElementById('citationsList').innerHTML = '';
        document.getElementById('chunksList').innerHTML = '';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Ask';
      }
    }

    async function submitUpload() {
      const input = document.getElementById('pdfFiles');
      const status = document.getElementById('uploadStatus');
      if (!input.files.length) { status.innerHTML = '<div class="item">Please select at least one PDF file.</div>'; return; }

      const form = new FormData();
      for (const f of input.files) form.append('files', f);

      const btn = document.getElementById('uploadBtn');
      btn.disabled = true;
      btn.textContent = 'Uploading & Indexing...';
      status.innerHTML = '<div class="meta">Uploading and generating embeddings... please wait.</div>';

      try {
        const res = await fetch('/api/upload', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
        status.innerHTML = data.map(doc => `
          <div class="item">
            <div class="item-title">${escapeHtml(doc.document_name)} — ${escapeHtml(doc.status)}</div>
            <div class="snippet">${doc.status === 'indexed'
              ? `Chunks: ${escapeHtml(doc.chunk_count)} | Pages: ${escapeHtml(doc.total_pages)}`
              : escapeHtml(doc.error_message || 'Failed')}</div>
          </div>`).join('');
      } catch (err) {
        status.innerHTML = `<div class="item">Upload failed: ${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Upload & Index';
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.post("/api/ask")
async def proxy_ask(request: Request):
    body = await request.json()
    print(f"[UI] Ask: {body.get('query')!r}", flush=True)
    try:
        resp = requests.post(f"{API_URL}/ask", json=body, timeout=120)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": str(e)})


@app.post("/api/upload")
async def proxy_upload(files: List[UploadFile] = File(...)):
    print(f"[UI] Upload: {[f.filename for f in files]}", flush=True)
    try:
        multipart = [
            ("files", (f.filename, await f.read(), f.content_type or "application/pdf"))
            for f in files
        ]
        resp = requests.post(f"{API_URL}/upload", files=multipart, timeout=600)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
