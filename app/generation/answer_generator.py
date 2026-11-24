import re
import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import urllib.request
import urllib.error

from app.schemas.search import SearchResultChunk
from app.schemas.answer import Citation, AskResponse

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    RAG Answer Generation with page-level citations.
    Supports local mock extractive synthesis, local Ollama (CPU LLMs),
    and OpenAI-compatible / Gemini APIs.
    """
    def __init__(
        self,
        provider: str = "mock",
        model: str = "mock-llm",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.model = model or os.getenv("LLM_MODEL", "mock-llm")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")

    def generate_answer(
        self,
        query: str,
        context_chunks: List[SearchResultChunk],
    ) -> Tuple[str, List[Citation]]:
        """
        Generates an answer grounded strictly in retrieved context chunks.
        Extracts document names and page numbers into citations.
        """
        if not context_chunks:
            return (
                "The uploaded documents do not contain sufficient information to answer this question.",
                [],
            )

        citations = self._build_citations(context_chunks)

        if self.provider == "ollama":
            answer = self._generate_ollama(query, context_chunks)
        elif self.provider in ("openai", "openai-compatible"):
            answer = self._generate_openai(query, context_chunks)
        elif self.provider == "gemini":
            answer = self._generate_gemini(query, context_chunks)
        else:
            answer = self._generate_mock(query, context_chunks)

        return answer, citations

    def _build_citations(self, chunks: List[SearchResultChunk]) -> List[Citation]:
        seen = set()
        citations = []
        for c in chunks:
            key = (c.document_id, c.page_number)
            if key not in seen:
                seen.add(key)
                snippet = c.chunk_text.strip().replace("\n", " ")[:200]
                citations.append(
                    Citation(
                        document_name=c.document_name,
                        document_id=c.document_id,
                        page_number=c.page_number,
                        section_title=getattr(c, "section_title", None),
                        chunk_id=c.chunk_id,
                        snippet=snippet,
                    )
                )
        return citations

    def _format_context_prompt(self, chunks: List[SearchResultChunk]) -> str:
        prompt_parts = []
        for idx, c in enumerate(chunks, 1):
            sec = f" | Section: {c.section_title}" if getattr(c, "section_title", None) else ""
            prompt_parts.append(
                f"[Source {idx} - Document: {c.document_name}, Page: {c.page_number}{sec}]\n{c.chunk_text}"
            )
        return "\n\n".join(prompt_parts)

    def _generate_mock(self, query: str, chunks: List[SearchResultChunk]) -> str:
        """
        Deterministic, local extractive answer generator:
        - Extracts sentences and table rows addressing the query
        - Clearly formats findings with page references
        - If query terms are not in context, responds with insufficient information
        """
        q_clean = re.findall(r"\b\w+\b", query.lower())
        stopwords = {"what", "is", "the", "are", "how", "why", "in", "of", "and", "to", "a", "an", "do", "does"}
        keywords = [w for w in q_clean if w not in stopwords]

        relevant_snippets = []
        table_snippets = []

        for c in chunks:
            text = c.chunk_text
            doc_ref = f"[{c.document_name}, Page {c.page_number}]"

            # Check if chunk contains tables
            if c.content_type == "table":
                for line in text.split("\n"):
                    if any(k in line.lower() for k in keywords) and "|" in line:
                        table_snippets.append(f"- From table in {doc_ref}: `{line.strip()}`")

            # Check sentences
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                s_strip = s.strip()
                if not s_strip:
                    continue
                match_count = sum(1 for k in keywords if k in s_strip.lower())
                if match_count > 0:
                    relevant_snippets.append((match_count, f"{s_strip} {doc_ref}"))

        relevant_snippets.sort(key=lambda x: x[0], reverse=True)

        if not relevant_snippets and not table_snippets:
            return (
                "Based on the uploaded documents, there is no direct information regarding your query. "
                "The documents do not contain sufficient evidence to formulate an answer."
            )

        answer_parts = []
        answer_parts.append("Based on the uploaded documents, here is the verified information:")
        
        # Include top relevant statements
        for _, snip in relevant_snippets[:4]:
            answer_parts.append(f"• {snip}")

        # Include table findings if present
        if table_snippets:
            answer_parts.append("\nRelevant Tabular Data:")
            for t_snip in table_snippets[:3]:
                answer_parts.append(t_snip)

        answer_parts.append("\n*Note: Citations indicate the exact source document and page number.*")
        return "\n".join(answer_parts)

    def _generate_ollama(self, query: str, chunks: List[SearchResultChunk]) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate" if self.base_url else "http://localhost:11434/api/generate"
        context_str = self._format_context_prompt(chunks)
        prompt = (
            f"You are a factual assistant. Answer the user's question using ONLY the context provided below.\n"
            f"Always cite the source document name and page number for each fact.\n"
            f"If the answer cannot be found in the context, say 'The uploaded documents do not contain sufficient information.'\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}. Falling back to local extractive.")
            return self._generate_mock(query, chunks)

    def _generate_openai(self, query: str, chunks: List[SearchResultChunk]) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions" if self.base_url else "https://api.openai.com/v1/chat/completions"
        context_str = self._format_context_prompt(chunks)
        messages = [
            {
                "role": "system",
                "content": "You are a factual AI assistant. Answer using ONLY the provided document context. Cite document names and page numbers. Never make up facts."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {query}"
            }
        ]
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": 0.0}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}. Falling back to local extractive.")
            return self._generate_mock(query, chunks)

    def _generate_gemini(self, query: str, chunks: List[SearchResultChunk]) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        context_str = self._format_context_prompt(chunks)
        prompt_text = (
            f"Answer the question strictly using the document context provided below.\n"
            f"Include document and page citations.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {query}"
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt_text}]}]
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}. Falling back to local extractive.")
            return self._generate_mock(query, chunks)
