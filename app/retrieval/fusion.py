from typing import List, Dict, Optional
from collections import defaultdict
from app.schemas.search import SearchResultChunk

class ReciprocalRankFusion:
    """
    Combines ranked results from multiple retrieval strategies using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum_{m in M} 1 / (k + rank_m(d))
    where k is a smoothing constant (standard default = 60).
    """
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        ranked_lists: List[List[SearchResultChunk]],
        top_k: int = 20,
    ) -> List[SearchResultChunk]:
        """
        Takes candidate lists from different retrievers (e.g. BM25, SPLADE, Dense),
        fuses their rankings using RRF, removes duplicates, and returns top_k fused candidates.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, SearchResultChunk] = {}
        methods_seen: Dict[str, List[str]] = defaultdict(list)

        for ranked_list in ranked_lists:
            for rank_idx, chunk in enumerate(ranked_list, start=1):
                chunk_id = chunk.chunk_id
                rrf_scores[chunk_id] += 1.0 / (self.k + rank_idx)
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk
                methods_seen[chunk_id].append(getattr(chunk, "retrieval_method", "unknown"))

        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_results = []
        for cid in sorted_chunk_ids[:top_k]:
            orig = chunk_map[cid]
            methods_str = "+".join(sorted(set(methods_seen[cid])))
            fused_chunk = SearchResultChunk(
                chunk_id=orig.chunk_id,
                document_id=orig.document_id,
                document_name=orig.document_name,
                page_number=orig.page_number,
                section_title=getattr(orig, "section_title", None),
                chunk_text=orig.chunk_text,
                token_count=getattr(orig, "token_count", 0),
                content_type=getattr(orig, "content_type", "text"),
                table_id=getattr(orig, "table_id", None),
                image_references=getattr(orig, "image_references", []),
                score=round(float(rrf_scores[cid]), 6),
                retrieval_method=f"rrf({methods_str})",
            )
            fused_results.append(fused_chunk)

        return fused_results
