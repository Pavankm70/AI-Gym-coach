"""
Exercise Knowledge Retrieval Module
===================================
Provides semantic search interface for the Exercise Knowledge Tool (Tool 2).
Retrieves grounded exercise science principles, biomechanical explanations,
and form corrections with full source citations, exercise filtering,
score thresholding, and diagnostic tracing.
"""

from typing import List, Dict, Any, Optional
from agent.rag.vector_store import PersistentExerciseVectorStore
from agent.rag.ingestion import ingest_knowledge_base

_GLOBAL_VECTOR_STORE: Optional[PersistentExerciseVectorStore] = None


def get_vector_store() -> PersistentExerciseVectorStore:
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        _GLOBAL_VECTOR_STORE = PersistentExerciseVectorStore()
        # Auto-ingest if empty
        if not _GLOBAL_VECTOR_STORE.documents:
            ingest_knowledge_base(vector_store=_GLOBAL_VECTOR_STORE)
    return _GLOBAL_VECTOR_STORE


def retrieve_exercise_knowledge(
    query: str,
    exercise_name: Optional[str] = None,
    top_k: int = 3,
    min_score: float = 0.15
) -> Dict[str, Any]:
    """
    Retrieves grounded exercise knowledge from the persistent vector store.

    Returns structured output:
    {
        "query": str,
        "results_count": int,
        "passages": [
            {
                "chunk_id": str,
                "content": str,
                "section": str,
                "exercise": str,
                "source": str,
                "citation": str,
                "score": float
            }
        ],
        "formatted_context": str
    }
    """
    store = get_vector_store()
    raw_results, _ = store.similarity_search_with_diagnostics(
        query=query,
        top_k=top_k,
        filter_exercise=exercise_name,
        min_score=min_score
    )

    passages = []
    formatted_lines = []

    for item in raw_results:
        meta = item.get("metadata", {})
        passage = {
            "chunk_id": meta.get("chunk_id", "unknown"),
            "content": item["content"],
            "section": meta.get("section_title", "General"),
            "exercise": meta.get("exercise", "General"),
            "source": meta.get("source_name", "Exercise Science Literature"),
            "pdf_name": meta.get("pdf_name", "exercise_guide.pdf"),
            "page_number": meta.get("page_number", 1),
            "citation": meta.get("citation", "NSCA / ACSM Guidelines"),
            "score": round(item.get("score", 0.0), 3)
        }
        passages.append(passage)

        formatted_lines.append(
            f"--- [Chunk: {passage['chunk_id']}] [PDF: {passage['pdf_name']} (Page {passage['page_number']})] [Source: {passage['source']}] [{passage['citation']}] ---\n"
            f"{passage['content']}\n"
        )

    return {
        "query": query,
        "exercise_filter": exercise_name,
        "results_count": len(passages),
        "passages": passages,
        "formatted_context": "\n".join(formatted_lines) if formatted_lines else "No specific research literature found for this query."
    }


def retrieve_exercise_knowledge_diagnostic(
    query: str,
    exercise_name: Optional[str] = None,
    top_k: int = 3,
    min_score: float = 0.15
) -> Dict[str, Any]:
    """
    Diagnostic retrieval mode exposing chunk IDs, similarity scores,
    exercise filters, and acceptance/rejection decisions for developer inspection.
    """
    store = get_vector_store()
    accepted, diagnostic_trace = store.similarity_search_with_diagnostics(
        query=query,
        top_k=top_k,
        filter_exercise=exercise_name,
        min_score=min_score
    )

    return {
        "query": query,
        "exercise_filter": exercise_name,
        "accepted_count": len(accepted),
        "total_candidates_examined": len(diagnostic_trace),
        "accepted_passages": [
            {
                "chunk_id": item["metadata"].get("chunk_id"),
                "exercise": item["metadata"].get("exercise"),
                "section": item["metadata"].get("section_title"),
                "score": item["score"],
                "citation": item["metadata"].get("citation")
            }
            for item in accepted
        ],
        "diagnostic_trace": diagnostic_trace[:10]  # Top candidates examined
    }
