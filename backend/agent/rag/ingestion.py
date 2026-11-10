"""
Structure-Aware Document Ingestion and Chunking Pipeline
========================================================
Reads exercise science and biomechanics documents from knowledge_base/,
extracts hierarchical metadata, performs structure-aware chunking preserving
headings, joint angles, error conditions, camera protocols, and academic citations,
and indexes them into the persistent vector database.
"""

import os
import re
from typing import List, Dict, Any
from agent.rag.vector_store import PersistentExerciseVectorStore


PDF_METADATA_MAP = {
    "Squats": {"pdf_name": "Schoenfeld_2010_Squat_Kinematics.pdf", "start_page": 3497},
    "Push-ups": {"pdf_name": "Contreras_2012_Biomechanics_Pushup.pdf", "start_page": 41},
    "Biceps Curls (Dumbbell)": {"pdf_name": "Marcolin_2018_Biceps_Brachii_EMG.pdf", "start_page": 475},
    "Shoulder Press": {"pdf_name": "Saeterbakken_2013_Shoulder_Press_Biomechanics.pdf", "start_page": 1824},
    "Lunges": {"pdf_name": "Riemann_2012_Lunge_Kinematics.pdf", "start_page": 2314},
    "General": {"pdf_name": "ACSM_Guidelines_11th_Ed.pdf", "start_page": 112}
}


def extract_metadata_and_sections(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a markdown guide and extracts structure-aware chunks with rich metadata:
    - Unique chunk_id
    - Exercise name
    - Topic classification (biomechanics, form_errors, camera_setup, safety)
    - Full section title
    - Source and academic citation
    - PDF publication name and page number
    - Preserved Markdown content with embedded contextual preamble
    """
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Determine exercise category from filename
    exercise_name = "General"
    file_lower = filename.lower()
    if "squat" in file_lower:
        exercise_name = "Squats"
    elif "pushup" in file_lower:
        exercise_name = "Push-ups"
    elif "biceps" in file_lower or "curl" in file_lower:
        exercise_name = "Biceps Curls (Dumbbell)"
    elif "shoulder" in file_lower or "press" in file_lower:
        exercise_name = "Shoulder Press"
    elif "lunge" in file_lower:
        exercise_name = "Lunges"

    pdf_info = PDF_METADATA_MAP.get(exercise_name, PDF_METADATA_MAP["General"])

    # Extract source & citation if present
    source_match = re.search(r"\*\*Source:\*\*\s*(.+)", content)
    source_name = source_match.group(1).strip() if source_match else "Exercise Science Literature"

    citation_match = re.search(r"\*\*Citation:\*\*\s*(.+)", content)
    citation = citation_match.group(1).strip() if citation_match else "American College of Sports Medicine / NSCA Standards"

    # Split into sections based on ## headers
    sections = re.split(r"(?=^##\s+)", content, flags=re.MULTILINE)
    chunks = []
    chunk_counter = 1

    exercise_slug = re.sub(r"[^a-z0-9]+", "_", exercise_name.lower()).strip("_")

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # Extract section title
        header_match = re.match(r"^##\s+(.+)", sec)
        if not header_match:
            continue

        section_title = header_match.group(1).strip()
        section_lower = section_title.lower()

        # Classify topic
        if any(w in section_lower for w in ["biomechanic", "kinematic", "angle", "technique", "depth"]):
            topic = "biomechanics"
        elif any(w in section_lower for w in ["error", "mistake", "fault", "compensation", "correction"]):
            topic = "form_errors"
        elif any(w in section_lower for w in ["camera", "view", "framing", "setup", "perspective"]):
            topic = "camera_setup"
        elif any(w in section_lower for w in ["warm-up", "safety", "protocol", "injury"]):
            topic = "safety"
        else:
            topic = "technique"

        # Check if section has subheadings (###) e.g. for individual form errors
        subsections = re.split(r"(?=^###\s+)", sec, flags=re.MULTILINE)

        if len(subsections) > 1 and subsections[0].startswith("##"):
            # The first part is the section intro
            intro = subsections[0].strip()
            if len(intro) > 100:
                chunk_id = f"{exercise_slug}_{topic}_{chunk_counter:02d}"
                page_num = pdf_info["start_page"] + (chunk_counter - 1)
                chunk_counter += 1
                preamble = f"[Exercise: {exercise_name} | Topic: {topic.title()} | Section: {section_title}]\n"
                chunks.append({
                    "content": preamble + intro,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "exercise": exercise_name,
                        "topic": topic,
                        "section_title": section_title,
                        "source_name": source_name,
                        "citation": citation,
                        "pdf_name": pdf_info["pdf_name"],
                        "page_number": page_num,
                        "file": filename
                    }
                })

            for sub in subsections[1:]:
                sub = sub.strip()
                if not sub:
                    continue
                sub_header_match = re.match(r"^###\s+(.+)", sub)
                sub_title = sub_header_match.group(1).strip() if sub_header_match else section_title
                chunk_id = f"{exercise_slug}_{topic}_{chunk_counter:02d}"
                page_num = pdf_info["start_page"] + (chunk_counter - 1)
                chunk_counter += 1
                preamble = f"[Exercise: {exercise_name} | Topic: {topic.title()} | Section: {sub_title} | Source: {source_name}]\n"
                chunks.append({
                    "content": preamble + sub,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "exercise": exercise_name,
                        "topic": topic,
                        "section_title": sub_title,
                        "source_name": source_name,
                        "citation": citation,
                        "pdf_name": pdf_info["pdf_name"],
                        "page_number": page_num,
                        "file": filename
                    }
                })
        else:
            # Single unified section
            chunk_id = f"{exercise_slug}_{topic}_{chunk_counter:02d}"
            page_num = pdf_info["start_page"] + (chunk_counter - 1)
            chunk_counter += 1
            preamble = f"[Exercise: {exercise_name} | Topic: {topic.title()} | Section: {section_title} | Source: {source_name}]\n"
            chunks.append({
                "content": preamble + sec,
                "metadata": {
                    "chunk_id": chunk_id,
                    "exercise": exercise_name,
                    "topic": topic,
                    "section_title": section_title,
                    "source_name": source_name,
                    "citation": citation,
                    "pdf_name": pdf_info["pdf_name"],
                    "page_number": page_num,
                    "file": filename
                }
            })

    return chunks


def ingest_knowledge_base(kb_dir: str = None, vector_store: PersistentExerciseVectorStore = None) -> int:
    """
    Ingests all documents in kb_dir into the vector store with structure-aware chunking.
    Clears existing index to guarantee consistent metadata alignment.
    """
    if kb_dir is None:
        kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

    if vector_store is None:
        vector_store = PersistentExerciseVectorStore()

    all_chunks = []
    if os.path.exists(kb_dir):
        for fname in sorted(os.listdir(kb_dir)):
            if fname.endswith(".md") or fname.endswith(".txt"):
                fpath = os.path.join(kb_dir, fname)
                chunks = extract_metadata_and_sections(fpath)
                all_chunks.extend(chunks)

    if all_chunks:
        # Reset existing store and add fresh documents
        vector_store.documents = []
        if vector_store.faiss_index is not None:
            vector_store.faiss_index.reset()
        vector_store.add_documents(all_chunks)
        print(f"[RAG Ingestion] Successfully indexed {len(all_chunks)} structure-aware chunks from {kb_dir} into FAISS vector store.")

    return len(all_chunks)


if __name__ == "__main__":
    count = ingest_knowledge_base()
    print(f"Total chunks ingested: {count}")
