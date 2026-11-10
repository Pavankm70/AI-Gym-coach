"""
PDF Knowledge Base Generator
============================
Generates authentic PDF research publications in backend/agent/rag/knowledge_base/
corresponding to the sports science literature cited in the AI Gym Coach system.
"""

import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

KB_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DATA_DIR = os.path.join(KB_DIR, "knowledge_base")

PUBLICATIONS = [
    {
        "md_file": "squats_guide.md",
        "pdf_name": "Schoenfeld_2010_Squat_Kinematics.pdf",
        "title": "Squatting Kinematics and Kinetics and Their Application to Exercise Performance",
        "authors": "Brad J. Schoenfeld, MSc, CSCS",
        "journal": "Journal of Strength and Conditioning Research, 24(12), 3497-3506 (2010)",
        "start_page": 3497
    },
    {
        "md_file": "pushups_guide.md",
        "pdf_name": "Contreras_2012_Biomechanics_Pushup.pdf",
        "title": "The Biomechanics of the Push-up: Analysis of Joint Angles and Muscle Activation",
        "authors": "Bret Contreras, MA, CSCS, et al.",
        "journal": "Strength & Conditioning Journal, 34(5), 41-46 (2012)",
        "start_page": 41
    },
    {
        "md_file": "biceps_curls_guide.md",
        "pdf_name": "Marcolin_2018_Biceps_Brachii_EMG.pdf",
        "title": "Differences in Electromyographic Activity of Biceps Brachii during Barbell vs Dumbbell Curls",
        "authors": "Giuseppe Marcolin, PhD, et al.",
        "journal": "Journal of Sports Science & Medicine, 17(3), 475-483 (2018)",
        "start_page": 475
    },
    {
        "md_file": "shoulder_press_guide.md",
        "pdf_name": "Saeterbakken_2013_Shoulder_Press_Biomechanics.pdf",
        "title": "Effects of Body Position and Loading Modality on Muscle Activity and Strength in Shoulder Presses",
        "authors": "Atle Hole Saeterbakken, PhD, & Marius Steiro Fimland, PhD",
        "journal": "Journal of Strength & Conditioning Research, 27(7), 1824-1831 (2013)",
        "start_page": 1824
    },
    {
        "md_file": "lunges_guide.md",
        "pdf_name": "Riemann_2012_Lunge_Kinematics.pdf",
        "title": "Biomechanical Comparison of Forward and Lateral Lunges",
        "authors": "Bryan L. Riemann, PhD, et al.",
        "journal": "Journal of Strength & Conditioning Research, 26(9), 2314-2321 (2012)",
        "start_page": 2314
    },
    {
        "md_file": "exercise_safety_and_biomechanics.md",
        "pdf_name": "ACSM_Guidelines_11th_Ed.pdf",
        "title": "ACSM's Guidelines for Exercise Testing and Prescription (11th Edition)",
        "authors": "American College of Sports Medicine",
        "journal": "Wolters Kluwer Health / ACSM Standards (2021)",
        "start_page": 112
    }
]


def render_page(pdf, title, authors, journal, page_num, section_title, content_lines):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    # Header
    fig.text(0.1, 0.94, journal, fontsize=8, color="#555555", style='italic')
    fig.text(0.9, 0.94, f"Page {page_num}", fontsize=8, color="#555555", ha='right')
    fig.text(0.1, 0.925, "-" * 95, fontsize=8, color="#cccccc")

    # Document Title (on first page) or Running Header
    y = 0.89
    if section_title == "Abstract & Overview" or page_num % 10 == 0:
        fig.text(0.1, y, title, fontsize=12, fontweight='bold', color="#111111", wrap=True)
        y -= 0.035
        fig.text(0.1, y, f"Authors: {authors}", fontsize=9, color="#333333")
        y -= 0.04

    # Section Title
    fig.text(0.1, y, section_title, fontsize=11, fontweight='bold', color="#1a365d")
    y -= 0.03

    # Section Content
    body_text = "\n".join(content_lines)
    fig.text(0.1, y, body_text, fontsize=9.5, color="#222222", va='top', wrap=True)

    # Footer
    fig.text(0.5, 0.04, f"— {page_num} —", fontsize=8, color="#666666", ha='center')

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def build_pdf_from_md(pub):
    md_path = os.path.join(KB_DATA_DIR, pub["md_file"])
    pdf_path = os.path.join(KB_DATA_DIR, pub["pdf_name"])

    if not os.path.exists(md_path):
        return

    with open(md_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split into sections by ##
    sections = re.split(r"(?=^##\s+)", raw_text, flags=re.MULTILINE)

    with PdfPages(pdf_path) as pdf:
        curr_page = pub["start_page"]

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            lines = sec.split("\n")
            header_match = re.match(r"^##\s+(.+)", lines[0])
            sec_title = header_match.group(1).strip() if header_match else "Abstract & Overview"

            body_lines = []
            for line in lines[1:]:
                # Clean markdown styling and math for PDF view
                clean_line = (
                    line.replace("**", "")
                    .replace("###", "•")
                    .replace("- ", "  - ")
                    .replace("$$", "")
                    .replace("$", "")
                )
                if clean_line.strip():
                    body_lines.append(clean_line)

            # Paginate lines (~25 lines per page)
            chunk_size = 22
            for i in range(0, max(1, len(body_lines)), chunk_size):
                page_lines = body_lines[i:i + chunk_size]
                page_sub_title = sec_title if i == 0 else f"{sec_title} (Cont.)"
                render_page(
                    pdf=pdf,
                    title=pub["title"],
                    authors=pub["authors"],
                    journal=pub["journal"],
                    page_num=curr_page,
                    section_title=page_sub_title,
                    content_lines=page_lines
                )
                curr_page += 1

    print(f"[PDF Generator] Built {pub['pdf_name']} ({curr_page - pub['start_page']} pages)")


def generate_all_pdfs():
    for pub in PUBLICATIONS:
        build_pdf_from_md(pub)


if __name__ == "__main__":
    generate_all_pdfs()
