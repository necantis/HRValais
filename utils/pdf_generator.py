"""
utils/pdf_generator.py
Generates assets/fiches_pratiques.pdf — the HR Valais survey PDF.
Uses fpdf2. Called once on first app run if the PDF doesn't exist.
"""

from __future__ import annotations
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets"

SURVEY_STRUCTURE = [
    ("Recrutement", [
        "1. Le poste correspond-il aux tâches réelles effectuées ?",
        "2. L'intégration (onboarding) était-elle clairement structurée ?",
        "3. L'entretien de recrutement reflétait-il la culture d'entreprise ?",
    ]),
    ("Gestion des compétences", [
        "4. Avez-vous accès à des formations adaptées à votre poste ?",
        "5. Vos compétences sont-elles régulièrement évaluées ?",
        "6. Vos rôles et responsabilités sont-ils clairement définis ?",
    ]),
    ("Évaluation de la performance", [
        "7. Vos objectifs sont-ils clairement établis ?",
        "8. Recevez-vous un retour constructif au moins une fois par an ?",
        "9. Le processus d'évaluation vous semble-t-il équitable ?",
    ]),
    ("Rémunération", [
        "10. Votre salaire est-il équitable par rapport à la région/secteur ?",
        "11. Les critères d'augmentation sont-ils transparents ?",
        "12. Les avantages non financiers sont-ils satisfaisants ?",
    ]),
    ("Qualité de vie au travail (QVT)", [
        "13. Bénéficiez-vous d'une flexibilité suffisante (horaires/télétravail) ?",
        "14. Votre espace de travail est-il sûr et ergonomique ?",
        "15. La charge de travail est-elle gérable ?",
    ]),
    ("Droit du travail", [
        "16. Êtes-vous informé(e) de vos droits en tant qu'employé(e) ?",
        "17. Les changements contractuels vous sont-ils communiqués à l'avance ?",
        "18. Vos droits au repos légal sont-ils respectés ?",
    ]),
    ("Thématiques transverses", [
        "19. La communication stratégique de l'entreprise est-elle claire ?",
        "20. L'environnement de travail est-il inclusif et respectueux ?",
        "21. Les outils numériques mis à disposition sont-ils adéquats ?",
    ]),
]


def generate_survey_pdf(output_path: Path | None = None) -> Path:
    """Generate the survey PDF and return its path."""
    from fpdf import FPDF

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = ASSETS_DIR / "fiches_pratiques.pdf"

    if output_path.exists():
        return output_path

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "HR Valais - Fiches pratiques RH", ln=True, align="C", fill=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6,
             "Sondage annuel de bien-être et d'engagement — Répondez sur une échelle de 1 (pas du tout) à 5 (tout à fait).",
             ln=True, align="C")
    pdf.ln(6)

    PILLAR_COLORS = [
        (79, 142, 247), (34, 197, 94), (251, 191, 36),
        (239, 68, 68), (168, 85, 247), (20, 184, 166), (249, 115, 22),
    ]

    for idx, (pillar, questions) in enumerate(SURVEY_STRUCTURE):
        r, g, b = PILLAR_COLORS[idx % len(PILLAR_COLORS)]
        # Pillar header
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, f"  Pilier {idx + 1} : {pillar}", ln=True, fill=True)
        pdf.ln(2)

        # Questions
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 11)
        for q in questions:
            pdf.multi_cell(0, 7, f"  {q}", border=0)
            # Likert scale row
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 6, "     1 = Pas du tout   2 = Plutôt non   3 = Neutre   4 = Plutôt oui   5 = Tout à fait",
                     ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.ln(1)
        pdf.ln(4)

    # Footer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, "© 2026 HR Valais — Prototype confidentiel. Ne pas diffuser.", align="C")

    pdf.output(str(output_path))
    return output_path
