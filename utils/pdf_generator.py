"""
utils/pdf_generator.py
Generates assets/fiches_pratiques.pdf  -  the HR Valais survey PDF.
Uses fpdf2. Called once on first app run if the PDF doesn't exist.
"""

from __future__ import annotations
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets"

# ---------------------------------------------------------------------------
# Canonical survey structure — 7 dimensions × N questions
# Keys are the EXACT strings used both in the survey form and in url_mapping.
# ---------------------------------------------------------------------------
SURVEY_STRUCTURE: list[tuple[str, list[str]]] = [
    ("Recrutement", [
        "Le modèle d'annonce est clairement aligné avec les missions et le profil recherché.",
        "Les canaux de diffusion des annonces sont choisis en fonction du poste à pourvoir.",
        "Les candidatures sont triées sur la base de critères de sélection définis à l'avance.",
        "Les entretiens suivent une structure commune définie pour l'ensemble de l'entreprise",
        "Les candidats sont évalués à l'aide d'une grille ou d'un système de points.",
        "La phase d'intégration des nouveaux collaborateurs comporte des étapes définies et un processus de suivi.",
    ]),
    ("Gestion des compétences", [
        "Chaque fonction de l'entreprise dispose d'un descriptif de poste formalisé et à jour (missions, responsabilités, conditions).",
        "Pour chaque poste, un profil de compétences requises est défini.",
        "L'entreprise dispose d'un catalogue ou d'un plan de formation pour développer les compétences des employés.",
        "L'entreprise suit l'évolution des compétences pour les années à venir en lien avec son domaine d'activité.",
    ]),
    ("Évaluation & Performance", [
        "Les employés bénéficient d'au moins une évaluation formelle par an.",
        "Des objectifs annuels sont définis pour les postes et formulés de manière claire et mesurable.",
        "Les managers donnent régulièrement un feedback structuré sur la performance.",
        "Les employés reçoivent une reconnaissance concrète pour leur travail.",
    ]),
    ("Rémunération", [
        "La politique salariale repose sur des principes clairs et transparents.",
        "Les salaires pratiqués dans l'entreprise sont compétitifs par rapport au marché.",
        "Les fiches/bulletins de salaire sont claires et compréhensibles.",
        "Les avantages de l'entreprise sont communiqués de manière attractive.",
    ]),
    ("Qualité de vie au travail (QVT)", [
        "L'entreprise identifie et informe les employés des principaux risques liés à la santé et à la sécurité.",
        "L'entreprise prend les mesures adéquates pour protéger ses collaborateurs.",
        "L'entreprise offre, lorsque c'est possible, une certaine flexibilité des horaires",
        "L'entreprise offre, lorsque c'est possible, une certaine flexibilité du lieu de travail",
        "L'entreprise est à l'écoute de ses collaborateurs.",
        "L'entreprise met en place des actions pour fidéliser ses collaborateurs à long terme",
    ]),
    ("Droit du travail", [
        "Les éléments essentiels du contrat de travail (fonction, salaire, taux d'activité, durée, etc.) sont clairement formalisés.",
        "Les droits et obligations liés au contrat de travail (horaires, vacances, résiliation, etc.) sont clairement expliqués et facilement accessibles aux collaborateurs.",
        "Les conditions de travail convenues dans le contrat correspondent à la réalité du poste.",
        "Les conditions du contrat de travail sont adaptées en cas de changement (fonction, salaire, taux d'activité, etc.).",
        "Les démarches à appliquer pour la fin d'un rapport de travail (résiliation, communication et certificat de travail) sont claires et communiquées.",
    ]),
    ("Thématiques transverses", [
        "Les outils numériques RH utilisés dans l'entreprise correspondent aux besoins réels de l'organisation.",
        "Lors de l'utilisation de nouvelles technologies, y compris l'intelligence artificielle, une attention particulière est portée aux questions d'éthique, de confidentialité et de protection des données.",
        "L'entreprise donne une image cohérente et positive en tant qu'employeur, autant en interne qu'en externe.",
        "Les collaborateurs connaissent les principales exigences de la loi sur la protection des données en lien avec les parties internes et externes",
    ]),
]

# ---------------------------------------------------------------------------
# URL mapping — identical keys, fiches HR Valais links
# ---------------------------------------------------------------------------
URL_MAPPING: dict[str, str] = {
    "Le modèle d'annonce est clairement aligné avec les missions et le profil recherché.":
        "https://www.hr-valais.ch/fiches-rh-pme/1-1-redaction-et-diffusion-de-l-annonce/viewdocument/48",
    "Les canaux de diffusion des annonces sont choisis en fonction du poste à pourvoir.":
        "https://www.hr-valais.ch/fiches-rh-pme/1-1-redaction-et-diffusion-de-l-annonce/viewdocument/48",
    "Les candidatures sont triées sur la base de critères de sélection définis à l'avance.":
        "https://www.hr-valais.ch/fiches-rh-pme/1-2-selection-et-entretiens/viewdocument/50",
    "Les entretiens suivent une structure commune définie pour l'ensemble de l'entreprise":
        "https://www.hr-valais.ch/fiches-rh-pme/1-2-selection-et-entretiens/viewdocument/50",
    "Les candidats sont évalués à l'aide d'une grille ou d'un système de points.":
        "https://www.hr-valais.ch/fiches-rh-pme/1-2-selection-et-entretiens/viewdocument/50",
    "La phase d'intégration des nouveaux collaborateurs comporte des étapes définies et un processus de suivi.":
        "https://www.hr-valais.ch/fiches-rh-pme/1-3-onboarding-processus-d-integration/viewdocument/53",
    "Chaque fonction de l'entreprise dispose d'un descriptif de poste formalisé et à jour (missions, responsabilités, conditions).":
        "https://www.hr-valais.ch/fiches-rh-pme/2-1-descriptif-de-poste/viewdocument/[ID]",
    "Pour chaque poste, un profil de compétences requises est défini.":
        "https://www.hr-valais.ch/fiches-rh-pme/2-2-profil-et-catalogue-de-competences/viewdocument/[ID]",
    "L'entreprise dispose d'un catalogue ou d'un plan de formation pour développer les compétences des employés.":
        "https://www.hr-valais.ch/fiches-rh-pme/2-3-developpement-des-competences/viewdocument/[ID]",
    "L'entreprise suit l'évolution des compétences pour les années à venir en lien avec son domaine d'activité.":
        "https://www.hr-valais.ch/fiches-rh-pme/2-3-developpement-des-competences/viewdocument/[ID]",
    "Les employés bénéficient d'au moins une évaluation formelle par an.":
        "https://www.hr-valais.ch/fiches-rh-pme/3-1-processus-d-evaluation-annuelle-et-entretien-de-suivi/viewdocument/[ID]",
    "Des objectifs annuels sont définis pour les postes et formulés de manière claire et mesurable.":
        "https://www.hr-valais.ch/fiches-rh-pme/3-2-definition-des-objectifs-annuels-individuels-et-collectifs/viewdocument/[ID]",
    "Les managers donnent régulièrement un feedback structuré sur la performance.":
        "https://www.hr-valais.ch/fiches-rh-pme/3-3-evaluation-feed-back-et-reconnaissance/viewdocument/[ID]",
    "Les employés reçoivent une reconnaissance concrète pour leur travail.":
        "https://www.hr-valais.ch/fiches-rh-pme/3-3-evaluation-feed-back-et-reconnaissance/viewdocument/[ID]",
    "La politique salariale repose sur des principes clairs et transparents.":
        "https://www.hr-valais.ch/fiches-rh-pme/4-1-developpement-de-la-politique-de-remuneration/viewdocument/[ID]",
    "Les salaires pratiqués dans l'entreprise sont compétitifs par rapport au marché.":
        "https://www.hr-valais.ch/fiches-rh-pme/4-1-developpement-de-la-politique-de-remuneration/viewdocument/[ID]",
    "Les fiches/bulletins de salaire sont claires et compréhensibles.":
        "https://www.hr-valais.ch/fiches-rh-pme/4-2-fiche-de-salaire-deductions-et-assurances-sociales/viewdocument/[ID]",
    "Les avantages de l'entreprise sont communiqués de manière attractive.":
        "https://www.hr-valais.ch/fiches-rh-pme/4-3-les-differents-types-de-remuneration/viewdocument/[ID]",
    "L'entreprise identifie et informe les employés des principaux risques liés à la santé et à la sécurité.":
        "https://www.hr-valais.ch/fiches-rh-pme/5-1-sante-et-securite-au-travail/viewdocument/[ID]",
    "L'entreprise prend les mesures adéquates pour protéger ses collaborateurs.":
        "https://www.hr-valais.ch/fiches-rh-pme/5-1-sante-et-securite-au-travail/viewdocument/[ID]",
    "L'entreprise offre, lorsque c'est possible, une certaine flexibilité des horaires":
        "https://www.hr-valais.ch/fiches-rh-pme/5-2-flexibilite-des-horaires-et-du-lieu-de-travail/viewdocument/[ID]",
    "L'entreprise offre, lorsque c'est possible, une certaine flexibilité du lieu de travail":
        "https://www.hr-valais.ch/fiches-rh-pme/5-2-flexibilite-des-horaires-et-du-lieu-de-travail/viewdocument/[ID]",
    "L'entreprise est à l'écoute de ses collaborateurs.":
        "https://www.hr-valais.ch/fiches-rh-pme/5-3-fidelisation/viewdocument/[ID]",
    "L'entreprise met en place des actions pour fidéliser ses collaborateurs à long terme":
        "https://www.hr-valais.ch/fiches-rh-pme/5-3-fidelisation/viewdocument/[ID]",
    "Les éléments essentiels du contrat de travail (fonction, salaire, taux d'activité, durée, etc.) sont clairement formalisés.":
        "https://www.hr-valais.ch/fiches-rh-pme/6-4-contrat-de-travail/viewdocument/[ID]",
    "Les droits et obligations liés au contrat de travail (horaires, vacances, résiliation, etc.) sont clairement expliqués et facilement accessibles aux collaborateurs.":
        "https://www.hr-valais.ch/fiches-rh-pme/6-1-les-obligations-de-l-employeur/viewdocument/[ID]",
    "Les conditions de travail convenues dans le contrat correspondent à la réalité du poste.":
        "https://www.hr-valais.ch/fiches-rh-pme/6-4-contrat-de-travail/viewdocument/[ID]",
    "Les conditions du contrat de travail sont adaptées en cas de changement (fonction, salaire, taux d'activité, etc.).":
        "https://www.hr-valais.ch/fiches-rh-pme/6-4-contrat-de-travail/viewdocument/[ID]",
    "Les démarches à appliquer pour la fin d'un rapport de travail (résiliation, communication et certificat de travail) sont claires et communiquées.":
        "https://www.hr-valais.ch/fiches-rh-pme/6-5-resiliation-des-rapports-de-travail/viewdocument/[ID]",
    "Les outils numériques RH utilisés dans l'entreprise correspondent aux besoins réels de l'organisation.":
        "https://www.hr-valais.ch/fiches-rh-pme/7-2-nouvelles-technologies-sirh/viewdocument/[ID]",
    "Lors de l'utilisation de nouvelles technologies, y compris l'intelligence artificielle, une attention particulière est portée aux questions d'éthique, de confidentialité et de protection des données.":
        "https://www.hr-valais.ch/fiches-rh-pme/7-3-nouvelles-technologies-et-intelligence-artificielle/viewdocument/[ID]",
    "L'entreprise donne une image cohérente et positive en tant qu'employeur, autant en interne qu'en externe.":
        "https://www.hr-valais.ch/fiches-rh-pme/7-4-marque-employeur-attirer-retenir-communiquer/viewdocument/[ID]",
    "Les collaborateurs connaissent les principales exigences de la loi sur la protection des données en lien avec les parties internes et externes":
        "https://www.hr-valais.ch/fiches-rh-pme/7-1-protection-des-donnees/viewdocument/[ID]",
}


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
    pdf.cell(
        0, 6,
        "Sondage annuel  -  Échelle: 0 = Je ne sais pas  |  1 = Faible  |  2 = Partiel  |  3 = Avancé  |  4 = Optimal",
        ln=True, align="C",
    )
    pdf.ln(6)

    PILLAR_COLORS = [
        (79, 142, 247), (34, 197, 94), (251, 191, 36),
        (239, 68, 68), (168, 85, 247), (20, 184, 166), (249, 115, 22),
    ]

    for idx, (pillar, questions) in enumerate(SURVEY_STRUCTURE):
        r, g, b = PILLAR_COLORS[idx % len(PILLAR_COLORS)]
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, f"  Pilier {idx + 1} : {pillar}", ln=True, fill=True)
        pdf.ln(2)

        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 11)
        for q in questions:
            pdf.multi_cell(0, 7, f"  {q}", border=0)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(
                0, 6,
                "     0 = Je ne sais pas   1 = Faible   2 = Partiel   3 = Avancé   4 = Optimal",
                ln=True,
            )
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.ln(1)
        pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, "(c) 2026 HR Valais  -  Prototype confidentiel. Ne pas diffuser.", align="C")

    pdf.output(str(output_path))
    return output_path
