"""Export the final Markdown report as a styled, self-contained PDF."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SOURCE = PROJECT_ROOT / "reports" / "final_report.md"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "final_report.pdf"


def _register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("ProjectSans", regular))
        pdfmetrics.registerFont(TTFont("ProjectSansBold", bold))
        return "ProjectSans", "ProjectSansBold"
    return "Helvetica", "Helvetica-Bold"


def _inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = text.replace("+/-", "&#177;")
    return text


def _page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(0.65 * inch, 0.62 * inch, A4[0] - 0.65 * inch, 0.62 * inch)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.setFont(doc.body_font, 8)
    canvas.drawString(0.68 * inch, 0.4 * inch, "CSC-355 NLP Design Project | Namal University Mianwali")
    canvas.drawRightString(A4[0] - 0.68 * inch, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _styles(body_font: str, bold_font: str):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", fontName=bold_font, fontSize=24, leading=29, textColor=colors.HexColor("#0F172A"), alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="CoverMeta", fontName=body_font, fontSize=11, leading=17, textColor=colors.HexColor("#334155"), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="H1x", fontName=bold_font, fontSize=17, leading=21, textColor=colors.HexColor("#0F4C5C"), spaceBefore=10, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2x", fontName=bold_font, fontSize=13, leading=17, textColor=colors.HexColor("#1E6472"), spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="H3x", fontName=bold_font, fontSize=11, leading=15, textColor=colors.HexColor("#334155"), spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="Bodyx", fontName=body_font, fontSize=9.2, leading=13.5, textColor=colors.HexColor("#172033"), alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(name="Bulletx", fontName=body_font, fontSize=9.2, leading=13, leftIndent=14, firstLineIndent=-7, spaceAfter=3))
    styles.add(ParagraphStyle(name="Captionx", fontName=body_font, fontSize=8, leading=11, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="TableCellx", fontName=body_font, fontSize=7.5, leading=9.5))
    styles.add(ParagraphStyle(name="TableHeadx", fontName=bold_font, fontSize=7.5, leading=9.5, textColor=colors.white))
    return styles


def _table(lines: list[str], styles) -> Table:
    rows = []
    for index, line in enumerate(lines):
        values = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if index == 1 and all(set(value) <= {"-", ":"} for value in values):
            continue
        style = styles["TableHeadx"] if not rows else styles["TableCellx"]
        rows.append([Paragraph(_inline(value), style) for value in values])
    usable = A4[0] - 1.3 * inch
    table = Table(rows, colWidths=[usable / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F4C5C")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _markdown_story(markdown: str, styles) -> list:
    story: list = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.extend([_table(table_lines, styles), Spacer(1, 8)])
            continue
        if line.startswith("### "):
            story.append(Paragraph(_inline(line[4:]), styles["H3x"]))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), styles["H1x"]))
        elif line.startswith("# "):
            pass
        elif re.match(r"^\d+\.\s", line):
            story.append(Paragraph(_inline(line), styles["Bulletx"]))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + _inline(line[2:]), styles["Bulletx"]))
        else:
            paragraph = [line]
            while index + 1 < len(lines):
                candidate = lines[index + 1].strip()
                if not candidate or candidate.startswith(("#", "|", "- ")) or re.match(r"^\d+\.\s", candidate):
                    break
                paragraph.append(candidate)
                index += 1
            story.append(Paragraph(_inline(" ".join(paragraph)), styles["Bodyx"]))
        index += 1
    return story


def export_pdf() -> Path:
    body_font, bold_font = _register_fonts()
    styles = _styles(body_font, bold_font)
    markdown = REPORT_SOURCE.read_text(encoding="utf-8")
    doc = BaseDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.78 * inch,
        title="Group-Safe Urdu Tweet Sentiment and Emotion Classification",
        author="M. Raqib Hayat and Abu Bakar",
    )
    doc.body_font = body_font
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=_page))

    story = [
        Spacer(1, 0.8 * inch),
        Paragraph("Group-Safe Urdu Tweet Sentiment and Emotion Classification", styles["CoverTitle"]),
        Spacer(1, 0.18 * inch),
        Paragraph("CSC-355 Natural Language Processing Design Project", styles["CoverMeta"]),
        Paragraph("Department of Computer Science, Namal University Mianwali", styles["CoverMeta"]),
        Spacer(1, 0.25 * inch),
        Paragraph("M. Raqib Hayat (NUM-BSCS-2022-40)<br/>Abu Bakar (NUM-BSCS-2022-41)", styles["CoverMeta"]),
        Spacer(1, 0.18 * inch),
        Paragraph("Instructor: Dr. Muzamil Ahmed<br/>Final verified benchmark: June 15, 2026", styles["CoverMeta"]),
        Spacer(1, 0.55 * inch),
        Paragraph("36 official runs | 2 tasks | 8 models | validation-only selection", styles["CoverMeta"]),
        PageBreak(),
    ]
    story.extend(_markdown_story(markdown, styles))
    story.append(PageBreak())
    story.append(Paragraph("Official Figures", styles["H1x"]))
    figure_dir = PROJECT_ROOT / "reports" / "figures"
    figures = [
        ("sentiment_class_distribution.png", "Sentiment class distribution after group-safe filtering"),
        ("emotion_class_distribution.png", "Emotion class distribution after group-safe filtering"),
        ("sentiment_validation_macro_f1.png", "Sentiment model ranking by validation macro-F1"),
        ("emotion_validation_macro_f1.png", "Emotion model ranking by validation macro-F1"),
        ("sentiment_confusion_matrix.png", "Selected sentiment Linear SVM confusion matrix"),
        ("emotion_confusion_matrix.png", "Selected emotion Linear SVM confusion matrix"),
    ]
    for filename, caption in figures:
        path = figure_dir / filename
        image = Image(str(path), width=6.2 * inch, height=3.55 * inch)
        image.hAlign = "CENTER"
        story.append(KeepTogether([image, Paragraph(caption, styles["Captionx"])]))
    doc.build(story)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(export_pdf())
