#!/usr/bin/env python3
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import html

MD_PATH = "PROJECT_REPORT.md"
PDF_PATH = "PROJECT_REPORT.pdf"

SECTION_STYLE = ParagraphStyle(
    'Section', parent=getSampleStyleSheet()['Heading2'], fontSize=16, spaceAfter=8, textColor=colors.darkblue
)
SUBSECTION_STYLE = ParagraphStyle(
    'SubSection', parent=getSampleStyleSheet()['Heading3'], fontSize=13, spaceAfter=6, textColor=colors.HexColor('#333333')
)
BODY = getSampleStyleSheet()['BodyText']
BODY.leading = 16
TITLE = ParagraphStyle('Title', parent=getSampleStyleSheet()['Heading1'], fontSize=22, alignment=1, textColor=colors.HexColor('#1f77b4'))


def _escape(text: str) -> str:
    # Escape HTML special chars and strip backticks
    text = text.replace('`', '')
    text = html.escape(text, quote=False)
    return text


def simple_md_to_flowables(md_text: str):
    lines = md_text.splitlines()
    flow = []
    flow.append(Paragraph("Hybrid Deepfake Detector - Comprehensive Project Report", TITLE))
    flow.append(Spacer(1, 0.25*inch))

    in_code_block = False
    for raw in lines:
        line = raw.rstrip("\n")
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            # Skip code block content to avoid complex formatting
            continue
        s = line.strip()
        if not s:
            flow.append(Spacer(1, 0.12*inch))
            continue
        if s.startswith('# '):
            flow.append(Paragraph(_escape(s[2:].strip()), TITLE))
        elif s.startswith('## '):
            flow.append(Paragraph(_escape(s[3:].strip()), SECTION_STYLE))
        elif s.startswith('### '):
            flow.append(Paragraph(_escape(s[4:].strip()), SUBSECTION_STYLE))
        elif s.startswith('- '):
            flow.append(Paragraph("• " + _escape(s[2:].strip()), BODY))
        else:
            flow.append(Paragraph(_escape(s), BODY))
    return flow


def build_pdf(md_path=MD_PATH, pdf_path=PDF_PATH):
    if not os.path.exists(md_path):
        raise FileNotFoundError(md_path)
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = simple_md_to_flowables(md_text)
    doc.build(story)
    return pdf_path

if __name__ == "__main__":
    out = build_pdf()
    print(f"[INFO] PDF created: {out}")
