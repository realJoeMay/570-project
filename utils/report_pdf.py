"""Convert generated Markdown reports to PDFs without external executables."""

from html import escape
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import reportlab
from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, TableStyle


def _register_fonts():
    """Use ReportLab's bundled Unicode fonts on every platform."""
    font_dir = Path(reportlab.__file__).parent / "fonts"
    for name, filename in [
        ("Vera", "Vera.ttf"), ("Vera-Bold", "VeraBd.ttf"),
        ("Vera-Italic", "VeraIt.ttf"), ("Vera-BoldItalic", "VeraBI.ttf"),
    ]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        "Vera", normal="Vera", bold="Vera-Bold",
        italic="Vera-Italic", boldItalic="Vera-BoldItalic",
    )


def _inline(tokens):
    """Translate parsed inline Markdown into ReportLab paragraph markup."""
    parts = []
    for token in tokens:
        if token.type in {"text", "code_inline"}:
            parts.append(escape(token.content))
        elif token.type in {"softbreak", "hardbreak"}:
            parts.append("<br/>" if token.type == "hardbreak" else " ")
        elif token.type in {"strong_open", "strong_close", "em_open", "em_close"}:
            tag = "b" if token.type.startswith("strong") else "i"
            parts.append(f"<{('/' if token.nesting == -1 else '')}{tag}>")
        elif token.type == "link_open":
            url = token.attrGet("href")
            parsed = urlsplit(url)
            if not parsed.scheme and not parsed.netloc and parsed.path.endswith(".md"):
                url = urlunsplit(parsed._replace(path=str(Path(parsed.path).with_suffix(".pdf"))))
            parts.append(f'<link href="{escape(url, quote=True)}" color="#176C78">')
        elif token.type == "link_close":
            parts.append("</link>")
        else:
            raise ValueError(f"Unsupported inline Markdown: {token.type}")
    return "".join(parts)


def markdown_report_to_pdf(source, destination):
    """Render a generated report, preserving tables, emphasis, and links.

    Local Markdown links become PDF links so the converted index works within
    the output folder. Only local text is read; source websites are not fetched.
    Unsupported Markdown raises an error instead of silently dropping content.
    """
    source, destination = Path(source), Path(destination)
    _register_fonts()
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "Vera"
        style.fontSize = 9
        style.leading = 13
    for name, size in [("Heading1", 19), ("Heading2", 13), ("Heading3", 11)]:
        styles[name].fontName = "Vera-Bold"
        styles[name].fontSize = size
        styles[name].leading = size + 5
    cell_style = ParagraphStyle(
        "ReportCell", parent=styles["BodyText"], fontSize=8, leading=11,
        spaceBefore=0, spaceAfter=0, splitLongWords=True,
    )
    tokens = MarkdownIt("commonmark").enable("table").parse(source.read_text(encoding="utf-8"))
    story, rows, row = [], None, None
    heading = None
    in_list_item = False
    for token in tokens:
        kind = token.type
        if kind == "table_open":
            rows = []
        elif kind == "tr_open":
            row = []
        elif kind == "inline":
            content = _inline(token.children or [])
            if rows is not None:
                row.append(Paragraph(content, cell_style))
            else:
                style = styles[heading or "BodyText"]
                story.append(Paragraph(content, style, bulletText="-" if in_list_item else None))
        elif kind == "tr_close":
            rows.append(row)
        elif kind == "table_close":
            column_count = len(rows[0])
            # Give the source-evidence column more room in four-column tables.
            weights = [0.29, 0.23, 0.10, 0.38] if column_count == 4 else [1 / column_count] * column_count
            table = LongTable(
                rows, colWidths=[504 * weight for weight in weights],
                repeatRows=1, hAlign="LEFT", spaceBefore=8, spaceAfter=12,
                splitInRow=1,
            )
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#799699")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(table)
            rows = None
        elif kind == "heading_open":
            heading = f"Heading{min(int(token.tag[1:]), 3)}"
        elif kind == "heading_close":
            heading = None
        elif kind == "list_item_open":
            in_list_item = True
        elif kind == "list_item_close":
            in_list_item = False
        elif kind not in {
            "paragraph_open", "paragraph_close", "bullet_list_open", "bullet_list_close",
            "ordered_list_open", "ordered_list_close", "thead_open", "thead_close",
            "tbody_open", "tbody_close", "th_open", "th_close", "td_open", "td_close",
        }:
            raise ValueError(f"Unsupported Markdown block: {kind}")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Vera", 8)
        canvas.drawRightString(letter[0] - 54, 28, f"Page {doc.page}")
        canvas.restoreState()

    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(destination), pagesize=letter, leftMargin=54, rightMargin=54,
        topMargin=48, bottomMargin=48, title=source.stem.replace("_", " "),
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return destination


def convert_markdown_reports(report_dir, pdf_dir=None):
    """Convert all top-level .md files, including index.md; return manifest rows.

    Reruns replace matching PDFs, leave Markdown unchanged, and retain unrelated
    output files. Failures propagate with the source filename for diagnosis.
    """
    report_dir = Path(report_dir)
    sources = sorted(report_dir.glob("*.md"))
    if not sources:
        raise FileNotFoundError(f"No Markdown reports found in {report_dir}")
    pdf_dir = Path(pdf_dir) if pdf_dir is not None else report_dir / "pdf"
    manifest = []
    for source in sources:
        destination = pdf_dir / source.with_suffix(".pdf").name
        try:
            markdown_report_to_pdf(source, destination)
        except Exception as exc:
            raise RuntimeError(f"PDF conversion failed for {source}") from exc
        manifest.append({"report_path": source.as_posix(), "pdf_path": destination.as_posix()})
    return manifest
