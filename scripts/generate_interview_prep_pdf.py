#!/usr/bin/env python3
"""Generate CareerFlow interview prep PDF from markdown source."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "careerflow_interview_prep_50qa.md"
OUTPUT = ROOT / "docs" / "careerflow_interview_prep_50qa.pdf"


class InterviewPrepPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        font_dir = Path(__file__).resolve().parent / "fonts"
        regular = font_dir / "DejaVuSans.ttf"
        bold = font_dir / "DejaVuSans-Bold.ttf"
        italic = font_dir / "DejaVuSans-Oblique.ttf"
        if regular.exists():
            if not bold.exists():
                bold.write_bytes(regular.read_bytes())
            if not italic.exists():
                italic.write_bytes(regular.read_bytes())
            self.add_font("DejaVu", "", str(regular))
            self.add_font("DejaVu", "B", str(bold))
            self.add_font("DejaVu", "I", str(italic))
            self._body_font = ("DejaVu", "")
            self._bold_font = ("DejaVu", "B")
            self._title_font = ("DejaVu", "B")
        else:
            self._body_font = ("Helvetica", "")
            self._bold_font = ("Helvetica", "B")
            self._title_font = ("Helvetica", "B")

    def footer(self):
        self.set_y(-15)
        self.set_font(*self._body_font, 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_title_page(self, title: str, subtitle: str):
        self.add_page()
        self.set_font(*self._title_font, 22)
        self.set_text_color(20, 40, 80)
        self.multi_cell(self.epw, 12, title, align="C")
        self.ln(8)
        self.set_font(*self._body_font, 12)
        self.set_text_color(60, 60, 60)
        self.multi_cell(self.epw, 7, subtitle, align="C")
        self.ln(10)
        self.set_font(*self._body_font, 10)
        self.multi_cell(
            self.epw,
            6,
            normalize_text(
                "CareersLow x CareerFlow AI Engineer - Round 2 onsite prep. "
                "Conversational answers with trade-offs, eval, latency/cost, and reroutes."
            ),
            align="C",
        )

    def add_section(self, text: str):
        self.ln(4)
        self.set_font(*self._title_font, 14)
        self.set_text_color(20, 40, 80)
        self.multi_cell(self.epw, 8, text)
        self.ln(2)

    def add_question(self, text: str):
        self.ln(3)
        self.set_font(*self._bold_font, 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(self.epw, 6, text)

    def add_answer(self, text: str):
        self.set_font(*self._body_font, 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self.epw, 5.5, text)
        self.ln(2)


def parse_markdown(content: str) -> list[tuple[str, str, list[str]]]:
    """Return list of (section, question, answer_paragraphs)."""
    blocks: list[tuple[str, str, list[str]]] = []
    section = "Introduction"
    current_q = ""
    current_a: list[str] = []

    def flush():
        nonlocal current_q, current_a
        if current_q and current_a:
            blocks.append((section, current_q, current_a))
        current_q = ""
        current_a = []

    for line in content.splitlines():
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            continue
        m = re.match(r"^\*\*(.+?)\*\*\s*$", line.strip())
        if m:
            flush()
            current_q = m.group(1).strip()
            continue
        if line.strip() == "---":
            continue
        if current_q and line.strip():
            current_a.append(line.strip())

    flush()
    return blocks


def ensure_fonts(font_dir: Path) -> None:
    """Try to load Unicode TTF fonts; Helvetica fallback uses normalize_text()."""
    font_dir.mkdir(parents=True, exist_ok=True)
    regular = font_dir / "DejaVuSans.ttf"
    if regular.exists():
        return

    system_candidates = [
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    for src in system_candidates:
        if src.exists():
            try:
                data = src.read_bytes()
                (font_dir / "DejaVuSans.ttf").write_bytes(data)
                (font_dir / "DejaVuSans-Bold.ttf").write_bytes(data)
                (font_dir / "DejaVuSans-Oblique.ttf").write_bytes(data)
                return
            except OSError:
                pass

    try:
        import urllib.request

        base = "https://cdn.jsdelivr.net/gh/dejavu-fonts/dejavu-fonts@version_2_37/ttf/"
        for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"):
            dest = font_dir / name
            urllib.request.urlretrieve(base + name, dest)
    except Exception as exc:
        print(f"Warning: no Unicode fonts ({exc}); using ASCII-safe Helvetica.", file=sys.stderr)


def normalize_text(text: str) -> str:
    """Make text safe for PDF core fonts (Helvetica) if Unicode TTF unavailable."""
    replacements = {
        "\u2014": " - ",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def build_pdf(source: Path, output: Path) -> None:
    ensure_fonts(Path(__file__).resolve().parent / "fonts")
    content = normalize_text(source.read_text(encoding="utf-8"))
    blocks = parse_markdown(content)

    pdf = InterviewPrepPDF()
    pdf.add_title_page(
        normalize_text("50 CareerFlow Interview Q&A"),
        normalize_text(
            "Conversational answers for AI Engineer Round 2 (onsite / deep dive)"
        ),
    )

    last_section = ""
    for section, question, paragraphs in blocks:
        if section != last_section:
            pdf.add_section(section)
            last_section = section
        pdf.add_question(question)
        pdf.add_answer("\n\n".join(paragraphs))

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))
    print(f"Wrote {output} ({output.stat().st_size // 1024} KB, {len(blocks)} Q&A)")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT
    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        sys.exit(1)
    build_pdf(src, out)
