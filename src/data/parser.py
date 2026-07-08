"""SEC filing parser & smart chunker — HTML/PDF → structured text chunks.

Parsing chain: pdfplumber → PyMuPDF → BeautifulSoup (at least 2 fallbacks).
Smart chunking preserves table boundaries and section structure.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("data.parser")

# ── Constants ─────────────────────────────────────────────────────
CHUNK_SIZE_CHARS = 2000   # ~500-800 tokens for English
OVERLAP_CHARS = 400       # ~100 tokens context overlap
MIN_CHUNK_SIZE = 200      # Minimum chars for a meaningful chunk

# 10-K section header patterns (robust to formatting variants)
SECTION_PATTERN = re.compile(
    r"^(Item\s+\d+[A-Z]?[\.\s].*?)$",
    re.MULTILINE | re.IGNORECASE,
)

# Common 10-K items for metadata extraction
KNOWN_ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# Financial table keywords for table identification
FINANCIAL_TABLE_TERMS = [
    "consolidated", "balance sheet", "statement of operations",
    "income statement", "cash flow", "statement of", "shareholders",
    "revenue", "net income", "total assets", "operating income",
    "in millions", "in thousands", "fiscal year", "year ended",
]


# ═══════════════════════════════════════════════════════════════════
# 1. parse_filing
# ═══════════════════════════════════════════════════════════════════

def parse_filing(file_path: str) -> str:
    """Parse a SEC filing document (HTML or PDF) into plain text.

    Strategy (tried in order):
      1. pdfplumber (for real PDFs, skipped for HTML)
      2. PyMuPDF / fitz (for real PDFs, skipped for HTML)
      3. BeautifulSoup (HTML / iXBRL primary, universal fallback)

    Args:
        file_path: Path to the filing document (.pdf or .htm).

    Returns:
        Extracted plain text content.
    """
    import gc

    path = Path(file_path)
    logger.info("parse_start", file=path.name)

    # ── Detect file type ─────────────────────────────────────
    is_html = _is_html_file(path)
    text = ""

    if not is_html:
        # ── Strategy 1: pdfplumber (real PDFs only) ──────────
        try:
            text = _parse_with_pdfplumber(path)
            if text and len(text.strip()) > 500:
                logger.info("parse_success", engine="pdfplumber", chars=len(text))
                gc.collect()
                return text
        except Exception as e:
            logger.debug("pdfplumber_failed", error=str(e))

        # ── Strategy 2: PyMuPDF ───────────────────────────────
        try:
            text = _parse_with_pymupdf(path)
            if text and len(text.strip()) > 500:
                logger.info("parse_success", engine="pymupdf", chars=len(text))
                gc.collect()
                return text
        except Exception as e:
            logger.debug("pymupdf_failed", error=str(e))

    # ── Strategy 3: BeautifulSoup (HTML / universal fallback) ─
    try:
        text = _parse_with_bs4(path)
        logger.info("parse_success", engine="beautifulsoup", chars=len(text))
        gc.collect()
        return text
    except Exception as e:
        logger.error("all_parsers_failed", file=path.name, error=str(e))
        # Last resort: read raw bytes as text
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            logger.info("parse_success", engine="raw_fallback", chars=len(text))
            gc.collect()
            return text
        except Exception:
            return ""

    gc.collect()
    return text


# ═══════════════════════════════════════════════════════════════════
# 2. extract_tables
# ═══════════════════════════════════════════════════════════════════

def extract_tables(file_path: str) -> list[list[list[str]]]:
    """Extract all page-level tables from a filing document.

    Uses pdfplumber for PDF files; falls back to BeautifulSoup
    table extraction for HTML/iXBRL documents.

    Args:
        file_path: Path to the filing document.

    Returns:
        Three-dimensional list: tables[table_idx][row_idx][cell_value].
    """
    path = Path(file_path)
    logger.info("table_extraction_start", file=path.name)

    tables: list[list[list[str]]] = []

    # ── Try pdfplumber first ──────────────────────────────────
    try:
        tables = _extract_tables_pdfplumber(path)
        if tables:
            logger.info("tables_extracted", engine="pdfplumber", count=len(tables))
            return tables
    except Exception as e:
        logger.debug("pdfplumber_tables_failed", error=str(e))

    # ── Fallback: BeautifulSoup table extraction ──────────────
    try:
        tables = _extract_tables_bs4(path)
        logger.info("tables_extracted", engine="beautifulsoup", count=len(tables))
        return tables
    except Exception as e:
        logger.error("table_extraction_failed", file=path.name, error=str(e))

    return tables


# ═══════════════════════════════════════════════════════════════════
# 3. chunk_filing
# ═══════════════════════════════════════════════════════════════════

def chunk_filing(
    text: str,
    tables: list[list[list[str]]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split parsed filing text into smart, overlapping chunks.

    Rules:
      1. Split on section headers (Item X. ...).
      2. Keep financial tables as independent, uncut semantic units.
      3. Text sections: ~2000 chars per chunk with ~400 char overlap.
      4. Each chunk carries full metadata.

    Args:
        text: Full parsed text from parse_filing().
        tables: Extracted tables from extract_tables().
        metadata: Dict with keys: company, year, form.

    Returns:
        List of chunk dicts: {"content", "metadata": {..., section, has_table,
        page, chunk_index}}.
    """
    logger.info(
        "chunk_start",
        company=metadata.get("company"),
        year=metadata.get("year"),
        text_length=len(text),
        table_count=len(tables),
    )

    chunks: list[dict[str, Any]] = []
    sections = _split_by_sections(text)

    # ── Defense: max iteration guard ──────────────────────────
    MAX_SECTION_ITER = 5000
    section_iter = 0

    chunk_index = 0
    for section_title, section_text in sections:
        section_iter += 1
        if section_iter > MAX_SECTION_ITER:
            raise RuntimeError(
                f"chunk_filing section loop exceeded {MAX_SECTION_ITER} iterations — aborting"
            )
        # Detect if this section title suggests it contains tables
        has_tables = (
            "financial statement" in section_title.lower()
            or "item 8" in section_title.lower()
            or "item 15" in section_title.lower()
        )

        item_number = _extract_item_number(section_title)
        section_label = KNOWN_ITEMS.get(item_number, section_title.strip())

        if has_tables and tables:
            # Emit each table as a standalone chunk
            for table_idx, table_data in enumerate(tables):
                table_content = _format_table_as_text(table_data)
                if len(table_content) < 50:
                    continue
                chunk = {
                    "content": table_content,
                    "metadata": {
                        **metadata,
                        "section": section_label,
                        "has_table": True,
                        "page": 0,
                        "chunk_index": chunk_index,
                        "table_index": table_idx,
                    },
                }
                chunks.append(chunk)
                chunk_index += 1
        else:
            # Text section — split into overlapping chunks
            text_chunks = _split_text_with_overlap(
                section_text, CHUNK_SIZE_CHARS, OVERLAP_CHARS
            )
            for tc in text_chunks:
                chunk = {
                    "content": tc,
                    "metadata": {
                        **metadata,
                        "section": section_label,
                        "has_table": False,
                        "page": 0,
                        "chunk_index": chunk_index,
                    },
                }
                chunks.append(chunk)
                chunk_index += 1

    logger.info(
        "chunk_complete",
        company=metadata.get("company"),
        total_chunks=len(chunks),
    )
    return chunks


# ═══════════════════════════════════════════════════════════════════
# 4. process_all_filings
# ═══════════════════════════════════════════════════════════════════

def process_all_filings(
    pdf_dir: str = "data/pdfs",
    output_dir: str = "data/chunks",
) -> list[dict[str, Any]]:
    """Batch-process all downloaded filings: parse → extract tables → chunk → save.

    Args:
        pdf_dir: Root directory containing {ticker}/{year}_10k.pdf files.
        output_dir: Directory to save chunked JSON output.

    Returns:
        List of processing summaries: {company, year, file, chunks, tables, chars}.
    """
    base = Path(pdf_dir)
    out_base = Path(output_dir)
    summaries: list[dict[str, Any]] = []

    if not base.exists():
        logger.error("pdf_dir_not_found", path=str(base))
        return summaries

    for ticker_dir in sorted(base.iterdir()):
        if not ticker_dir.is_dir():
            continue

        company = ticker_dir.name
        logger.info("processing_company", company=company)

        for pdf_file in sorted(ticker_dir.glob("*_10k.pdf")):
            # Extract year from filename
            year_match = re.match(r"(\d{4})_10k\.pdf", pdf_file.name)
            year = year_match.group(1) if year_match else "unknown"

            logger.info("processing_file", company=company, year=year, file=pdf_file.name)

            try:
                # Free memory before each file
                import gc
                gc.collect()

                # Step 1: Parse
                text = parse_filing(str(pdf_file))
                if not text or len(text.strip()) < 500:
                    logger.warning("empty_parse_result", company=company, year=year)
                    continue

                # Step 2: Extract tables
                tables = extract_tables(str(pdf_file))

                # Step 3: Chunk
                metadata = {
                    "company": company,
                    "year": year,
                    "form": "10-K",
                }
                chunks = chunk_filing(text, tables, metadata)

                # Step 4: Save chunks
                out_path = out_base / company / f"{year}_10k_chunks.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)

                summary = {
                    "company": company,
                    "year": year,
                    "file": pdf_file.name,
                    "chunks": len(chunks),
                    "tables": len(tables),
                    "chars": len(text),
                }
                summaries.append(summary)

                logger.info(
                    "filing_processed",
                    company=company,
                    year=year,
                    chunks=len(chunks),
                    tables=len(tables),
                    chars=len(text),
                )

            except Exception as e:
                logger.error(
                    "processing_failed",
                    company=company,
                    year=year,
                    error=str(e),
                )

    logger.info("batch_complete", total_files=len(summaries))
    return summaries


# ═══════════════════════════════════════════════════════════════════
# Internal: PDF/HTML parsers
# ═══════════════════════════════════════════════════════════════════

def _is_html_file(path: Path) -> bool:
    """Detect if file is HTML/iXBRL by reading first bytes."""
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:200].strip().lower()
        return head.startswith("<html") or head.startswith("<?xml")
    except Exception:
        return False


def _parse_with_pdfplumber(path: Path) -> str:
    """Extract text using pdfplumber."""
    import pdfplumber  # type: ignore[import-untyped]

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n\n".join(pages)


def _parse_with_pymupdf(path: Path) -> str:
    """Extract text using PyMuPDF (fitz)."""
    import fitz  # type: ignore[import-untyped]

    pages: list[str] = []
    doc = fitz.open(str(path))
    try:
        for page in doc:
            page_text = page.get_text()
            if page_text:
                pages.append(page_text)
    finally:
        doc.close()
    return "\n\n".join(pages)


def _parse_with_bs4(path: Path) -> str:
    """Extract visible text from HTML/iXBRL using BeautifulSoup + SoupStrainer.

    Uses memory-efficient strategy:
      1. String-replace heavy XBRL blocks before parsing (avoids regex backtracking).
      2. SoupStrainer to parse only <body> (skips head/metadata).
      3. Built-in html.parser (lighter than lxml).
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")

    # ── Pre-filter heavy XBRL blocks with simple string ops ───
    raw = _strip_xbrl_blocks(raw)

    # ── Parse only body content ───────────────────────────────
    from bs4 import SoupStrainer

    strainer = SoupStrainer("body")
    soup = BeautifulSoup(raw, "html.parser", parse_only=strainer)

    # Remove leftover display:none divs
    for div in soup.find_all("div", style=lambda s: s and "display:none" in s.replace(" ", "")):
        div.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)

    return text


def _strip_xbrl_blocks(raw: str) -> str:
    """Remove heavy iXBRL metadata blocks using fast string operations."""
    # Remove ix:hidden sections (largest blocks)
    while "<ix:hidden>" in raw:
        start = raw.find("<ix:hidden>")
        end = raw.find("</ix:hidden>", start)
        if end == -1:
            break
        raw = raw[:start] + raw[end + len("</ix:hidden>"):]

    # Remove ix:resources (large XBRL context definitions)
    while "<ix:resources>" in raw:
        start = raw.find("<ix:resources>")
        end = raw.find("</ix:resources>", start)
        if end == -1:
            break
        raw = raw[:start] + raw[end + len("</ix:resources>"):]

    # Remove ix:references (schema refs)
    while "<ix:references>" in raw:
        start = raw.find("<ix:references>")
        end = raw.find("</ix:references>", start)
        if end == -1:
            break
        raw = raw[:start] + raw[end + len("</ix:references>"):]

    return raw


# ═══════════════════════════════════════════════════════════════════
# Internal: Table extraction
# ═══════════════════════════════════════════════════════════════════

def _extract_tables_pdfplumber(path: Path) -> list[list[list[str]]]:
    """Extract tables using pdfplumber."""
    import pdfplumber  # type: ignore[import-untyped]

    all_tables: list[list[list[str]]] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables()
            for table in page_tables:
                if not table:
                    continue
                # Filter out empty rows
                cleaned: list[list[str]] = []
                for row in table:
                    if row is None:
                        continue
                    cells = [_clean_cell(c) for c in row]
                    if any(c for c in cells):
                        cleaned.append(cells)
                if cleaned:
                    all_tables.append(cleaned)
    return all_tables


def _extract_tables_bs4(path: Path) -> list[list[list[str]]]:
    """Extract tables from HTML using BeautifulSoup.

    Only returns tables that appear to contain financial data
    (detected via keyword matching in header rows).
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _strip_xbrl_blocks(raw)

    from bs4 import SoupStrainer

    strainer = SoupStrainer(["table", "tr", "td", "th"])
    soup = BeautifulSoup(raw, "html.parser", parse_only=strainer)

    all_tables: list[list[list[str]]] = []
    for table_elem in soup.find_all("table"):
        rows = table_elem.find_all("tr")
        if len(rows) < 2:  # Skip trivial tables
            continue

        parsed_rows: list[list[str]] = []
        for tr in rows:
            cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if any(c for c in cells):
                parsed_rows.append(cells)

        if not parsed_rows:
            continue

        # Quick financial table heuristic
        header_text = " ".join(parsed_rows[0]).lower() if parsed_rows else ""
        is_financial = any(
            kw in header_text for kw in FINANCIAL_TABLE_TERMS
        )

        # Accept tables with >1 row and some numeric cells
        numeric_count = sum(
            1 for row in parsed_rows
            for cell in row
            if re.search(r"\d", cell)
        )
        if is_financial or (numeric_count >= 3 and len(parsed_rows) >= 3):
            all_tables.append(parsed_rows)

    return all_tables


def _clean_cell(value: Any) -> str:
    """Normalize a table cell value to string."""
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ")


# ═══════════════════════════════════════════════════════════════════
# Internal: Chunking helpers
# ═══════════════════════════════════════════════════════════════════

def _split_by_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_title, section_body) pairs.

    Handles duplicate section headers by only keeping the first occurrence
    of each major section (TOC references are skipped).
    """
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [("Full Document", text)]

    sections: list[tuple[str, str]] = []
    seen_positions: set[int] = set()

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Skip TOC entries (usually at the beginning, short text after)
        body = text[start:end].strip()
        if len(body) < 200 and i < 10:
            continue

        # Deduplicate: keep first occurrence only
        item_num = _extract_item_number(title)
        if item_num and item_num in seen_positions:
            continue
        seen_positions.add(item_num)

        sections.append((title, body))

    # If no real sections found, return whole doc
    if not sections:
        return [("Full Document", text)]

    return sections


def _extract_item_number(section_title: str) -> str:
    """Extract item number like '1A' from 'Item 1A. Risk Factors'."""
    m = re.match(r"Item\s+([\d]+[A-Z]?)", section_title, re.IGNORECASE)
    return m.group(1) if m else ""


def _split_text_with_overlap(
    text: str, chunk_size: int = 500, overlap: int = 100
) -> list[str]:
    """Split long text into chunks with overlap between adjacent chunks.

    Guaranteed to never loop infinitely — three exit conditions guard
    against all pathological cases.

    Args:
        text: Input text to split.
        chunk_size: Target chars per chunk.
        overlap: Number of overlapping chars between adjacent chunks.

    Returns:
        List of chunk strings.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    iteration = 0
    MAX_ITER = 10000

    while start < len(text):
        iteration += 1
        if iteration > MAX_ITER:
            raise RuntimeError(
                f"_split_text_with_overlap exceeded {MAX_ITER} iterations — aborting"
            )

        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Calculate next starting position
        next_start = end - overlap

        # Three exit conditions:
        # 1. Reached end of text
        # 2. No forward progress (prevents infinite loop)
        # 3. next_start beyond text boundary
        if end >= len(text) or next_start <= start or next_start >= len(text):
            break

        start = next_start

    return chunks


def _format_table_as_text(table: list[list[str]]) -> str:
    """Format a 2D table as readable plain text."""
    if not table:
        return ""

    lines: list[str] = []
    # Calculate column widths
    col_widths: list[int] = []
    for row in table:
        for i, cell in enumerate(row):
            cell_len = len(cell) if cell else 0
            if i >= len(col_widths):
                col_widths.append(cell_len)
            else:
                col_widths[i] = max(col_widths[i], cell_len)

    for row in table:
        padded = [
            (cell or "").ljust(col_widths[i])
            for i, cell in enumerate(row)
            if i < len(col_widths)
        ]
        lines.append(" | ".join(padded))

    return "\n".join(lines)
