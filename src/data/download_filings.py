"""Download 10-K annual report documents from SEC EDGAR.

Downloads filings for specified companies and saves them as PDF-equivalent files.
Modern iXBRL filings (HTML) are downloaded when PDF is unavailable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from edgar import Company, set_identity
from edgar.httprequests import download_file
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.logging import get_logger, setup_logging

# ── Identity ──────────────────────────────────────────────────────
set_identity(os.getenv("SEC_USER_AGENT", "Liang Chang 18845976355@163.com"))

setup_logging()
logger = get_logger("data.download_filings")

# ── Constants ─────────────────────────────────────────────────────
DEFAULT_WAIT_SECONDS = 0.5
DOWNLOAD_TIMEOUT = 60


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((OSError, IOError, TimeoutError)),
    reraise=True,
)
def _download_with_retry(url: str, output_path: Path) -> bool:
    """Download a file with retry logic.

    Args:
        url: Source URL to download.
        output_path: Local path to save the file.

    Returns:
        True on success, raises exception on failure.
    """
    logger.info("downloading", url=url)
    content = download_file(url, as_text=False)

    if content is None:
        raise OSError(f"Empty response from {url}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))

    file_size = output_path.stat().st_size
    logger.info("download_complete", path=str(output_path), size_bytes=file_size)
    return True


def _build_pdf_url(cik: int, accession_number: str) -> str:
    """Construct the standard SEC PDF URL for a filing.

    Args:
        cik: Company CIK number.
        accession_number: Filing accession number (with dashes).

    Returns:
        Full URL to the potential PDF file.
    """
    acc_no_dashes = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}"
        f"/{acc_no_dashes}/{accession_number}.pdf"
    )


def _get_filing_year(filing: Any) -> int:
    """Extract the year from a filing's date.

    Args:
        filing: An edgartools EntityFiling object.

    Returns:
        Four-digit year integer.
    """
    filing_date = getattr(filing, "filing_date", None)
    if filing_date is not None:
        try:
            return int(str(filing_date)[:4])
        except (TypeError, ValueError):
            pass
    return 0


def download_10k_pdfs(
    tickers: list[str],
    years: int = 3,
    output_dir: str = "data/pdfs",
) -> list[str]:
    """Download 10-K annual reports for multiple companies.

    Downloads the latest N years of 10-K filings for each ticker.
    Tries PDF first; falls back to primary HTML document for modern iXBRL filings.

    Args:
        tickers: List of stock ticker symbols (e.g. ['AAPL', 'MSFT']).
        years: Number of recent fiscal years to download (default 3).
        output_dir: Base directory for saving PDF files.

    Returns:
        List of successfully downloaded file paths (as strings).
    """
    base_dir = Path(output_dir)
    downloaded: list[str] = []

    logger.info(
        "download_start",
        tickers=tickers,
        years=years,
        output_dir=output_dir,
    )

    for ticker in tickers:
        ticker_upper = ticker.upper()
        logger.info("processing_ticker", ticker=ticker_upper)

        try:
            company = Company(ticker_upper)
        except Exception as e:
            logger.error("company_lookup_failed", ticker=ticker_upper, error=str(e))
            continue

        try:
            filings = company.get_filings(form="10-K")
        except Exception as e:
            logger.error("filings_lookup_failed", ticker=ticker_upper, error=str(e))
            continue

        # Download latest N filings
        downloaded_count = 0
        for filing in filings.latest(years):
            cik = getattr(filing, "cik", 0)
            accession = getattr(filing, "accession_number", "")
            filing_year = _get_filing_year(filing)

            if filing_year == 0:
                logger.warning("unknown_filing_year", ticker=ticker_upper)
                continue

            # Output path
            ticker_dir = base_dir / ticker_upper
            ticker_dir.mkdir(parents=True, exist_ok=True)
            output_path = ticker_dir / f"{filing_year}_10k.pdf"

            logger.info(
                "attempting_download",
                ticker=ticker_upper,
                year=filing_year,
                accession=accession,
            )

            try:
                # Strategy 1: Try standard PDF URL
                pdf_url = _build_pdf_url(cik, accession)
                _download_with_retry(pdf_url, output_path)
                downloaded.append(str(output_path))
                downloaded_count += 1
                logger.info(
                    "pdf_downloaded",
                    ticker=ticker_upper,
                    year=filing_year,
                    path=str(output_path),
                )
            except Exception:
                # Strategy 2: Fallback — download primary HTML document
                logger.info(
                    "pdf_unavailable_trying_html",
                    ticker=ticker_upper,
                    year=filing_year,
                )
                try:
                    primary_docs = filing.primary_documents
                    if primary_docs and len(primary_docs) > 0:
                        doc_url = primary_docs[0].url
                        _download_with_retry(doc_url, output_path)
                        downloaded.append(str(output_path))
                        downloaded_count += 1
                        logger.info(
                            "html_downloaded_as_pdf",
                            ticker=ticker_upper,
                            year=filing_year,
                            path=str(output_path),
                        )
                    else:
                        logger.error(
                            "no_documents_found",
                            ticker=ticker_upper,
                            year=filing_year,
                        )
                except Exception as html_err:
                    logger.error(
                        "download_failed",
                        ticker=ticker_upper,
                        year=filing_year,
                        error=str(html_err),
                    )

            # Rate limiting
            time.sleep(DEFAULT_WAIT_SECONDS)

        logger.info(
            "ticker_complete",
            ticker=ticker_upper,
            files_downloaded=downloaded_count,
        )

    logger.info(
        "download_finished",
        total_files=len(downloaded),
        paths=downloaded,
    )
    return downloaded


# ═══════════════════════════════════════════════════════════════════
# CLI entrypoint
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m data.download_filings AAPL MSFT NVDA [--years 3]")
        sys.exit(1)

    args = sys.argv[1:]
    num_years = 3
    if "--years" in args:
        idx = args.index("--years")
        num_years = int(args[idx + 1])
        args = args[:idx] + args[idx + 2 :]

    result = download_10k_pdfs(args, years=num_years)
    print(f"\n✅ Downloaded {len(result)} files:")
    for p in result:
        print(f"   {p}")
