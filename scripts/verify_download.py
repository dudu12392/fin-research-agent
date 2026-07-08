"""Bulk download 10-K PDFs for all 32 companies and verify results."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.company_list import ALL_COMPANIES, SECTOR_MAP
from src.data.download_filings import download_10k_pdfs


def verify_downloads(
    tickers: list[str],
    years: int = 3,
    output_dir: str = "data/pdfs",
) -> None:
    """Download and verify 10-K PDFs for specified companies.

    Args:
        tickers: List of stock ticker symbols.
        years: Number of recent fiscal years.
        output_dir: Base directory for PDFs.
    """
    print(f"\n{'='*70}")
    print(f"📥 批量下载 {len(tickers)} 家公司最近 {years} 年的 10-K 年报")
    print(f"{'='*70}")
    for sector, names in [
        ("Technology", ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","INTC","CRM","ADBE","ORCL"]),
        ("Consumer", ["WMT","COST","HD","MCD","NKE","SBUX","TGT","LOW"]),
        ("Finance", ["JPM","BAC","GS","MS","V","MA","BRK-B"]),
        ("Healthcare", ["JNJ","PFE","UNH","ABBV","MRK","LLY"]),
    ]:
        count = sum(1 for t in tickers if t.upper() in [x.upper() for x in names])
        print(f"   {sector}: {count} 家")

    start_time = time.time()

    # Download
    paths = download_10k_pdfs(tickers, years=years, output_dir=output_dir)

    elapsed = time.time() - start_time

    # Verify
    print(f"\n{'='*70}")
    print("📋 下载验证报告")
    print(f"{'='*70}")
    print(f"{'公司':<8} {'行业':<12} {'文件数':<8}")
    print(f"{'-'*70}")

    total = 0
    success = 0
    failed_tickers: list[tuple[str, str]] = []

    for ticker in tickers:
        ticker_dir = Path(output_dir) / ticker.upper()
        sector = SECTOR_MAP.get(ticker.upper(), "Unknown")
        if not ticker_dir.exists():
            failed_tickers.append((ticker.upper(), "目录不存在"))
            continue

        files = sorted(ticker_dir.iterdir())
        file_count = 0
        for f in files:
            total += 1
            if f.stat().st_size > 0:
                success += 1
                file_count += 1

        status = "✅" if file_count == years else f"⚠️ {file_count}/{years}"
        print(f"{ticker.upper():<8} {sector:<12} {status}")

    print(f"{'-'*70}")
    print(f"\n⏱️  耗时: {elapsed/60:.1f} 分钟")
    print(f"✅ 成功: {success} 份")
    print(f"❌ 失败: {total - success} 份")

    if failed_tickers:
        print(f"\n⚠️  失败的公司:")
        for t, reason in failed_tickers:
            print(f"   {t}: {reason}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    verify_downloads(ALL_COMPANIES, years=3)
