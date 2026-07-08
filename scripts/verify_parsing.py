"""Verify parsing and chunking on ALL downloaded 10-K filings.

Processes ALL filings under data/pdfs/, reports per-file statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.parser import process_all_filings


def verify_parsing() -> None:
    """Run the full parse pipeline and print a summary report."""
    print(f"\n{'='*70}")
    print("📊 批量解析 & 智能分块")
    print(f"{'='*70}\n")

    summaries = process_all_filings("data/pdfs", "data/chunks")

    print(f"\n{'='*70}")
    print(f"{'公司':<8} {'年份':<6} {'块数':<8} {'表格':<8} {'字符数':<12}")
    print(f"{'-'*70}")

    total_chunks = 0
    total_tables = 0
    total_chars = 0
    failed: list[str] = []

    for s in summaries:
        print(
            f"{s['company']:<8} "
            f"{s['year']:<6} "
            f"{s['chunks']:<8} "
            f"{s['tables']:<8} "
            f"{s['chars']:>10,d}"
        )
        total_chunks += s["chunks"]
        total_tables += s["tables"]
        total_chars += s["chars"]

    print(f"{'-'*70}")
    print(f"{'合计':<8} {len(summaries)}份{'':>2} {total_chunks:<8} {total_tables:<8} {total_chars:>10,d}")
    print(f"{'='*70}")

    # Verify chunk files
    chunks_dir = Path("data/chunks")
    chunk_files = list(chunks_dir.rglob("*_10k_chunks.json"))
    print(f"\n✅ 生成 {len(chunk_files)} 个 chunk JSON 文件")

    if failed:
        print(f"\n❌ 失败: {len(failed)} 份")
        for f in failed:
            print(f"   {f}")

    print(f"\n🎉 总块数: {total_chunks}, 总字符数: {total_chars:,}")


if __name__ == "__main__":
    verify_parsing()
