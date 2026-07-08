"""Vectorize all chunked 10-K data with ChromaDB and run test queries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_vector_db(chunks_dir: str = "data/chunks", db_dir: str = "data/vector_db"):
    """Read all chunk JSONs, build ChromaDB collection, run test queries."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    print(f"\n{'='*70}")
    print("🔢 向量化全量数据")
    print(f"{'='*70}\n")

    # ── Load all chunks ────────────────────────────────────────
    base = Path(chunks_dir)
    all_chunks: list[dict] = []
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metadatas: list[dict] = []

    for json_file in sorted(base.rglob("*_10k_chunks.json")):
        with open(json_file, encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            content = chunk.get("content", "")
            if not content or len(content.strip()) < 50:
                continue

            chunk_id = (
                f"{meta.get('company','?')}_{meta.get('year','?')}"
                f"_{meta.get('chunk_index',0)}"
            )

            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_docs.append(content)
            all_metadatas.append({
                "company": meta.get("company", ""),
                "year": meta.get("year", ""),
                "form": meta.get("form", ""),
                "section": meta.get("section", ""),
                "has_table": meta.get("has_table", False),
                "chunk_index": meta.get("chunk_index", 0),
            })

    print(f"📦 加载完成: {len(all_docs)} 条记录")
    print(f"   来自 {len(set(m['company'] for m in all_metadatas))} 家公司\n")

    # ── Build ChromaDB ─────────────────────────────────────────
    print("⏳ 构建向量数据库...")
    client = chromadb.PersistentClient(
        path=str(Path(db_dir).resolve()),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    collection_name = "sec_10k_chunks"
    # Drop existing collection if re-running
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "SEC 10-K filing chunks for 32 S&P 500 companies"},
    )

    # Batch insert in groups of 500
    batch_size = 500
    for i in range(0, len(all_docs), batch_size):
        end = min(i + batch_size, len(all_docs))
        collection.add(
            ids=all_ids[i:end],
            documents=all_docs[i:end],
            metadatas=all_metadatas[i:end],
        )
        pct = round(end / len(all_docs) * 100)
        print(f"   [{pct:3d}%] {end}/{len(all_docs)} 条已入库", end="\r")
    print(f"\n✅ 向量库构建完成: {collection.count()} 条\n")

    # ── Test queries ───────────────────────────────────────────
    test_queries = [
        "Apple Inc. total revenue 2025",
        "JPMorgan net income 2024",
        "Walmart gross margin percentage",
        "Johnson & Johnson R&D expenses",
        "Compare NVIDIA and AMD gross margins",
    ]

    print(f"{'='*70}")
    print("🔍 检索测试 (Top-3 结果)")
    print(f"{'='*70}")

    for query in test_queries:
        print(f"\n📝 查询: \"{query}\"")
        print(f"{'-'*60}")

        results = collection.query(
            query_texts=[query],
            n_results=3,
        )

        for rank in range(3):
            doc_id = results["ids"][0][rank]
            meta = results["metadatas"][0][rank]
            distance = results.get("distances", [[0, 0, 0]])[0][rank]

            company = meta.get("company", "?")
            year = meta.get("year", "?")
            section = meta.get("section", "?")

            print(f"  #{rank+1} [{company}] {year} — {section}")
            if distance:
                print(f"       distance: {distance:.4f}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("📊 向量库摘要")
    print(f"{'='*70}")
    print(f"  总记录: {collection.count()}")
    print(f"  存储路径: {Path(db_dir).resolve()}")

    companies_in_db = sorted(set(m["company"] for m in all_metadatas))
    print(f"  覆盖公司: {len(companies_in_db)} 家")

    print(f"\n🎉 向量化完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    build_vector_db()
