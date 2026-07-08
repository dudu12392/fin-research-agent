"""Head-to-head comparison: pure vector vs hybrid (BM25 + Vector + Cross-Encoder) retrieval."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

from src.tools.hybrid_search import HybridSearcher


# ═══════════════════════════════════════════════════════════════════
# Baseline: pure vector searcher
# ═══════════════════════════════════════════════════════════════════

class VectorSearcher:
    """Pure dense-vector retrieval via ChromaDB (baseline)."""

    def __init__(self, collection: Any):
        self.collection = collection

    def search(self, query: str, top_k: int = 10, **_kw) -> list[dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        output = []
        for rank, (doc_id, meta) in enumerate(zip(ids, metadatas), 1):
            output.append({
                "id": doc_id,
                "rank": rank,
                "metadata": dict(meta),
                "score": 0,
            })
        return output


# ═══════════════════════════════════════════════════════════════════
# Shared evaluation logic
# ═══════════════════════════════════════════════════════════════════

def _company_match(expected_companies: list[str], company: str) -> bool:
    """Check if returned company matches any expected."""
    return company.upper() in [c.upper() for c in expected_companies]


def evaluate_searcher(
    searcher: Any,
    queries: list[dict],
    chunks: list[dict],
    top_k: int = 5,
    label: str = "",
) -> dict[str, Any]:
    """Run evaluation on a given searcher.

    Args:
        searcher: Object with .search(query, top_k) → list[{id, rank, metadata}]
        queries: Test query list from test_queries.json
        chunks: Full chunk list (for mapping ids)
        top_k: Results per query
        label: Human-readable name for this searcher
    """
    print(f"\n{'='*60}")
    print(f"📊 评估: {label}")
    print(f"{'='*60}")

    details: list[dict] = []
    type_groups: dict[str, list[dict]] = {}

    for q in queries:
        qid = q["id"]
        qtype = q["type"]
        query_text = q["query"]

        expected_company = q.get("expected_company", "")
        expected = q.get("expected_companies", [])
        if expected_company and expected_company not in expected:
            expected = [expected_company] + list(expected)
        if not expected and expected_company:
            expected = [expected_company]
        expected_year = q.get("expected_year")

        t0 = time.perf_counter()
        results = searcher.search(query_text, top_k=top_k)
        elapsed = time.perf_counter() - t0

        hit_ranks = []
        returned_companies = []
        returned_years = []

        for r in results:
            meta = r.get("metadata", r.get("meta", {}))
            company = meta.get("company", "")
            year = int(meta.get("year", 0))
            returned_companies.append(company)
            returned_years.append(year)
            if _company_match(expected, company):
                hit_ranks.append(r["rank"])

        top1_hit = hit_ranks and hit_ranks[0] == 1
        top3_hit = any(r <= 3 for r in hit_ranks)
        top5_hit = bool(hit_ranks)
        first = hit_ranks[0] if hit_ranks else 0
        mrr = 1.0 / first if first else 0.0
        year_ok = returned_years[0] == expected_year if (expected_year and returned_years) else None

        detail = {
            "id": qid,
            "query": query_text,
            "type": qtype,
            "expected": expected,
            "expected_year": expected_year,
            "returned_companies": returned_companies,
            "returned_years": returned_years,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "top5_hit": top5_hit,
            "mrr": round(mrr, 4),
            "year_match": year_ok,
            "elapsed_ms": round(elapsed * 1000, 1),
        }
        details.append(detail)
        type_groups.setdefault(qtype, []).append(detail)

    in_scope = [d for d in details if d["type"] != "out_of_scope"]
    n = len(in_scope)

    metrics = {
        "label": label,
        "top_1_hit_rate": sum(1 for d in in_scope if d["top1_hit"]) / n if n else 0,
        "top_3_hit_rate": sum(1 for d in in_scope if d["top3_hit"]) / n if n else 0,
        "top_5_hit_rate": sum(1 for d in in_scope if d["top5_hit"]) / n if n else 0,
        "mrr": sum(d["mrr"] for d in in_scope) / n if n else 0,
        "year_accuracy": (
            sum(1 for d in in_scope if d["year_match"] is True)
            / sum(1 for d in in_scope if d["year_match"] is not None)
            if any(d["year_match"] is not None for d in in_scope)
            else 0
        ),
        "avg_latency_ms": sum(d["elapsed_ms"] for d in details) / len(details) if details else 0,
        "type_breakdown": {},
        "details": details,
    }

    for t, group in type_groups.items():
        if t == "out_of_scope":
            continue
        gn = len(group)
        metrics["type_breakdown"][t] = {
            "count": gn,
            "top_1_hit_rate": sum(1 for d in group if d["top1_hit"]) / gn if gn else 0,
            "top_3_hit_rate": sum(1 for d in group if d["top3_hit"]) / gn if gn else 0,
            "mrr": sum(d["mrr"] for d in group) / gn if gn else 0,
        }

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load chunks & test queries ────────────────────────────
    chunks_dir = Path("data/chunks")
    all_chunks: list[dict[str, Any]] = []
    all_docs: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict[str, Any]] = []

    for jf in sorted(chunks_dir.rglob("*_10k_chunks.json")):
        with open(jf, encoding="utf-8") as f:
            file_chunks = json.load(f)
        for chunk in file_chunks:
            c = chunk.get("content", "")
            if not c or len(c.strip()) < 50:
                continue
            meta = chunk.get("metadata", {})
            all_chunks.append({"content": c, "metadata": meta})
            all_docs.append(c)
            all_ids.append(
                f"{meta.get('company','?')}_{meta.get('year','?')}_{meta.get('chunk_index',0)}"
            )
            all_metas.append({
                "company": meta.get("company", ""),
                "year": str(meta.get("year", "")),
                "section": meta.get("section", ""),
            })

    # Subsample for speed (5000 docs)
    random.seed(42)
    MAX_DOCS = 5000
    if len(all_chunks) > MAX_DOCS:
        indices = random.sample(range(len(all_chunks)), MAX_DOCS)
        all_chunks = [all_chunks[i] for i in indices]
        all_docs = [all_docs[i] for i in indices]
        all_ids = [all_ids[i] for i in indices]
        all_metas = [all_metas[i] for i in indices]

    print(f"📦 加载 {len(all_chunks)} chunks（已抽样）")

    # ── Load test queries ─────────────────────────────────────
    with open("src/evaluation/test_queries.json", encoding="utf-8") as f:
        test_queries = json.load(f)

    # ── Build ChromaDB in-memory ──────────────────────────────
    client = chromadb.Client()
    try:
        client.delete_collection("hybrid_eval")
    except Exception:
        pass
    collection = client.create_collection(name="hybrid_eval")

    batch = 500
    for i in range(0, len(all_docs), batch):
        end = min(i + batch, len(all_docs))
        collection.add(ids=all_ids[i:end], documents=all_docs[i:end], metadatas=all_metas[i:end])
    print(f"✅ ChromaDB 就绪: {len(all_docs)} 条")

    # ── Evaluate: pure vector (baseline) ─────────────────────
    vec = VectorSearcher(collection)
    baseline_metrics = evaluate_searcher(vec, test_queries, all_chunks, top_k=5, label="纯向量检索（基线）")

    # ── Evaluate: hybrid ─────────────────────────────────────
    hybrid = HybridSearcher(collection, all_chunks)
    hybrid_metrics = evaluate_searcher(hybrid, test_queries, all_chunks, top_k=5, label="混合检索（BM25 + 向量 + Cross-Encoder）")

    # ── Comparison ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("📊 优化前后对比")
    print(f"{'='*70}")
    print(f"{'指标':<22} {'纯向量':>10} {'混合检索':>10} {'提升':>12}")
    print(f"{'-'*56}")

    for key in ["top_1_hit_rate", "top_3_hit_rate", "top_5_hit_rate", "mrr", "year_accuracy"]:
        b = baseline_metrics[key]
        h = hybrid_metrics[key]
        delta = h - b
        name = key.replace("_", " ").title()
        arrow = "🟢" if delta > 0 else ("🔴" if delta < 0 else "➖")
        print(f"{name:<22} {b:>9.1%} {h:>9.1%} {arrow} {delta:>+9.1%}")

    print(f"{'-'*56}")
    print(f"{'平均延迟 (ms)':<22} {baseline_metrics['avg_latency_ms']:>9.1f} {hybrid_metrics['avg_latency_ms']:>9.1f}")

    # ── Per-type breakdown ────────────────────────────────────
    print(f"\n{'='*70}")
    print("📋 按查询类型分组对比")
    print(f"{'='*70}")
    print(f"{'类型':<16} {'纯向量 Top-1':>12} {'混合 Top-1':>12} {'纯向量 MRR':>12} {'混合 MRR':>12}")
    print(f"{'-'*60}")

    for qtype in sorted(set(baseline_metrics["type_breakdown"]) | set(hybrid_metrics["type_breakdown"])):
        bt = baseline_metrics["type_breakdown"].get(qtype, {})
        ht = hybrid_metrics["type_breakdown"].get(qtype, {})
        print(
            f"{qtype:<16} "
            f"{bt.get('top_1_hit_rate', 0):>11.1%} "
            f"{ht.get('top_1_hit_rate', 0):>11.1%} "
            f"{bt.get('mrr', 0):>11.4f} "
            f"{ht.get('mrr', 0):>11.4f}"
        )

    # ── Save comparison report ────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "baseline": baseline_metrics,
        "hybrid": hybrid_metrics,
    }
    with open(out_dir / "hybrid_comparison.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 对比数据已保存: output/hybrid_comparison.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
