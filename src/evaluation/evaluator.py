"""RAG retrieval evaluation — hit rate, MRR, year accuracy, failure diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb


# ═══════════════════════════════════════════════════════════════════
# 1. run_evaluation
# ═══════════════════════════════════════════════════════════════════

def run_evaluation(
    test_queries_path: str = "src/evaluation/test_queries.json",
    top_k: int = 5,
    db_dir: str = "data/vector_db",
) -> dict[str, Any]:
    """Run retrieval evaluation against the ChromaDB vector store.

    Args:
        test_queries_path: Path to test_queries.json.
        top_k: Number of results to retrieve per query.
        db_dir: Path to ChromaDB persistent directory.

    Returns:
        Metrics dict with top_1_hit_rate, top_3_hit_rate, top_5_hit_rate,
        mrr, year_accuracy, type_breakdown, and per_query_details.
    """
    # ── Load test queries ─────────────────────────────────────
    with open(test_queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    # ── Build in-memory vector DB from chunk JSONs ────────────
    import random
    random.seed(42)

    chunks_dir = Path("data/chunks")
    all_docs: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for json_file in sorted(chunks_dir.rglob("*_10k_chunks.json")):
        with open(json_file, encoding="utf-8") as f:
            file_chunks = json.load(f)
        for chunk in file_chunks:
            c = chunk.get("content", "")
            if not c or len(c.strip()) < 50:
                continue
            meta = chunk.get("metadata", {})
            all_docs.append(c)
            all_ids.append(
                f"{meta.get('company','?')}_{meta.get('year','?')}_{meta.get('chunk_index',0)}"
            )
            all_metadatas.append({
                "company": meta.get("company", ""),
                "year": str(meta.get("year", "")),
                "section": meta.get("section", ""),
                "has_table": meta.get("has_table", False),
            })

    # Subsample for manageable evaluation time (~5000 docs)
    MAX_EVAL_DOCS = 5000
    if len(all_docs) > MAX_EVAL_DOCS:
        indices = random.sample(range(len(all_docs)), MAX_EVAL_DOCS)
        all_docs = [all_docs[i] for i in indices]
        all_ids = [all_ids[i] for i in indices]
        all_metadatas = [all_metadatas[i] for i in indices]

    client = chromadb.Client()  # in-memory ephemeral
    collection_name = "eval_collection"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)

    # Batch insert
    batch = 500
    for i in range(0, len(all_docs), batch):
        end = min(i + batch, len(all_docs))
        collection.add(
            ids=all_ids[i:end],
            documents=all_docs[i:end],
            metadatas=all_metadatas[i:end],
        )

    total_docs = len(all_docs)
    print(f"\n📊 运行评估: {len(queries)} 个查询 × top-{top_k}")
    print(f"   向量库: {total_docs} 条记录 (in-memory)\n")

    # ── Per-query evaluation ──────────────────────────────────
    results: list[dict] = []
    type_groups: dict[str, list[dict]] = {}

    for q in queries:
        qid = q["id"]
        qtype = q["type"]
        query_text = q["query"]

        # Expected
        expected_company = q.get("expected_company", "")
        expected_companies = q.get("expected_companies", [])
        if expected_company:
            expected_companies = [expected_company] if not expected_companies else expected_companies
        expected_year = q.get("expected_year")

        # Retrieve
        retrieval = collection.query(
            query_texts=[query_text],
            n_results=top_k,
        )

        hit_ranks: list[int] = []
        returned_companies: list[list[str]] = []
        returned_years: list[int] = []

        for rank in range(top_k):
            meta = retrieval["metadatas"][0][rank]
            r_company = meta.get("company", "")
            r_year = int(meta.get("year", 0))

            returned_companies.append(r_company)
            returned_years.append(r_year)

            if r_company in expected_companies:
                hit_ranks.append(rank + 1)

        # Compute per-query metrics
        top1_hit = hit_ranks and hit_ranks[0] == 1
        top3_hit = any(r <= 3 for r in hit_ranks)
        top5_hit = bool(hit_ranks)
        first_hit_rank = hit_ranks[0] if hit_ranks else 0
        mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
        year_match = returned_years[0] == expected_year if (expected_year and returned_years) else None

        detail = {
            "id": qid,
            "query": query_text,
            "type": qtype,
            "expected_companies": expected_companies,
            "expected_year": expected_year,
            "returned_companies": returned_companies,
            "returned_years": returned_years,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "top5_hit": top5_hit,
            "mrr": round(mrr, 4),
            "year_match": year_match,
        }
        results.append(detail)
        type_groups.setdefault(qtype, []).append(detail)

    # ── Aggregate metrics ─────────────────────────────────────
    total = len(results)
    # Exclude out_of_scope from hit rate (they should NOT match anything)
    in_scope = [r for r in results if r["type"] != "out_of_scope"]
    in_scope_total = len(in_scope)
    out_of_scope = [r for r in results if r["type"] == "out_of_scope"]

    metrics: dict[str, Any] = {
        "total_queries": total,
        "in_scope_queries": in_scope_total,
        "out_of_scope_queries": len(out_of_scope),
        "top_k": top_k,
        "total_docs": total_docs,
        # Overall (in-scope only)
        "top_1_hit_rate": (
            sum(1 for r in in_scope if r["top1_hit"]) / in_scope_total
            if in_scope_total else 0
        ),
        "top_3_hit_rate": (
            sum(1 for r in in_scope if r["top3_hit"]) / in_scope_total
            if in_scope_total else 0
        ),
        "top_5_hit_rate": (
            sum(1 for r in in_scope if r["top5_hit"]) / in_scope_total
            if in_scope_total else 0
        ),
        "mrr": (
            sum(r["mrr"] for r in in_scope) / in_scope_total
            if in_scope_total else 0
        ),
        "year_accuracy": (
            sum(1 for r in in_scope if r["year_match"] is True)
            / sum(1 for r in in_scope if r["year_match"] is not None)
            if any(r["year_match"] is not None for r in in_scope) else 0
        ),
        # Out-of-scope: should return irrelevant results (high distance)
        "out_of_scope_noise_rate": (
            sum(1 for r in out_of_scope if r["top5_hit"]) / len(out_of_scope)
            if out_of_scope else 0
        ),
        # Per-type breakdown
        "type_breakdown": {},
        # Full details
        "details": results,
    }

    for qtype, group in type_groups.items():
        if qtype == "out_of_scope":
            continue
        n = len(group)
        metrics["type_breakdown"][qtype] = {
            "count": n,
            "top_1_hit_rate": sum(1 for r in group if r["top1_hit"]) / n if n else 0,
            "top_3_hit_rate": sum(1 for r in group if r["top3_hit"]) / n if n else 0,
            "top_5_hit_rate": sum(1 for r in group if r["top5_hit"]) / n if n else 0,
            "mrr": sum(r["mrr"] for r in group) / n if n else 0,
        }

    return metrics


# ═══════════════════════════════════════════════════════════════════
# 2. generate_report
# ═══════════════════════════════════════════════════════════════════

def generate_report(metrics: dict[str, Any], output_path: str) -> None:
    """Generate a Markdown evaluation report.

    Args:
        metrics: Metrics dict from run_evaluation().
        output_path: Path to save the .md report.
    """
    if not metrics:
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# RAG 检索评估报告",
        "",
        f"**向量库记录数**: {metrics.get('total_docs', 0):,}",
        f"**测试查询数**: {metrics.get('total_queries', 0)} "
        f"(in-scope: {metrics.get('in_scope_queries', 0)}, "
        f"out-of-scope: {metrics.get('out_of_scope_queries', 0)})",
        f"**Top-K**: {metrics.get('top_k', 5)}",
        "",
        "---",
        "",
        "## 📊 整体指标",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| Top-1 命中率 | {metrics.get('top_1_hit_rate', 0):.1%} |",
        f"| Top-3 命中率 | {metrics.get('top_3_hit_rate', 0):.1%} |",
        f"| Top-5 命中率 | {metrics.get('top_5_hit_rate', 0):.1%} |",
        f"| MRR (Mean Reciprocal Rank) | {metrics.get('mrr', 0):.4f} |",
        f"| 年份准确率 | {metrics.get('year_accuracy', 0):.1%} |",
        f"| 范围外噪声率 | {metrics.get('out_of_scope_noise_rate', 0):.1%} |",
        "",
        "---",
        "",
        "## 📋 按查询类型分组",
        "",
        "| 类型 | 数量 | Top-1 | Top-3 | Top-5 | MRR |",
        "|------|------|-------|-------|-------|-----|",
    ]

    for qtype, tb in sorted(metrics.get("type_breakdown", {}).items()):
        lines.append(
            f"| {qtype} | {tb['count']} | {tb['top_1_hit_rate']:.1%} "
            f"| {tb['top_3_hit_rate']:.1%} | {tb['top_5_hit_rate']:.1%} "
            f"| {tb['mrr']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## ❌ 失败案例",
        "",
    ])

    failures = [d for d in metrics.get("details", []) if not d["top1_hit"] and d["type"] != "out_of_scope"]
    if failures:
        lines.append("| ID | 查询 | 期望 | 实际 Top-1 |")
        lines.append("|----|------|------|-----------|")
        for f in failures:
            query_short = f["query"][:50]
            expected = ", ".join(f["expected_companies"][:3])
            actual = f["returned_companies"][0] if f["returned_companies"] else "N/A"
            lines.append(f"| {f['id']} | {query_short} | {expected} | {actual} |")
    else:
        lines.append("✅ 无失败案例（所有 in-scope 查询 Top-1 命中）")

    lines.append("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📄 报告已保存: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# 3. diagnose_failures
# ═══════════════════════════════════════════════════════════════════

def diagnose_failures(
    test_queries_path: str = "src/evaluation/test_queries.json",
    output_path: str = "output/failure_analysis.md",
    db_dir: str = "data/vector_db",
) -> None:
    """Generate detailed failure analysis with possible causes.

    Args:
        test_queries_path: Path to test_queries.json.
        output_path: Path to save failure analysis report.
        db_dir: Path to ChromaDB persistent directory.
    """
    with open(test_queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    # Build in-memory collection (same as run_evaluation)
    import random
    random.seed(42)

    chunks_dir = Path("data/chunks")
    all_docs: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []
    for json_file in sorted(chunks_dir.rglob("*_10k_chunks.json")):
        with open(json_file, encoding="utf-8") as f:
            file_chunks = json.load(f)
        for chunk in file_chunks:
            c = chunk.get("content", "")
            if not c or len(c.strip()) < 50:
                continue
            meta = chunk.get("metadata", {})
            all_docs.append(c)
            all_ids.append(f"{meta.get('company','?')}_{meta.get('year','?')}_{meta.get('chunk_index',0)}")
            all_metadatas.append({
                "company": meta.get("company", ""),
                "year": str(meta.get("year", "")),
                "section": meta.get("section", ""),
            })
    MAX_EVAL_DOCS = 5000
    if len(all_docs) > MAX_EVAL_DOCS:
        indices = random.sample(range(len(all_docs)), MAX_EVAL_DOCS)
        all_docs = [all_docs[i] for i in indices]
        all_ids = [all_ids[i] for i in indices]
        all_metadatas = [all_metadatas[i] for i in indices]

    client = chromadb.Client()
    try:
        client.delete_collection("diag_collection")
    except Exception:
        pass
    collection = client.create_collection(name="diag_collection")
    batch = 500
    for i in range(0, len(all_docs), batch):
        end = min(i + batch, len(all_docs))
        collection.add(ids=all_ids[i:end], documents=all_docs[i:end], metadatas=all_metadatas[i:end])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# 🔍 检索失败诊断报告",
        "",
        "---",
        "",
    ]

    failure_count = 0
    for q in queries:
        qid = q["id"]
        qtype = q["type"]
        query_text = q["query"]

        expected_company = q.get("expected_company", "")
        expected_companies = q.get("expected_companies", [])
        if expected_company:
            expected_companies = [expected_company] if not expected_companies else expected_companies
        expected_year = q.get("expected_year")

        if qtype == "out_of_scope":
            continue

        retrieval = collection.query(query_texts=[query_text], n_results=5)
        top_meta = retrieval["metadatas"][0][0]
        top_company = top_meta.get("company", "")
        top_year = top_meta.get("year", "")

        if top_company in expected_companies and (not expected_year or int(top_year) == expected_year):
            continue  # Hit — skip in failure report

        failure_count += 1
        lines.append(f"## 失败 #{failure_count}: Q{qid} [{qtype}]")
        lines.append("")
        lines.append(f"**查询**: `{query_text}`")
        lines.append(f"**期望公司**: {', '.join(expected_companies)}")
        lines.append(f"**期望年份**: {expected_year or '不限定'}")
        lines.append("")
        lines.append("**实际 Top-3 结果**:")
        lines.append("")
        lines.append("| 排名 | 公司 | 年份 | 章节 |")
        lines.append("|------|------|------|------|")

        for rank in range(3):
            meta = retrieval["metadatas"][0][rank]
            lines.append(
                f"| #{rank+1} | {meta.get('company','?')} "
                f"| {meta.get('year','?')} "
                f"| {meta.get('section','?')} |"
            )

        lines.append("")

        # Possible causes
        causes = []
        if top_company in expected_companies and int(top_year) != expected_year:
            causes.append("年份不匹配：公司命中但年份错误")
        elif top_company not in expected_companies:
            causes.append(f"公司不匹配：返回了 {top_company} 而非期望的 {expected_companies[0]}")
        if qtype == "ambiguous":
            causes.append("查询本身歧义高，关键词过于泛化")
        if qtype == "industry":
            causes.append("行业查询涉及多公司排序，向量检索可能只匹配到单个公司关键词")
        if qtype == "compare":
            causes.append("多公司对比查询，嵌入向量可能偏向某一家公司的上下文")

        lines.append("**可能原因**:")
        for c in causes:
            lines.append(f"- {c}")
        lines.append("")
        lines.append("---")
        lines.append("")

    if failure_count == 0:
        lines.append("✅ 所有查询命中，无失败案例需要诊断。")
    else:
        lines.insert(4, f"共 **{failure_count}** 个失败案例\n")

    lines.append("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"🔍 诊断报告已保存: {output_path}")
