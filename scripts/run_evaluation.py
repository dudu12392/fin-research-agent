"""Run RAG retrieval evaluation on the sec_10k_chunks vector database."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import run_evaluation, generate_report, diagnose_failures

# ── Run evaluation ────────────────────────────────────────────────
metrics = run_evaluation(
    test_queries_path="src/evaluation/test_queries.json",
    top_k=5,
    db_dir="data/vector_db",
)

if not metrics:
    print("❌ 评估失败 — 检查向量库是否存在")
    sys.exit(1)

# ── Print summary ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("📊 评估摘要")
print(f"{'='*60}")
print(f"Top-1 命中率:    {metrics['top_1_hit_rate']:.1%}")
print(f"Top-3 命中率:    {metrics['top_3_hit_rate']:.1%}")
print(f"Top-5 命中率:    {metrics['top_5_hit_rate']:.1%}")
print(f"MRR:             {metrics['mrr']:.4f}")
print(f"年份准确率:      {metrics['year_accuracy']:.1%}")
print(f"范围外噪声率:    {metrics['out_of_scope_noise_rate']:.1%}")

print(f"\n按查询类型分组：")
for qtype, m in sorted(metrics["type_breakdown"].items()):
    print(
        f"  {qtype:<15} count={m['count']:>2}  "
        f"Top-1={m['top_1_hit_rate']:.1%}  "
        f"Top-3={m['top_3_hit_rate']:.1%}  "
        f"MRR={m['mrr']:.4f}"
    )

# ── Generate reports ──────────────────────────────────────────────
generate_report(metrics, "output/evaluation_report.md")
diagnose_failures(
    "src/evaluation/test_queries.json",
    "output/failure_analysis.md",
    "data/vector_db",
)

print(f"\n📊 报告已保存到 output/")
