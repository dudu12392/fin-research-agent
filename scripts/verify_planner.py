"""Verify PlannerAgent decomposition for compare and trend queries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.planner import PlannerAgent

planner = PlannerAgent()

test_cases = [
    {
        "query": "对比 Apple 和 Microsoft 的毛利率",
        "intent": "compare",
        "expected_companies": ["AAPL", "MSFT"],
        "expected_metric": "gross margin",
        "expected_min_subqueries": 2,
    },
    {
        "query": "NVIDIA 和 AMD 谁的研发投入更高",
        "intent": "compare",
        "expected_companies": ["NVDA", "AMD"],
        "expected_metric": "R&D expense",
        "expected_min_subqueries": 2,
    },
    {
        "query": "比较 JPMorgan、Goldman Sachs 和 Morgan Stanley 的营收",
        "intent": "compare",
        "expected_companies": ["JPM", "GS", "MS"],
        "expected_metric": "revenue",
        "expected_min_subqueries": 3,
    },
    {
        "query": "Apple 近三年营收变化趋势",
        "intent": "trend",
        "expected_companies": ["AAPL"],
        "expected_metric": "revenue",
        "expected_min_subqueries": 3,
    },
    {
        "query": "NVIDIA 净利润从 2023 到 2025 年的增长情况",
        "intent": "trend",
        "expected_companies": ["NVDA"],
        "expected_metric": "net income",
        "expected_min_subqueries": 3,
    },
]

results = []
for tc in test_cases:
    query = tc["query"]
    intent = tc["intent"]

    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"意图: {intent}")

    sub_queries = planner.decompose(query, intent)
    print(f"拆解: {len(sub_queries)} 个子问题")

    for i, sq in enumerate(sub_queries):
        print(f"  [{i+1}] {sq.get('sub_query','?')[:60]}")
        print(f"       公司={sq.get('target_company','?')}  年份={sq.get('target_year','?')}  指标={sq.get('target_metric','?')}")

    # Checks
    sq_text = str(sub_queries).upper()
    checks = {
        "min_subqueries": len(sub_queries) >= tc["expected_min_subqueries"],
        "companies_match": all(c.upper() in sq_text for c in tc["expected_companies"]),
        "metric_included": (
            tc["expected_metric"].replace(" ", "").replace("&", "").lower()
            in sq_text.replace(" ", "").replace("_", "").lower()
        ),
    }
    all_pass = all(checks.values())
    status = "✅" if all_pass else "❌"
    print(f"  {status} 检查: {checks}")

    results.append({
        "query": query, "intent": intent,
        "sub_queries_count": len(sub_queries),
        "checks": checks, "passed": all_pass,
    })

# Summary
print(f"\n{'='*60}")
print("📊 Planner 拆解验证汇总")
print(f"{'='*60}")
passed = sum(1 for r in results if r["passed"])
print(f"通过率: {passed}/{len(results)} = {passed/len(results):.0%}")
for r in results:
    print(f"  {'✅' if r['passed'] else '❌'} {r['query'][:50]}")

# Save report
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "planner_evaluation_report.md", "w", encoding="utf-8") as f:
    f.write("# Planner 任务拆解评估报告\n\n")
    f.write(f"## 总体\n{passed}/{len(results)} = {passed/len(results):.0%}\n\n")
    for r in results:
        f.write(f"- {'✅' if r['passed'] else '❌'} {r['query']}: {r['sub_queries_count']} 子问题\n")
        f.write(f"  {r['checks']}\n\n")

print(f"\n📄 报告: output/planner_evaluation_report.md")
