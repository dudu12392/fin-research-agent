"""Verify intent routing accuracy: 6 types × 3 queries = 18 test cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.generator import RAGGenerator

print("⏳ 初始化 RAGGenerator...")
gen = RAGGenerator()

test_cases = [
    # single_fact
    ("Apple 2025 年营收是多少？", "single_fact"),
    ("Microsoft 2024 年净利润？", "single_fact"),
    ("JPMorgan 的总资产规模？", "single_fact"),
    # compare
    ("对比 Apple 和 Microsoft 的毛利率", "compare"),
    ("NVIDIA 和 AMD 谁的研发投入更高？", "compare"),
    ("比较 JPMorgan 和 Goldman Sachs 的营收", "compare"),
    # trend
    ("Apple 近三年营收变化趋势", "trend"),
    ("NVIDIA 净利润增长情况如何？", "trend"),
    ("Microsoft 研发费用过去三年的变化", "trend"),
    # ambiguous
    ("Apple 利润", "ambiguous"),
    ("Microsoft 收入多少", "ambiguous"),
    ("NVIDIA 毛利率怎么样", "ambiguous"),
    # chitchat
    ("你好", "chitchat"),
    ("你能做什么？", "chitchat"),
    ("谢谢", "chitchat"),
    # out_of_scope
    ("明天 Apple 股票会涨吗？", "out_of_scope"),
    ("美联储什么时候降息？", "out_of_scope"),
    ("比特币会涨到多少？", "out_of_scope"),
]

correct = 0
results_by_type: dict[str, dict] = {}

print(f"\n{'='*70}")
print(f"{'':<4} {'查询':<44} {'期望':<14} {'实际':<14}")
print(f"{'='*70}")

for query, expected in test_cases:
    result = gen.answer(query)
    actual = result.get("intent", "unknown")

    expected_clean = expected
    if expected not in results_by_type:
        results_by_type[expected_clean] = {"correct": 0, "total": 0}
    results_by_type[expected_clean]["total"] += 1

    is_correct = actual == expected
    if is_correct:
        correct += 1
        results_by_type[expected_clean]["correct"] += 1

    status = "✅" if is_correct else "❌"
    print(f"{status} {query:<42} {expected:<14} {actual:<14}")

# Summary
print(f"\n{'='*60}")
print("📊 意图分类准确率汇总")
print(f"{'='*60}")
print(f"总体: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")
print(f"\n按类型:")
for t, s in sorted(results_by_type.items()):
    acc = s["correct"] / s["total"]
    bar = "█" * int(acc * 20)
    print(f"  {t:<16}: {s['correct']}/{s['total']} = {acc:.0%}  {bar}")

# Save report
report_dir = Path("output")
report_dir.mkdir(parents=True, exist_ok=True)
with open(report_dir / "intent_evaluation_report.md", "w", encoding="utf-8") as f:
    f.write("# 意图分类评估报告\n\n")
    f.write(f"## 总体准确率\n\n{correct}/{len(test_cases)} = {correct/len(test_cases):.1%}\n\n")
    f.write("## 按类型分组\n\n")
    for t, s in sorted(results_by_type.items()):
        acc = s["correct"] / s["total"]
        f.write(f"- {t}: {s['correct']}/{s['total']} = {acc:.0%}\n")

print(f"\n📄 报告: output/intent_evaluation_report.md")
