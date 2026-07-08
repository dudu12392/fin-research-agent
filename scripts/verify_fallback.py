"""Verify fallback strategies: 7 failure scenarios, graceful degradation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.generator import RAGGenerator

print("⏳ 初始化...")
gen = RAGGenerator()

test_cases = [
    {
        "name": "知识库外公司",
        "query": "小卖部连锁集团 2025 年营收是多少？",
        "expected_behavior": "明确告知公司不在知识库中，不编造数字",
        "check": lambda a: any(k in a.lower() for k in ["无法", "找不到", "不存在", "没有", "未收录"]),
        "should_not": lambda a: not any(str(i) in a for i in range(100, 999)),
    },
    {
        "name": "年份超范围",
        "query": "Apple 2035 年营收预测是多少？",
        "expected_behavior": "说明仅有历史数据，无法预测未来",
        "check": lambda a: any(k in a.lower() for k in ["无法", "预测", "历史", "未来", "没有"]),
    },
    {
        "name": "股价预测",
        "query": "明天 Apple 股票会涨到多少？",
        "expected_behavior": "拒绝预测股价",
        "check": lambda a: any(k in a.lower() for k in ["无法", "预测", "股价", "短期"]),
    },
    {
        "name": "非财报领域",
        "query": "美联储什么时候降息？",
        "expected_behavior": "说明超出财报范围",
        "check": lambda a: any(k in a.lower() for k in ["无法", "超出", "范围", "财报", "知识库"]),
    },
    {
        "name": "模糊查询",
        "query": "Apple 利润",
        "expected_behavior": "追问或标注不完整",
        "check": lambda a: any(k in a.lower() for k in ["毛利", "净利", "具体", "可能", "澄清"]),
    },
    {
        "name": "部分数据缺失",
        "query": "对比 Apple 和 某不知名公司的研发费用",
        "expected_behavior": "标注某公司无数据，仍提供Apple数据",
        "check": lambda a: "apple" in a.lower() and any(k in a.lower() for k in ["无法", "找不到", "无", "没有"]),
    },
    {
        "name": "非财务指标",
        "query": "Apple 2025 年员工满意度是多少？",
        "expected_behavior": "说明财报不包含该指标",
        "check": lambda a: any(k in a.lower() for k in ["无法", "不包含", "找不到", "没有", "非财务"]),
    },
]

print(f"\n{'='*60}")
print(f"📊 兜底策略验证 ({len(test_cases)} 场景)")
print(f"{'='*60}")

passed = 0
results = []

for tc in test_cases:
    print(f"\n--- {tc['name']} ---")
    print(f"查询: {tc['query']}")
    print(f"预期: {tc['expected_behavior']}")

    result = gen.answer(tc["query"])
    answer = result.get("answer", "")

    check_ok = tc["check"](answer)
    not_ok = tc["should_not"](answer) if "should_not" in tc else True
    all_ok = check_ok and not_ok
    if all_ok:
        passed += 1

    print(f"回答: {answer[:200]}...")
    print(f"  检查: {'✅' if check_ok else '❌'}  违规: {'✅' if not_ok else '❌'}")
    print(f"  {'✅ 通过' if all_ok else '❌ 失败'}")

    results.append({"name": tc["name"], "passed": all_ok, "answer": answer[:100]})

# Summary
print(f"\n{'='*60}")
print(f"📊 汇总: {passed}/{len(test_cases)} = {passed/len(test_cases):.0%}")
for r in results:
    print(f"  {'✅' if r['passed'] else '❌'} {r['name']}")

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "fallback_evaluation_report.md", "w", encoding="utf-8") as f:
    f.write(f"# 兜底策略评估\n\n通过: {passed}/{len(test_cases)} = {passed/len(test_cases):.0%}\n\n")
    for r in results:
        f.write(f"- {'✅' if r['passed'] else '❌'} {r['name']}: {r['answer']}...\n")

print(f"\n📄 报告: output/fallback_evaluation_report.md")
