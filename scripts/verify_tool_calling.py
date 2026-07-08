"""Verify tool calling: LLM selects correct tool + fills parameters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.tools.tool_registry import ToolRegistry

registry = ToolRegistry()

test_cases = [
    {
        "query": "Apple 2025 年营收是多少",
        "expected_tool": "search_financial_data",
        "expected_params": {"company": "AAPL", "year": 2025, "metric": "revenue"},
    },
    {
        "query": "Microsoft 2024 年净利润",
        "expected_tool": "search_financial_data",
        "expected_params": {"company": "MSFT", "year": 2024, "metric": "net_income"},
    },
    {
        "query": "JPMorgan 的总资产规模有多大",
        "expected_tool": "search_financial_data",
        "expected_params": {"company": "JPM", "metric": "total_assets"},
    },
    {
        "query": "计算 391035 除以 416161 的比率",
        "expected_tool": "calculate_ratio",
        "expected_params": {"numerator": 391035, "denominator": 416161},
    },
    {
        "query": "Apple 是做什么的",
        "expected_tool": "get_company_info",
        "expected_params": {"ticker": "AAPL"},
    },
    {
        "query": "帮我算一下毛利率，营收 416161，毛利 180000",
        "expected_tool": "calculate_ratio",
        "expected_params": {"numerator": 180000, "denominator": 416161},
    },
]

results = []
for tc in test_cases:
    query = tc["query"]
    expected_tool = tc["expected_tool"]

    print(f"\n{'='*60}")
    print(f"查询: {query}")

    tool_call = registry.select_tool(query)

    actual_tool = tool_call.get("name", "none") if tool_call else "none"
    actual_params = tool_call.get("parameters", {}) if tool_call else {}

    tool_match = actual_tool == expected_tool

    params_ok = True
    params_detail: dict = {}
    for key, expected_val in tc["expected_params"].items():
        actual_val = actual_params.get(key)
        if actual_val is not None:
            match = str(actual_val).upper() == str(expected_val).upper()
        else:
            match = False
        if not match:
            params_ok = False
        params_detail[key] = {"expected": expected_val, "actual": actual_val, "match": match}

    all_pass = tool_match and params_ok
    status = "✅" if all_pass else "❌"

    print(f"  期望: {expected_tool}, 实际: {actual_tool} {'✅' if tool_match else '❌'}")
    print(f"  参数:")
    for k, v in params_detail.items():
        print(f"    {k}: 期望={v['expected']}, 实际={v['actual']} {'✅' if v['match'] else '❌'}")
    print(f"  {status} {'通过' if all_pass else '失败'}")

    results.append({
        "query": query, "expected_tool": expected_tool, "actual_tool": actual_tool,
        "tool_match": tool_match, "params_ok": params_ok,
        "all_pass": all_pass, "params_detail": params_detail,
    })

# Summary
print(f"\n{'='*60}")
print("📊 工具调用验证汇总")
print(f"{'='*60}")
tool_pass = sum(1 for r in results if r["tool_match"])
param_pass = sum(1 for r in results if r["params_ok"])
overall = sum(1 for r in results if r["all_pass"])
print(f"工具选择: {tool_pass}/{len(results)} = {tool_pass/len(results):.0%}")
print(f"参数填充: {param_pass}/{len(results)} = {param_pass/len(results):.0%}")
print(f"整体通过: {overall}/{len(results)} = {overall/len(results):.0%}")
for r in results:
    print(f"  {'✅' if r['all_pass'] else '❌'} {r['query'][:50]}")

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "tool_calling_evaluation_report.md", "w", encoding="utf-8") as f:
    f.write(f"# 工具调用评估报告\n\n")
    f.write(f"## 总体\n")
    f.write(f"- 工具选择: {tool_pass}/{len(results)} = {tool_pass/len(results):.0%}\n")
    f.write(f"- 参数填充: {param_pass}/{len(results)} = {param_pass/len(results):.0%}\n")
    f.write(f"- 整体通过: {overall}/{len(results)} = {overall/len(results):.0%}\n\n")
    for r in results:
        f.write(f"- {'✅' if r['all_pass'] else '❌'} {r['query']}\n")
        f.write(f"  期望: {r['expected_tool']}, 实际: {r['actual_tool']}\n")

print(f"\n📄 报告: output/tool_calling_evaluation_report.md")
