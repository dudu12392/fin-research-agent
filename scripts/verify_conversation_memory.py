"""Verify conversation memory: pronoun resolution, follow-up, clarification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.generator import RAGGenerator

print("⏳ 初始化 RAGGenerator...")
gen = RAGGenerator()

scenarios = [
    {
        "name": "指代消解（省略追问）",
        "turns": [
            "Apple 2025 年营收是多少？",
            "那毛利率呢？",
            "Microsoft 呢？",
        ],
        "expectations": [
            {"should_contain": ["Apple", "416"], "should_not_contain": ["无法回答"]},
            {"should_contain": ["毛利率", "gross"], "should_not_contain": []},
            {"should_contain": ["Microsoft", "MSFT"], "should_not_contain": []},
        ],
    },
    {
        "name": "扩展对比",
        "turns": [
            "对比 Apple 和 Microsoft 的研发费用",
            "再帮我加上 NVIDIA",
        ],
        "expectations": [
            {"should_contain": ["Apple", "Microsoft"], "should_not_contain": []},
            {"should_contain": ["NVIDIA", "NVDA"], "should_not_contain": []},
        ],
    },
    {
        "name": "澄清追问",
        "turns": [
            "Apple 利润",
            "我说的是净利润",
        ],
        "expectations": [
            {"should_contain": [], "should_not_contain": []},
            {"should_contain": ["net income", "净利润"], "should_not_contain": []},
        ],
    },
]

total_turns = 0
passed_turns = 0

for scenario in scenarios:
    print(f"\n{'='*60}")
    print(f"📋 场景: {scenario['name']}")
    print(f"{'='*60}")

    for i, (query, expected) in enumerate(zip(scenario["turns"], scenario["expectations"])):
        print(f"\n  轮次 {i+1}: \"{query}\"")
        result = gen.answer(query)
        answer = result.get("answer", "")
        intent = result.get("intent", "?")

        checks = []
        for term in expected.get("should_contain", []):
            if term.lower() in answer.lower():
                checks.append(f"✅ 含'{term}'")
            else:
                checks.append(f"❌ 缺'{term}'")
        for term in expected.get("should_not_contain", []):
            if term.lower() not in answer.lower():
                checks.append(f"✅ 无'{term}'")
            else:
                checks.append(f"❌ 有'{term}'")

        all_pass = all("✅" in c for c in checks) or not checks
        total_turns += 1
        if all_pass:
            passed_turns += 1
        status = "✅" if all_pass else "❌"

        print(f"  意图: {intent}")
        print(f"  回答: {answer[:200]}...")
        for c in checks:
            print(f"    {c}")
        print(f"  {status} {'通过' if all_pass else '失败'}")
        mem = len(gen.conversation_history) if hasattr(gen, "conversation_history") else 0
        print(f"  记忆: {mem} 轮")

# Summary
print(f"\n{'='*60}")
print("📊 对话记忆验证")
print(f"{'='*60}")
print(f"通过: {passed_turns}/{total_turns} = {passed_turns/total_turns:.0%}")

report_dir = Path("output")
report_dir.mkdir(parents=True, exist_ok=True)
with open(report_dir / "conversation_memory_evaluation_report.md", "w", encoding="utf-8") as f:
    f.write(f"# 对话记忆评估\n\n")
    f.write(f"## 通过率\n{passed_turns}/{total_turns} = {passed_turns/total_turns:.0%}\n\n")
    f.write("## 能力\n- 指代消解\n- 扩展追问\n- 澄清追问\n")

print(f"\n📄 报告: output/conversation_memory_evaluation_report.md")
