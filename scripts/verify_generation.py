"""Manual verification — test RAG generation on 6 hand-picked queries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agents.generator import RAGGenerator

print("⏳ 初始化 RAGGenerator...")
gen = RAGGenerator()

test_queries = [
    "Apple 2025 年营收是多少？",
    "Microsoft 2024 年净利润多少？",
    "对比 Apple 和 Microsoft 的毛利率",
    "NVIDIA 的研发费用在 FY2025 是多少？",
    "JPMorgan 的总资产规模有多大？",
    "明天 Apple 股票会涨吗？",
]

for q in test_queries:
    result = gen.answer(q)
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"{'='*60}")
    print(f"A: {result['answer'][:400]}")
    print(f"{'-'*40}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(
        f"  数字校验: {'✅ 通过' if result['verified'] else '⚠️ 未验证: ' + ', '.join(result['unverified_numbers'])}"
    )
    srcs = [
        f"{s['metadata']['company']} FY{s['metadata']['year']}"
        for s in result["sources"]
    ]
    print(f"  来源: {srcs}")

print(f"\n{'='*60}")
print("✅ 验证完成")
