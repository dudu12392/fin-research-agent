"""Verify full-pipeline structured logging audit."""

import sys
import shutil
import json
from pathlib import Path

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.agents.generator import RAGGenerator

# Clean old logs
log_dir = Path("logs")
if log_dir.exists():
    shutil.rmtree(log_dir)

gen = RAGGenerator()

# Execute one full query
result = gen.answer("Apple 2025 年营收是多少？")
print(f"查询结果: {result['answer'][:100]}...")

# Check log directory
log_dir = Path("logs")
if log_dir.exists():
    sessions = list(log_dir.iterdir())
    print(f"\n日志目录: {log_dir}")
    print(f"session 数量: {len(sessions)}")

    for session_dir in sessions:
        print(f"\n📁 Session: {session_dir.name}")
        for log_file in session_dir.iterdir():
            print(f"  📄 {log_file.name} ({log_file.stat().st_size} bytes)")
            if log_file.suffix in [".jsonl", ".log"]:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print(f"     记录: {len(lines)} 条")
                    for line in lines[:3]:
                        try:
                            entry = json.loads(line)
                            print(f"     - {entry.get('event','?'):20} {str(entry)[:80]}")
                        except Exception:
                            print(f"     - {line[:80]}")
else:
    print("❌ 日志目录不存在")

# Collect all events
all_events = []
for session_dir in log_dir.iterdir():
    for log_file in session_dir.glob("*"):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_events.append(json.loads(line).get("event", ""))
                except Exception:
                    pass

print(f"\n{'='*60}")
print("📊 审计事件清单")
print(f"{'='*60}")
for event in ["retrieval", "llm_call", "tool_call", "intent"]:
    matches = [e for e in all_events if event in e]
    print(f"  {'✅' if matches else '❌'} {event}: {matches[:2]}")

print(f"\n总事件数: {len(all_events)}")
print(f"唯一事件类型: {len(set(all_events))}")
print(f"✅ 全链路日志审计完成")
