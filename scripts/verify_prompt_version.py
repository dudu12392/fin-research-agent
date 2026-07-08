"""Verify prompt version management — single instance, one query."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.generator import RAGGenerator
import yaml

print("=" * 60)
print("📝 Prompt 版本管理验证")
print("=" * 60)

# Load both YAML files directly
for version in ["v1", "v2"]:
    path = Path(f"prompts/generator_{version}.yaml")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"\n--- {version} ---")
    print(f"  版本号: {config.get('version')}")
    print(f"  日期: {config.get('date')}")
    print(f"  描述: {config.get('description')}")
    print(f"  system长度: {len(config.get('system',''))}")
    print(f"  user长度: {len(config.get('user',''))}")

# Check v2 features
with open("prompts/generator_v2.yaml", encoding="utf-8") as f:
    v2 = yaml.safe_load(f)
system_v2 = v2.get("system", "")
checks = {
    "Few-shot示例": "Few-shot" in system_v2,
    "表格输出要求": "表格" in system_v2,
    "置信度标注": "置信度" in system_v2,
    "数据来源章节": "具体章节" in system_v2,
}
print(f"\n📊 v2 新增特性:")
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")

# Verify generator loads correct version
print(f"\n--- 生成器加载验证 ---")
gen = RAGGenerator()
print(f"  激活版本: {gen.prompt_version}")
print(f"  system首行: {gen.system_prompt[:60]}...")
assert gen.prompt_version == "1.0", "Expected v1.0 by default"
print(f"  ✅ 默认加载 v1.0")

# Switch version via _load_prompt (avoids re-embedding)
config_v2 = gen._load_prompt("v2")
print(f"  v2 system长度: {len(config_v2['system'])}")
assert config_v2["version"] == "2.0", "Expected v2.0"
print(f"  ✅ _load_prompt('v2') 加载成功")

print(f"\n{'='*60}")
print("✅ Prompt 版本管理验证完成")
