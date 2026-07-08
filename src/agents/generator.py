"""RAG generation pipeline: retrieve → contextualize → generate → verify."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import chromadb
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from core.logging import init_session, log_retrieval, log_llm_call

load_dotenv()


class RAGGenerator:
    """End-to-end RAG: vector retrieval + LLM generation + number verification.

    Pipeline:
      1. Retrieve top_k chunks from ChromaDB
      2. Build context string from retrieved chunks
      3. Generate answer via DeepSeek LLM
      4. Verify numbers in answer against context
      5. Calculate confidence score
    """

    def __init__(self, sample_size: int | None = 5000):
        """Initialise RAG generator with ChromaDB and LLM.

        Args:
            sample_size: Max chunks to load (limits memory).
        """
        # ── LLM ───────────────────────────────────────────────
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            temperature=0.1,
        )

        # ── Load prompt template (versioned) ───────────────────
        prompt_version = os.getenv("PROMPT_VERSION", "v1")
        prompt_config = self._load_prompt(version=prompt_version)
        self.system_prompt = prompt_config["system"].strip()
        self.user_template = prompt_config["user"].strip()
        self.prompt_version = prompt_config.get("version", "1.0")

        # ── Load pre-built persistent ChromaDB (no re-embedding) ──
        self.chunks: list[dict[str, Any]] = []

        chunks_dir = Path("data/chunks")
        for jf in sorted(chunks_dir.rglob("*_10k_chunks.json")):
            with open(jf, encoding="utf-8") as f:
                file_chunks = json.load(f)
            for chunk in file_chunks:
                c = chunk.get("content", "")
                if not c or len(c.strip()) < 50:
                    continue
                self.chunks.append({
                    "content": c,
                    "metadata": chunk.get("metadata", {}),
                })

        # Build in-memory ChromaDB with BGE-small (avoids HNSW persistence bugs)
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        # Prioritize table chunks (contain financial numbers), then fill with text
        table_indices = [i for i, c in enumerate(self.chunks) if c.get("metadata", {}).get("has_table")]
        text_indices = [i for i, c in enumerate(self.chunks) if not c.get("metadata", {}).get("has_table")]

        random.seed(42)
        if sample_size is not None:
            half = sample_size // 2
            n_tables = min(half, len(table_indices))
            n_text = min(sample_size - n_tables, len(text_indices))
            n_tables += min(sample_size - n_tables - n_text, len(table_indices) - n_tables)
            selected = random.sample(table_indices, n_tables) + random.sample(text_indices, n_text)
        else:
            selected = list(range(len(self.chunks)))

        selected.sort()
        self.chunks = [self.chunks[i] for i in selected]

        table_count = sum(1 for c in self.chunks if c.get("metadata", {}).get("has_table"))
        print(f"   数据构成: {table_count} 表格块 + {len(self.chunks)-table_count} 文本块 = {len(self.chunks)} 总计")

        ef = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5", device="cpu"
        )
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("rag_gen")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name="rag_gen", embedding_function=ef
        )

        batch = 500
        for i in range(0, len(self.chunks), batch):
            end = min(i + batch, len(self.chunks))
            self.collection.add(
                ids=[
                    f"{c['metadata'].get('company','?')}_{c['metadata'].get('year','?')}_{c['metadata'].get('chunk_index',0)}"
                    for c in self.chunks[i:end]
                ],
                documents=[c["content"] for c in self.chunks[i:end]],
                metadatas=[c["metadata"] for c in self.chunks[i:end]],
            )

        # ── Conversation memory ───────────────────────────────
        self.conversation_history: list[dict] = []
        self.last_user_query: str = ""

        # ── Audit session ─────────────────────────────────────
        self.session_id = init_session()

    # ── Public API ────────────────────────────────────────────────

    def answer(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Run full RAG pipeline and return structured result.

        Args:
            query: Natural-language financial question.
            top_k: Number of chunks to retrieve.

        Returns:
            {
                answer: str,
                sources: list[dict],
                verified: bool,
                unverified_numbers: list[str],
                confidence: float (0-1),
            }
        """
        # Step 0: Resolve follow-up queries via conversation history
        if self.last_user_query and self._is_followup_query(query):
            resolved = self._resolve_followup(query, self.last_user_query)
            if resolved != query:
                query = resolved

        # Step 1: Retrieve
        retrieved = self._retrieve(query, top_k)

        # Step 2: Build context
        context = self._build_context(retrieved)

        # Step 3: Generate
        answer = self._generate(query, context)

        # Step 4: Verify numbers
        verification = self._verify(answer, context)

        # Step 5: Confidence
        confidence = self._calculate_confidence(retrieved, verification["verified"])

        return {
            "answer": answer,
            "sources": retrieved,
            "verified": verification["verified"],
            "unverified_numbers": verification["unverified_numbers"],
            "confidence": round(confidence, 2),
        }

    # ── Step 1: Retrieve ──────────────────────────────────────────

    def _retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Vector search via ChromaDB, filtered to prioritize table chunks."""
        import time
        t0 = time.perf_counter()
        results = self.collection.query(query_texts=[query], n_results=top_k * 2)

        all_sources = []
        for rank, (doc_id, meta, doc_text) in enumerate(
            zip(results["ids"][0], results["metadatas"][0], results.get("documents", [[""]] * top_k * 2)[0]),
            1
        ):
            all_sources.append({
                "id": doc_id,
                "content": doc_text,
                "metadata": dict(meta),
                "rank": rank,
            })

        # Filter: keep table chunks first, fill remaining with text chunks
        table_sources = [s for s in all_sources if s["metadata"].get("has_table")]
        text_sources = [s for s in all_sources if not s["metadata"].get("has_table")]

        sources = table_sources[:top_k]
        if len(sources) < 3:
            needed = min(top_k - len(sources), len(text_sources))
            sources += text_sources[:needed]

        latency_ms = (time.perf_counter() - t0) * 1000
        log_retrieval(query, sources[:5], latency_ms, self.session_id)
        return sources[:top_k]

    # ── Step 2: Context ───────────────────────────────────────────

    def _build_context(self, results: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a structured context string."""
        parts = []
        for r in results:
            meta = r["metadata"]
            header = (
                f"[{meta.get('company','?')} FY{meta.get('year','?')} "
                f"| Section: {meta.get('section','?')}]"
            )
            content = r.get("content", "")[:1500]
            parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(parts)

    # ── Step 3: Generate ──────────────────────────────────────────

    def _generate(self, query: str, context: str) -> str:
        """Call DeepSeek LLM to generate an answer."""
        import time
        t0 = time.perf_counter()

        user_msg = self.user_template.format(query=query, context=context)
        messages = [("system", self.system_prompt)]

        # Inject conversation history (last 6 turns)
        for h in self.conversation_history[-6:]:
            role = h["role"]
            if role == "assistant":
                role = "ai"
            messages.append((role, h["content"]))

        messages.append(("human", user_msg))
        response = self.llm.invoke(messages)
        answer = response.content.strip() if response.content else ""

        # Record conversation
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})
        self.last_user_query = query

        latency_ms = (time.perf_counter() - t0) * 1000
        log_llm_call("", answer, os.getenv("LLM_MODEL", "deepseek-chat"), latency_ms, session_id=self.session_id)

        return answer

    # ── Step 4: Verify ────────────────────────────────────────────

    def _verify(self, answer: str, context: str) -> dict[str, Any]:
        """Check that all numbers in the answer appear in the context.

        Extracts numeric tokens (with optional comma/period/negative sign)
        from the answer and verifies each against the context.

        Returns:
            {verified: bool, unverified_numbers: list[str]}
        """
        # Extract all numeric patterns from answer
        # Matches: 416.1, 1,234, 112,010, -5.3, $100B, 100亿, 100 million
        number_pattern = re.compile(
            r'(?:(?<=\$)\s*)?[\d,]+(?:\.\d+)?\s*(?:亿|万|million|billion|trillion|%|B|M|K)?',
            re.IGNORECASE
        )
        answer_numbers = set()
        for m in number_pattern.finditer(answer):
            num = m.group().strip()
            if len(num) >= 2:  # Skip single digits
                answer_numbers.add(num)

        if not answer_numbers:
            return {"verified": True, "unverified_numbers": []}

        # Check each number in context
        unverified = []
        context_lower = context.lower()
        for num in answer_numbers:
            # Try exact match
            if num.lower() in context_lower:
                continue
            # Try comma-stripped version
            stripped = num.replace(",", "")
            if stripped.lower() in context_lower:
                continue
            # Try the raw digits without suffix
            digits_only = re.search(r'[\d,]+(?:\.\d+)?', num)
            if digits_only:
                raw = digits_only.group().replace(",", "")
                if raw.lower() in context_lower:
                    continue
            unverified.append(num)

        return {
            "verified": len(unverified) == 0,
            "unverified_numbers": unverified,
        }

    # ── Step 5: Confidence ────────────────────────────────────────

    def _calculate_confidence(
        self, results: list[dict[str, Any]], verified: bool
    ) -> float:
        """Calculate answer confidence (0-1) based on retrieval + verification."""
        if not results:
            return 0.0

        # Base: number of unique companies in results (diverse sources = better)
        companies = set(r["metadata"].get("company", "") for r in results)
        diversity = min(len(companies) / 3, 1.0)

        # How many results have actual content
        has_content = sum(1 for r in results if r.get("content") and len(r["content"]) > 100)

        # Verification bonus
        verify_bonus = 0.3 if verified else 0.0

        # Composite
        base = (has_content / len(results) + diversity) / 2
        confidence = min(base + verify_bonus, 1.0)
        return confidence

    # ── Follow-up query resolution ───────────────────────────

    def _is_followup_query(self, query: str) -> bool:
        """Detect if query is a short follow-up referring to prior context."""
        followup_patterns = ["那", "呢", "再", "也", "还"]
        return any(p in query for p in followup_patterns) and len(query) < 20

    def _resolve_followup(self, query: str, last_query: str) -> str:
        """Use LLM to expand a follow-up query into a complete question.

        E.g., "那毛利率呢？" + "Apple 2025 年营收是多少？"
           → "Apple 2025 年毛利率是多少？"
        """
        prompt = (
            "You are a query rewriter. Given a short follow-up query and the "
            "previous user question, rewrite it as a complete, standalone question.\n"
            f"Previous: {last_query}\n"
            f"Follow-up: {query}\n"
            "Rewritten query:"
        )
        response = self.llm.invoke([("human", prompt)])
        rewritten = (response.content or "").strip()
        if len(rewritten) > 5 and len(rewritten) < 200:
            return rewritten
        return query

    def _load_prompt(self, version: str = "v1") -> dict:
        """Load prompt template from versioned YAML file.

        Args:
            version: Prompt version identifier (e.g. 'v1', 'v2').

        Returns:
            Dict with 'version', 'date', 'system', 'user' keys.
        """
        import yaml as _yaml
        from pathlib import Path as _Path

        prompt_path = _Path(f"prompts/generator_{version}.yaml")
        if not prompt_path.exists():
            print(f"   ⚠️ Prompt {version} not found, falling back to v1")
            prompt_path = _Path("prompts/generator_v1.yaml")

        with open(prompt_path, "r", encoding="utf-8") as f:
            config = _yaml.safe_load(f)

        print(f"   📝 Prompt: {config.get('version','?')} ({config.get('description','')[:40]})")
        return config
