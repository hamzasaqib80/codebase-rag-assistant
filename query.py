"""
query.py — Interactive CLI for the Enterprise Codebase Onboarding & Q&A Assistant.

Usage
-----
    python query.py

Type your question at the prompt and press Enter.
Type 'quit' or 'exit' (or press Ctrl-C / Ctrl-D) to leave.
"""

import logging
import sys
from typing import List

from langchain_core.documents import Document

from rag_pipeline import get_qa_chain
from utils import setup_logging

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
logger = setup_logging()

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
BANNER = """
╔══════════════════════════════════════════════════════════════════════════╗
║        Enterprise Codebase Onboarding & Q&A Assistant                   ║
║        Powered by: Pinecone · Local Embeddings · Groq · Cohere Rerank   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Ask any question about the indexed repository — code structure,         ║
║  function behaviour, architecture decisions, or commit history.          ║
║  Type 'quit' or 'exit' (or Ctrl-C) to terminate.                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Source attribution helper
# ---------------------------------------------------------------------------

def _format_sources(context_docs: List[Document]) -> str:
    """
    Build a human-readable, deduplicated list of source references from the
    list of LangChain Documents returned in response['context'].

    Each line shows:
        <file_path or commit SHA>  [<node_type> / <symbol_name>]

    Falls back to a generic message when the context list is empty.
    """
    if not context_docs:
        return "  (No structured document footprints were directly matched)"

    seen: set = set()
    lines: List[str] = []

    for doc in context_docs:
        meta = doc.metadata or {}

        chunk_type: str = meta.get("chunk_type", "")
        node_type: str = meta.get("node_type", "")
        symbol_name: str = meta.get("symbol_name", "")
        file_path: str = meta.get("file_path", "")
        sha: str = meta.get("sha", "")
        parent_id: str = meta.get("parent_id", "")

        # Build the primary label.
        if file_path:
            label = file_path
        elif sha:
            label = f"commit:{sha}"
        elif parent_id:
            label = parent_id
        else:
            label = doc.page_content[:60].strip().replace("\n", " ") + " …"

        # Build the annotation (type / symbol).
        annotation_parts: List[str] = []
        if node_type and node_type not in ("full_file", "prose"):
            annotation_parts.append(node_type)
        if symbol_name and symbol_name != "unnamed":
            annotation_parts.append(symbol_name)
        annotation = "  [" + " / ".join(annotation_parts) + "]" if annotation_parts else ""

        entry = f"  • {label}{annotation}"
        if entry not in seen:
            seen.add(entry)
            lines.append(entry)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main() -> None:
    print(BANNER)

    # Initialise the chain once; this loads docstore, connects to Pinecone, etc.
    print("⏳  Initialising RAG pipeline — this may take a few seconds …\n")
    try:
        qa_chain = get_qa_chain()
    except FileNotFoundError as exc:
        print(f"\n❌  ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Fatal error during RAG pipeline initialisation.")
        print(f"\n❌  Failed to initialise pipeline: {exc}")
        sys.exit(1)

    print("✅  Pipeline ready. Ask your question below.\n")
    print("─" * 76)

    while True:
        # ------------------------------------------------------------------
        # Read user input
        # ------------------------------------------------------------------
        try:
            user_input = input("\n🔍  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("\nGoodbye! 👋")
            break

        # ------------------------------------------------------------------
        # Invoke the RAG chain
        # ------------------------------------------------------------------
        print("\n⌛  Thinking …\n")
        try:
            response = qa_chain.invoke({"input": user_input})
        except KeyboardInterrupt:
            print("\n\n⚠️   Interrupted. Type 'exit' to quit or ask another question.")
            continue
        except Exception as exc:
            logger.error("Chain invocation failed: %s", exc, exc_info=True)
            print(
                f"\n⚠️   An error occurred while generating the answer: {exc}\n"
                "Please try again or rephrase your question."
            )
            continue

        # ------------------------------------------------------------------
        # Display answer
        # ------------------------------------------------------------------
        answer: str = response.get("answer", "(No answer returned)")
        context_docs: List[Document] = response.get("context", [])

        print("─" * 76)
        print("🤖  Assistant:\n")
        print(answer)
        print("\n" + "─" * 76)
        print("📂  Sources referenced:\n")
        print(_format_sources(context_docs))
        print("─" * 76)


if __name__ == "__main__":
    main()
