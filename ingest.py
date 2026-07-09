"""
ingest.py — Clones the target repository, performs AST-based structural chunking
with Tree-sitter, extracts git commit history, and upserts everything into
Pinecone while persisting the parent docstore to disk.

Architecture decisions implemented here
----------------------------------------
• Pitfall #1: Tree-sitter >=0.23 API — Language(capsule) + Parser(language).
• Pitfall #2: PineconeVectorStore(text_key="page_content").
• Pitfall #3: Manual parent-document pattern — children → vectorstore,
  parents → InMemoryStore docstore, saved to docstore.pkl.

Embedding model: sentence-transformers/all-mpnet-base-v2 (768 dims, local,
no API key required). Replaces Google text-embedding-004 which was removed
from the v1beta gRPC endpoint used by langchain-google-genai.
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import git
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from utils import retry_on_failure, setup_logging

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logger = setup_logging()

# ---------------------------------------------------------------------------
# Config / Constants
# ---------------------------------------------------------------------------
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "codebase-rag")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
GITHUB_REPO_URL: str = os.getenv("GITHUB_REPO_URL", "")
CLONE_DIR: str = os.getenv("CLONE_DIR", "./cloned_repo")
DOCSTORE_PATH: str = "docstore.pkl"

# We will use Cohere embeddings which output 1024 dimensions for v3
EMBEDDING_DIMENSION: int = 1024
EMBEDDING_METRIC: str = "cosine"
COHERE_EMBED_MODEL: str = "embed-english-v3.0"
BATCH_SIZE: int = 50
MAX_COMMITS: int = 100
MAX_COMMIT_FILES: int = 10

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".py", ".js", ".ts", ".md", ".txt")
SKIP_DIRS: Tuple[str, ...] = (
    ".git", "node_modules", "venv", "__pycache__",
    "build", "dist", ".idea", ".vscode",
)

REQUIRED_ENV_VARS = [
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "GROQ_API_KEY",
    "COHERE_API_KEY",
    "GITHUB_REPO_URL",
]


# ---------------------------------------------------------------------------
# Tree-sitter helpers
# ---------------------------------------------------------------------------

def _build_parsers():
    """
    Build and return Tree-sitter parsers for Python and JavaScript/TypeScript.
    Uses the >=0.23 API: Language(capsule) and Parser(language).
    Returns (python_parser, js_parser, python_language, js_language).
    """
    try:
        import tree_sitter_python
        import tree_sitter_javascript
        from tree_sitter import Language, Parser

        python_language = Language(tree_sitter_python.language())
        js_language = Language(tree_sitter_javascript.language())

        python_parser = Parser(python_language)
        js_parser = Parser(js_language)

        return python_parser, js_parser, python_language, js_language
    except Exception as exc:
        logger.error("Failed to initialise Tree-sitter parsers: %s", exc)
        return None, None, None, None


PYTHON_PARSER, JS_PARSER, PYTHON_LANGUAGE, JS_LANGUAGE = _build_parsers()

# Node types to extract as child chunks
PYTHON_NODE_TYPES = {"function_definition", "class_definition", "decorated_definition"}
JS_NODE_TYPES = {
    "function_declaration",
    "function_expression",
    "class_declaration",
    "method_definition",
    "arrow_function",
    "generator_function_declaration",
    "export_statement",
}


def _extract_symbol_name(node, source_bytes: bytes, lang: str) -> str:
    """
    Safely extract the symbol name from an AST node.
    Returns 'unnamed' when no name child is found or text decoding fails.
    """
    try:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return "unnamed"
        raw = source_bytes[name_node.start_byte: name_node.end_byte]
        return raw.decode("utf-8", errors="ignore").strip() or "unnamed"
    except Exception:
        return "unnamed"


def _ast_chunk_file(
    file_path: str,
    source_text: str,
    parser,
    node_types: set,
    lang: str,
) -> List[Document]:
    """
    Parse `source_text` with the provided Tree-sitter `parser` and return one
    LangChain Document per top-level node whose type is in `node_types`.

    Each child Document carries metadata:
        file_path, node_type, symbol_name, start_line, end_line, source_lang.
    Falls back to a single full-file Document on any parse failure or if no
    matching nodes are found.
    """
    parent_id = f"file:{file_path}"

    try:
        source_bytes = source_text.encode("utf-8")
        tree = parser.parse(source_bytes)
        root = tree.root_node

        child_docs: List[Document] = []

        def walk(node):
            """Recursive DFS visitor."""
            if node.type in node_types:
                chunk_bytes = source_bytes[node.start_byte: node.end_byte]
                chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
                symbol_name = _extract_symbol_name(node, source_bytes, lang)

                child_docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "parent_id": parent_id,
                            "file_path": file_path,
                            "node_type": node.type,
                            "symbol_name": symbol_name,
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "source_lang": lang,
                            "chunk_type": "ast_child",
                        },
                    )
                )
                # Don't descend into matched nodes to avoid duplicate nesting.
                return
            for child in node.children:
                walk(child)

        walk(root)

        if child_docs:
            return child_docs

        # No matching nodes found — fall back to full-file chunk.
        logger.debug(
            "No AST nodes matched in %s — using full-file fallback chunk.", file_path
        )
    except Exception as exc:
        logger.warning("AST parse error for %s: %s — using full-file fallback.", file_path, exc)

    # Fallback: single whole-file child.
    return [
        Document(
            page_content=source_text,
            metadata={
                "parent_id": parent_id,
                "file_path": file_path,
                "node_type": "full_file",
                "symbol_name": Path(file_path).name,
                "start_line": 1,
                "end_line": source_text.count("\n") + 1,
                "source_lang": lang,
                "chunk_type": "full_file_fallback",
            },
        )
    ]


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------

def clone_or_pull_repo(repo_url: str, clone_dir: str) -> git.Repo:
    """Clone the repository if not present, or if it's a different repo."""
    import shutil
    clone_path = Path(clone_dir)
    if clone_path.exists() and (clone_path / ".git").exists():
        repo = git.Repo(clone_dir)
        try:
            current_url = next(repo.remotes.origin.urls, "")
        except StopIteration:
            current_url = ""
            
        if current_url.strip() == repo_url.strip():
            logger.info("Repository already cloned at %s — pulling latest …", clone_dir)
            try:
                repo.remotes.origin.pull()
                logger.info("Pull complete.")
            except git.GitCommandError as exc:
                logger.warning("Pull failed (continuing with existing state): %s", exc)
            return repo
        else:
            logger.info("Different repository URL detected. Wiping existing clone directory...")
            shutil.rmtree(clone_dir)

    logger.info("Cloning %s → %s …", repo_url, clone_dir)
    clone_path.mkdir(parents=True, exist_ok=True)
    repo = git.Repo.clone_from(repo_url, clone_dir)
    logger.info("Clone complete.")
    return repo


def collect_source_files(clone_dir: str) -> List[str]:
    """Walk the cloned repo and collect all supported source file paths."""
    source_files: List[str] = []
    base = Path(clone_dir)

    for root, dirs, files in os.walk(base):
        # Prune skip directories in-place so os.walk won't descend into them.
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if any(fname.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                source_files.append(str(Path(root) / fname))

    logger.info("Collected %d source files.", len(source_files))
    return source_files


def extract_commit_documents(repo: git.Repo) -> List[Document]:
    """
    Extract up to MAX_COMMITS from git history as LangChain Documents.
    Each Document page_content is the commit message.
    Metadata includes sha, changed_files list, and type='commit'.
    The parent_id is 'commit:<short_sha>' — commits serve as their own parents.
    """
    commit_docs: List[Document] = []

    try:
        commits = list(repo.iter_commits(max_count=MAX_COMMITS))
    except Exception as exc:
        logger.error("Failed to iterate git commits: %s", exc)
        return []

    for commit in commits:
        try:
            message = (commit.message or "").strip()
            if not message:
                continue

            short_sha = commit.hexsha[:7]
            parent_id = f"commit:{short_sha}"

            # Collect changed file paths (up to MAX_COMMIT_FILES).
            changed_files: List[str] = []
            try:
                if commit.parents:
                    diff = commit.parents[0].diff(commit)
                    changed_files = [
                        d.a_path or d.b_path
                        for d in diff
                        if (d.a_path or d.b_path)
                    ][:MAX_COMMIT_FILES]
            except Exception:
                pass  # Changed files are best-effort.

            content = (
                f"Commit: {short_sha}\n"
                f"Author: {commit.author.name} <{commit.author.email}>\n"
                f"Date: {commit.committed_datetime.isoformat()}\n"
                f"Message:\n{message}\n"
            )
            if changed_files:
                content += "Changed files:\n" + "\n".join(f"  - {f}" for f in changed_files)

            doc = Document(
                page_content=content,
                metadata={
                    "parent_id": parent_id,
                    "sha": short_sha,
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                    "changed_files": changed_files,
                    "type": "commit",
                    "chunk_type": "commit",
                },
            )
            commit_docs.append(doc)
        except Exception as exc:
            logger.warning("Error processing commit %s: %s", getattr(commit, "hexsha", "?")[:7], exc)

    logger.info("Extracted %d commit documents.", len(commit_docs))
    return commit_docs


# ---------------------------------------------------------------------------
# Pinecone helpers
# ---------------------------------------------------------------------------

@retry_on_failure(max_retries=4, delay=3, backoff=2)
def _ensure_pinecone_index(pc: Pinecone, index_name: str) -> None:
    """Create the Pinecone serverless index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        logger.info("Pinecone index '%s' already exists.", index_name)
        return

    logger.info("Creating Pinecone serverless index '%s' …", index_name)
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric=EMBEDDING_METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    logger.info("Index '%s' created successfully.", index_name)


@retry_on_failure(max_retries=3, delay=2, backoff=2)
def _build_embeddings() -> CohereEmbeddings:
    """
    Load the Cohere embedding model to run lightweight in the cloud.
    """
    logger.info("Loading Cohere embedding model '%s' …", COHERE_EMBED_MODEL)
    return CohereEmbeddings(
        model=COHERE_EMBED_MODEL,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )


@retry_on_failure(max_retries=3, delay=2, backoff=2)
def _build_vectorstore(embeddings: CohereEmbeddings) -> PineconeVectorStore:
    """
    Construct PineconeVectorStore with text_key='page_content' to align with
    LangChain Document field naming (pitfall #2).
    """
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        text_key="page_content",
        pinecone_api_key=PINECONE_API_KEY,
    )


# ---------------------------------------------------------------------------
# Batch upsert helper
# ---------------------------------------------------------------------------

@retry_on_failure(max_retries=4, delay=3, backoff=2)
def _upsert_batch(vectorstore: PineconeVectorStore, batch: List[Document]) -> None:
    vectorstore.add_documents(batch)


def _split_large_children(children: List[Document], chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split child documents that exceed a certain size threshold to ensure
    they fit within Pinecone's 40KB metadata limit and improve retrieval granularity.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    processed: List[Document] = []
    for doc in children:
        if len(doc.page_content) > chunk_size:
            split_docs = splitter.split_documents([doc])
            for i, sd in enumerate(split_docs):
                sd.metadata["chunk_index"] = i
                if "symbol_name" in sd.metadata:
                    sd.metadata["symbol_name"] = f"{sd.metadata['symbol_name']} (part {i+1})"
            processed.extend(split_docs)
        else:
            processed.append(doc)
    return processed


# ---------------------------------------------------------------------------
# Main ingestion orchestration
# ---------------------------------------------------------------------------

def run_ingestion(repo_url: Optional[str] = None) -> None:
    """Full ingestion pipeline: clone → chunk → embed → upsert → save docstore."""

    # --- Use provided URL or fallback to env ---
    active_repo_url = repo_url or GITHUB_REPO_URL
    if not active_repo_url:
        logger.error("No GITHUB_REPO_URL provided via parameter or environment variable.")
        return

    # --- Env var validation ---
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        logger.error(
            "The following required environment variables are not set: %s\n"
            "Copy .env.example to .env and fill in all values before running ingestion.",
            ", ".join(missing),
        )
        return

    # --- Clone / pull repo ---
    repo = clone_or_pull_repo(active_repo_url, CLONE_DIR)
    
    # Extract repo name for namespace isolation in local docstore
    repo_name = active_repo_url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    # --- Collect source files ---
    source_files = collect_source_files(CLONE_DIR)

    # --- Extract commit history ---
    commit_docs = extract_commit_documents(repo)

    # --- Initialise Pinecone index ---
    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ensure_pinecone_index(pc, PINECONE_INDEX_NAME)

    # --- Build embeddings & vectorstore ---
    logger.info("Initialising embeddings model …")
    embeddings = _build_embeddings()

    logger.info("Connecting to Pinecone vectorstore …")
    vectorstore = _build_vectorstore(embeddings)

    # --- Build local docstore ---
    docstore = InMemoryStore()

    # --- Process source files ---
    all_children: List[Document] = []

    for file_path in source_files:
        try:
            source_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Cannot read file %s: %s", file_path, exc)
            continue

        # Relative path for cleaner metadata
        try:
            rel_path = str(Path(file_path).relative_to(Path(CLONE_DIR)))
        except ValueError:
            rel_path = file_path

        parent_id = f"{repo_name}:{rel_path}"

        # Store the full file as the parent document in the docstore.
        parent_doc = Document(
            page_content=source_text,
            metadata={
                "type": "parent",
                "parent_id": parent_id,
                "file_path": rel_path,
            },
        )
        docstore.mset([(parent_id, parent_doc)])

        # Determine parser based on extension.
        ext = Path(file_path).suffix.lower()
        if ext == ".py" and PYTHON_PARSER is not None:
            children = _ast_chunk_file(rel_path, source_text, PYTHON_PARSER, PYTHON_NODE_TYPES, "python")
        elif ext in (".js", ".ts") and JS_PARSER is not None:
            children = _ast_chunk_file(rel_path, source_text, JS_PARSER, JS_NODE_TYPES, "javascript")
        else:
            # Markdown, text, or unsupported — single full-file child chunk.
            children = [
                Document(
                    page_content=source_text,
                    metadata={
                        "parent_id": parent_id,
                        "file_path": rel_path,
                        "node_type": "prose",
                        "symbol_name": Path(file_path).name,
                        "start_line": 1,
                        "end_line": source_text.count("\n") + 1,
                        "source_lang": ext.lstrip(".") or "text",
                        "chunk_type": "prose_full",
                    },
                )
            ]

        all_children.extend(children)

    # --- Process commit documents ---
    for commit_doc in commit_docs:
        pid = commit_doc.metadata["parent_id"]
        # Commits are their own parents — stored in docstore AND added to children list.
        docstore.mset([(pid, commit_doc)])
        all_children.append(commit_doc)

    # --- Split large children to prevent Pinecone 40KB metadata limit errors ---
    all_children = _split_large_children(all_children, chunk_size=2000, chunk_overlap=200)

    logger.info(
        "Total child documents to upsert: %d (%d from files, %d from commits).",
        len(all_children),
        len(all_children) - len(commit_docs),
        len(commit_docs),
    )

    # --- Batch upsert to Pinecone ---
    total_batches = (len(all_children) + BATCH_SIZE - 1) // BATCH_SIZE
    success_count = 0

    for i in range(0, len(all_children), BATCH_SIZE):
        batch = all_children[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info("Upserting batch %d/%d (%d docs) …", batch_num, total_batches, len(batch))
        try:
            _upsert_batch(vectorstore, batch)
            success_count += len(batch)
        except Exception as exc:
            logger.error(
                "Batch %d/%d failed after retries — skipping %d docs. Error: %s",
                batch_num,
                total_batches,
                len(batch),
                exc,
            )

    logger.info(
        "Upsert complete. %d/%d documents successfully indexed.",
        success_count,
        len(all_children),
    )

    # --- Persist docstore to disk ---
    with open(DOCSTORE_PATH, "wb") as fh:
        pickle.dump(docstore, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Docstore persisted to '%s'.", DOCSTORE_PATH)
    logger.info("Ingestion pipeline finished successfully.")


if __name__ == "__main__":
    run_ingestion()
