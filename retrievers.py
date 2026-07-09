"""
retrievers.py — Custom LangChain BaseRetriever implementing the Parent-Document Retrieval
pattern manually, as required by the architecture spec (pitfall #3).

Flow
----
1. similarity_search on the vectorstore to find the top-K *child* chunks.
2. Collect unique parent_id values, preserving insertion order.
3. docstore.mget(parent_ids) to fetch the full *parent* documents.
4. Filter out any None results (docstore misses) with a logged warning.
5. Return the parent documents to the caller.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.stores import BaseStore
from pydantic import Field, ConfigDict

logger = logging.getLogger(__name__)


class ParentFetchingRetriever(BaseRetriever):
    """
    Retriever that searches *child* AST chunks in the vectorstore and then
    fetches the corresponding *parent* (full file or full commit) from a local
    InMemoryStore docstore.

    Attributes
    ----------
    vectorstore : any
        A LangChain-compatible vectorstore (PineconeVectorStore).
    docstore : BaseStore
        A LangChain InMemoryStore keyed by parent_id strings.
    id_key : str
        Metadata field on child Documents that holds the parent_id value.
    k : int
        Number of child chunks to retrieve via similarity search before
        deduplicating and fetching their parents.
    """

    vectorstore: object = Field(..., description="PineconeVectorStore instance")
    docstore: object = Field(..., description="InMemoryStore keyed by parent_id")
    id_key: str = Field(default="parent_id", description="Metadata key for parent ID")
    k: int = Field(default=20, description="Number of child chunks to similarity-search")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Core retrieval logic.

        Parameters
        ----------
        query : str
            The search query string.
        run_manager : CallbackManagerForRetrieverRun, optional
            LangChain callback manager (unused directly here).

        Returns
        -------
        List[Document]
            Parent documents (full files / full commit blocks) whose children
            were among the top-K similarity matches.
        """
        # -------------------------------------------------------------------
        # Step 1: Similarity search to get child chunks.
        # -------------------------------------------------------------------
        try:
            child_docs: List[Document] = self.vectorstore.similarity_search(
                query, k=self.k
            )
        except Exception as exc:
            logger.error("Vectorstore similarity_search failed: %s", exc)
            return []

        if not child_docs:
            logger.debug("similarity_search returned no results for query: %r", query)
            return []

        # -------------------------------------------------------------------
        # Step 2: Collect unique parent_id values, preserving order.
        # -------------------------------------------------------------------
        seen: set = set()
        ordered_parent_ids: List[str] = []
        for doc in child_docs:
            pid: Optional[str] = doc.metadata.get(self.id_key)
            if pid is None:
                logger.warning(
                    "Child document missing '%s' in metadata — skipping. "
                    "Source snippet: %r",
                    self.id_key,
                    doc.page_content[:80],
                )
                continue
            if pid not in seen:
                seen.add(pid)
                ordered_parent_ids.append(pid)

        if not ordered_parent_ids:
            logger.warning("No valid parent IDs found among child documents.")
            return []

        # -------------------------------------------------------------------
        # Step 3: Fetch parent documents from the docstore.
        # -------------------------------------------------------------------
        try:
            fetched = self.docstore.mget(ordered_parent_ids)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.error("Docstore mget failed: %s", exc)
            return []

        # -------------------------------------------------------------------
        # Step 4: Filter out None results (docstore misses).
        # -------------------------------------------------------------------
        parent_docs: List[Document] = []
        for pid, parent_doc in zip(ordered_parent_ids, fetched):
            if parent_doc is None:
                logger.warning(
                    "Docstore miss for parent_id=%r — document was not stored "
                    "during ingestion or the docstore was not loaded correctly.",
                    pid,
                )
                continue
            parent_docs.append(parent_doc)

        logger.debug(
            "Returning %d parent docs (from %d child matches, %d unique parent IDs).",
            len(parent_docs),
            len(child_docs),
            len(ordered_parent_ids),
        )
        return parent_docs
