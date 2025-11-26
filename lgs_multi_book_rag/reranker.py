"""
Cosine-similarity based reranker for retrieved documents.

Currently not used by default, reranking calls are commented out. 
You can enable this easily in `app.py` where noted.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


def get_embedding(text: str, embeddings: OllamaEmbeddings):
    """
    Generate an embedding for a given text using the given embeddings object.
    """
    return embeddings.embed_query(text)


def rerank_documents(
    query: str,
    documents: List,
    embeddings: OllamaEmbeddings,
    top_n: int,
):
    """
    Rerank documents based on their cosine similarity to the query.

    Args:
        query: User query.
        documents: List of LangChain Document objects.
        embeddings: OllamaEmbeddings instance.
        top_n: How many top docs to keep.

    Returns:
        List of top_n documents, sorted by relevance.
    """
    query_embedding = get_embedding(query, embeddings)

    document_embeddings = [
        get_embedding(doc.page_content, embeddings) for doc in documents
    ]

    similarities = [
        cosine_similarity([query_embedding], [doc_embed])[0][0]
        for doc_embed in document_embeddings
    ]

    ranked_pairs = sorted(
        zip(similarities, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    print(f"len ranked_docs[0:top_n] {len(ranked_pairs[0:top_n])}")
    # Return only the documents (not the scores) to preserve upstream usage pattern
    return [doc for _, doc in ranked_pairs[0:top_n]]
