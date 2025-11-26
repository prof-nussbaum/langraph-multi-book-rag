"""
Vector store utilities for building/loading FAISS indexes for each textbook.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
from uuid import uuid4


def create_embeddings(model_name: str) -> OllamaEmbeddings:
    """
    Create an OllamaEmbeddings instance.
    """
    return OllamaEmbeddings(model=model_name)


def format_docs(docs) -> str:
    """
    Concatenate page contents from a list of documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_vs(
    file_root: str,
    embeddings: OllamaEmbeddings,
    chunk_size: int,
    chunk_overlap: int,
) -> FAISS:
    """
    Build a FAISS vector store for a given PDF (by root path without extension),
    save it to `<file_root>.fvs`, and return the FAISS object.
    """
    file = file_root + ".pdf"
    vs_file = file_root + ".fvs"

    print("\nLoading from file ", file)
    pages = []
    for doc in tqdm(PyPDFLoader(file, mode="page").lazy_load(), desc="Pages"):
        pages.append(doc)

    docs_list = [page for page in pages]
    print("\nNumber of pages in document", len(docs_list))
    print("# Split documents")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("\nNumber of split text segments in document", len(doc_splits))
    print("Build and save vector store")

    # Same pattern: FAISS IndexFlatL2 using embedding dimension from a sample
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(doc_splits))]
    vectorstore.add_documents(documents=doc_splits, ids=uuids)

    print("\n# Save FAISS vector DB as ", vs_file)
    vectorstore.save_local(vs_file)
    return vectorstore


def load_vs(file_root: str, embeddings: OllamaEmbeddings) -> FAISS:
    """
    Load a FAISS vector store from `<file_root>.fvs`.
    """
    vs_file = file_root + ".fvs"
    vectorstore = FAISS.load_local(
        vs_file,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def load_or_build_vectorstores(
    file_roots: List[str],
    embeddings: OllamaEmbeddings,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[list, bool]:
    """
    For each file root, either load an existing FAISS store or build a new one.

    Returns:
        (book_vs, created_new_faiss)
        - book_vs: list of [file_root, FAISS]
        - created_new_faiss: True if at least one FAISS store was created
    """
    print("\n# Load books")
    book_vs = []
    created_new_faiss = False

    for file_root in file_roots:
        file = file_root + ".pdf"
        vs_file = file_root + ".fvs"
        if os.path.isdir(vs_file):
            print("loading Vector DB for", file)
            book_vs.append([file_root, load_vs(file_root, embeddings)])
        else:
            print("Creating Vector DB for", file)
            created_new_faiss = True
            book_vs.append(
                [file_root, create_vs(file_root, embeddings, chunk_size, chunk_overlap)]
            )

    return book_vs, created_new_faiss
