\# LGS Multi-Book Agentic RAG (LGSv55)



This project is an agentic multi-book RAG (Retrieval-Augmented Generation) demo built on:



\- \[LangChain](https://python.langchain.com/)

\- \[LangGraph](https://langgraph.dev/)

\- \[Ollama](https://ollama.com/) (local LLM + embeddings)

\- FAISS vector stores per textbook



It uses multiple textbooks (business and culinary) as separate "experts". For each human question:



1\. Each book is ranked on how well it can answer the question.

2\. The top `max\_books` experts (books) retrieve relevant chunks from their vector stores.

3\. Each expert produces an answer using its textbook context.

4\. A critic agent filters out bad expert answers.

5\. A coordinator agent combines the remaining expert answers into a final answer.

6\. If the final answer is judged "bad", the question is reworded up to `max\_rewrite` times and the process repeats.



---



\## Requirements



\- Python 3.10+

\- \[Ollama](https://ollama.com/) running locally

\- The following models pulled in Ollama:



&nbsp; ```bash

&nbsp; ollama pull llama3.2:3b

&nbsp; ollama pull gemma3:12b-it-qat

&nbsp; ollama pull llama3   # embedding model name used in the script


---

Python Packages

pip install \
    langchain-ollama \
    langchain-community \
    langgraph \
    transformers \
    faiss-cpu \
    tqdm \
    scikit-learn \
    ipython


---

Project Structure

Project Structure

main.py
Entry point. Sets up and runs the interactive CLI loop.

lgs_multi_book_rag/config.py
Hyperparameters, model names, and textbook file roots.

lgs_multi_book_rag/prompts.py
All system prompts used in the workflow. These match the original strings exactly.

lgs_multi_book_rag/vectorstore.py
Utilities to build/load FAISS vector stores per PDF, using Ollama embeddings.

lgs_multi_book_rag/reranker.py
Hand-crafted cosine-similarity reranker. (Currently not used, mirroring your latest code which has reranking commented out.)

lgs_multi_book_rag/app.py
Main orchestration logic:

LLM + tokenizer setup

Vector store loading / creation

Book description generation and caching (book_desc.json)

LangGraph state & node definitions

Question rewriting loop, expert critic, and final answer node

ASCII graph printing

Interactive console loop

Usage

From the project root:

python main.py


You’ll see banner messages and an ASCII diagram of the LangGraph.

Then you can interact:

Type in the Human question or instructions to AI, or leave blank to exit.

[Human]:How do I maximize profit in my business?

Results <final combined answer here>
 TX_char  ...  TX_tokn ...  RX_char ...  RX_tokn ...


Press Enter on an empty line to exit.

Adding or Changing Textbooks

Place new PDFs in the project root (same directory as main.py) and follow the naming pattern used in config.FILE_ROOTS (paths without .pdf).

Add the new file root (without .pdf) to FILE_ROOTS in config.py.

On the next run:

If no .fvs exists for a PDF, the script will create it.

If book_desc.json is missing or FAISS stores had to be created, book descriptions will be regenerated.

All generated artifacts:

FAISS directories: <file_root>.fvs/

Book descriptions: book_desc.json

Expert responses: Expert_0_Response.txt, Expert_1_Response.txt, etc.

No-book expert response: Expert_NO BOOK_Response.txt

Developer Notes

Code is structured with functions, type hints, and docstrings but preserves the original prompts, hyperparameters, and relative paths.

Global TX/RX counters are preserved for measuring tokens and character counts.

The question rewrite logic and maximum rewrite count are preserved.

You can enable the custom reranker by wiring it back in where noted in comments in app.py.

Contributions:

Feel free to open issues or PRs to improve modularity, metrics, or add new expert types (e.g., web RAG, tools, etc.).


