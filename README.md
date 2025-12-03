\# Multi-Book Agentic RAG Demonstration



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

First make sure pip is installed
python.exe -m pip install --upgrade pip

Then install the required packages

pip install \
    langchain \
    langchain-ollama \
    langchain-community \
    langgraph \
    transformers \
    faiss-cpu \
    tqdm \
    scikit-learn \
    ipython  \
    pypdf

After all the packages are installed, you can check that they are installed with the correct version using the following command (response when running my Windows 11 computer shown as well).

pip list
---RESULT:
Package                  Version
------------------------ ----------
aiohappyeyeballs         2.6.1
aiohttp                  3.13.2
aiosignal                1.4.0
annotated-types          0.7.0
anyio                    4.12.0
asttokens                3.0.1
attrs                    25.4.0
certifi                  2025.11.12
charset-normalizer       3.4.4
colorama                 0.4.6
dataclasses-json         0.6.7
decorator                5.2.1
executing                2.2.1
faiss-cpu                1.13.0
filelock                 3.20.0
frozenlist               1.8.0
fsspec                   2025.10.0
greenlet                 3.2.4
h11                      0.16.0
httpcore                 1.0.9
httpx                    0.28.1
httpx-sse                0.4.3
huggingface-hub          0.36.0
idna                     3.11
ipython                  9.8.0
ipython_pygments_lexers  1.1.1
jedi                     0.19.2
joblib                   1.5.2
jsonpatch                1.33
jsonpointer              3.0.0
langchain                1.1.0
langchain-classic        1.0.0
langchain-community      0.4.1
langchain-core           1.1.0
langchain-ollama         1.0.0
langchain-text-splitters 1.0.0
langgraph                1.0.4
langgraph-checkpoint     3.0.1
langgraph-prebuilt       1.0.5
langgraph-sdk            0.2.12
langsmith                0.4.53
marshmallow              3.26.1
matplotlib-inline        0.2.1
multidict                6.7.0
mypy_extensions          1.1.0
numpy                    2.3.5
ollama                   0.6.1
orjson                   3.11.4
ormsgpack                1.12.0
packaging                25.0
parso                    0.8.5
pip                      25.3
prompt_toolkit           3.0.52
propcache                0.4.1
pure_eval                0.2.3
pydantic                 2.12.5
pydantic_core            2.41.5
pydantic-settings        2.12.0
Pygments                 2.19.2
python-dotenv            1.2.1
PyYAML                   6.0.3
regex                    2025.11.3
requests                 2.32.5
requests-toolbelt        1.0.0
safetensors              0.7.0
scikit-learn             1.7.2
scipy                    1.16.3
SQLAlchemy               2.0.44
stack-data               0.6.3
tenacity                 9.1.2
threadpoolctl            3.6.0
tokenizers               0.22.1
tqdm                     4.67.1
traitlets                5.14.3
transformers             4.57.3
typing_extensions        4.15.0
typing-inspect           0.9.0
typing-inspection        0.4.2
urllib3                  2.5.0
uuid_utils               0.12.0
wcwidth                  0.2.14
xxhash                   3.6.0
yarl                     1.22.0
zstandard                0.25.0

---
Project Structure

main.py
Entry point. Sets up and runs the interactive Command Line Interface (CLI) loop.

lgs_multi_book_rag/config.py
Hyperparameters, model names, and textbook file roots.

lgs_multi_book_rag/prompts.py
All system prompts used in the workflow. 

lgs_multi_book_rag/vectorstore.py
Utilities to build/load FAISS vector stores per PDF, using Ollama embeddings.

lgs_multi_book_rag/reranker.py
Hand-crafted cosine-similarity reranker. (Currently not used)

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

Adding or Changing Textbooks - NOTE: Original textbooks in PDF format are not included in this repository - only the FAISS vector store is included. The textbooks were downloaded from a number of open source/usage sources; so feel free to use your own!

Place new PDFs in the project root (same directory as main.py) and follow the naming pattern used in config.FILE_ROOTS (paths without .pdf).

Add the new file root (without .pdf) to FILE_ROOTS in config.py.

On the next run:

If no .fvs exists for a PDF, the script will create it.

If book_desc.json is missing or FAISS stores had to be created, book descriptions will be regenerated.

All generated artifacts:

FAISS directories: <file_root>.fvs/

Book descriptions: book_desc.json

Expert responses: Expert_0_Response.txt, Expert_1_Response.txt, etc.

Developer Notes

Code is structured with functions, type hints, and docstrings but preserves the original prompts, hyperparameters, and relative paths.

Global TX/RX counters are preserved for measuring tokens and character counts.

The question rewrite logic and maximum rewrite count are preserved.

You can enable the custom reranker by wiring it back in where noted in comments in app.py.

Contributions:

Feel free to open issues or PRs to improve modularity, metrics, or add new expert types (e.g., web RAG, tools, etc.).







