"""
Main orchestration logic for the LGS multi-book RAG demo.
Organized into functions and using explicit configuration and prompt modules.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Literal

import numpy as np
from IPython.display import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from typing_extensions import TypedDict

from .config import (
    MAX_BOOKS,
    K,
    TOP_N,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NUM_CTX,
    LOCAL_LLM_NAME,
    SMARTER_LLM_NAME,
    EMBEDDING_MODEL_NAME,
    FILE_ROOTS,
    BOOK_DESC_FILE,
)
from .prompts import (
    rag_prompt,
    final_answer_prompt,
    book_description_prompt,
    ranking_prompt,
    critique_prompt,
    reword_prompt,
)
from .vectorstore import (
    create_embeddings,
    format_docs,
    load_or_build_vectorstores,
)
# from .reranker import rerank_documents  # Optional: currently disabled 


#############################
# Global counters (TX/RX)
#############################

TX_char = 0
TX_tokn = 0
RX_char = 0
RX_tokn = 0

# Question rewrite tracking
rewrite_count = 0


#############################
# Utility: tokenizer & token counting
#############################

def build_tokenizer() -> PreTrainedTokenizerFast:
    """
    Build a tokenizer used for counting tokens and characters.

    Uses `tokenizer.json` in the current working directory, matching the original script.
    """
    return PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json",
        clean_up_tokenization_spaces=True,
    )


def token_len(tokenizer: PreTrainedTokenizerFast, text: str) -> tuple[int, int]:
    """
    Count characters and tokens for a given text.
    """
    tokens = len(tokenizer.encode(text=text))
    characters = len(text)
    return characters, tokens


#############################
# Book descriptions
#############################

def load_or_build_book_descriptions(
    book_vs,
    llm: ChatOllama,
    tokenizer: PreTrainedTokenizerFast,
    created_new_faiss: bool,
) -> list[str]:
    """
    Load book descriptions from BOOK_DESC_FILE if available and FAISS stores are not new.
    Otherwise, generate descriptions and save them.

    Behavior mirrors the original script.
    """
    global TX_char, TX_tokn, RX_char, RX_tokn

    book_descriptions: list[str] = []

    if not created_new_faiss:
        # Expect an existing book description file
        if os.path.isfile(BOOK_DESC_FILE):
            print("loading existing Book Descriptions ", BOOK_DESC_FILE)
            with open(BOOK_DESC_FILE, "r") as f:
                book_descriptions = json.load(f)
        else:
            print("Book Descriptions file ", BOOK_DESC_FILE, " needs to be created.")
            created_new_faiss = True

    num_books = len(book_vs)

    # Need to build descriptions
    if created_new_faiss:
        for i in range(num_books):
            book_name = book_vs[i][0]
            retriever = book_vs[i][1].as_retriever(k=K)
            question = book_description_prompt
            print("\nBeginning to summarize ", book_name, "\n")
            docs = retriever.invoke(question)
            docs_txt = format_docs(docs)
            rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
            generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

            # Update counters
            c, t = token_len(tokenizer, rag_prompt_formatted)
            if t > NUM_CTX:
                print(
                    f"\n-->WARNING Token count {t} of query to BOOK DESCRIPTION exceeded "
                    f"max context token limit of {NUM_CTX}"
                )
            TX_char += c
            TX_tokn += t
            c, t = token_len(tokenizer, generation.content)
            RX_char += c
            RX_tokn += t

            json_book_description = (
                f"book_name: {book_name}, book_id: {str(i)}, book_desc: {generation.content}"
            )
            json_book_description = "{" + json_book_description + "}"
            book_descriptions.append(json_book_description)
            print(
                "\nFirst 1000 characters of Book Description ",
                str(json_book_description)[0:1000],
                "\n",
            )

        with open(BOOK_DESC_FILE, "w") as f:
            json.dump(book_descriptions, f, indent=4)

    return book_descriptions


#############################
# Graph state definition
#############################

class State(TypedDict):
    original_human_question: str
    revised_question: str
    book_0_answer: str
    book_1_answer: str
    book_2_answer: str
    book_3_answer: str
    book_4_answer: str
    no_book_answer: str
    book_0_index: int
    book_1_index: int
    book_2_index: int
    book_3_index: int
    book_4_index: int
    combined_answer: str


#############################
# Graph node implementations
#############################

def get_question_node(
    state: State,
    llm: ChatOllama,
    tokenizer: PreTrainedTokenizerFast,
    book_descriptions: list[str],
    book_vs,
):
    """
    Determine which books are most relevant to the question.

    Returns fields:
        - book_0_index
        - book_1_index
        - book_2_index
        - book_3_index
        - book_4_index
    """
    global TX_char, TX_tokn, RX_char, RX_tokn

    question = state["revised_question"]
    num_books = len(book_vs)

    top_books = []
    for i in range(num_books):
        context = book_descriptions[i]
        ranking_prompt_formatted = ranking_prompt.format(
            context=context,
            question=question,
        )
        generation = llm.invoke([HumanMessage(content=ranking_prompt_formatted)])

        c, t = token_len(tokenizer, ranking_prompt_formatted)
        if t > NUM_CTX:
            print(
                f"\n-->WARNING Token count {t} of query to FINDING THE BEST BOOKS "
                f"exceeded max context token limit of {NUM_CTX}"
            )
        TX_char += c
        TX_tokn += t
        c, t = token_len(tokenizer, generation.content)
        RX_char += c
        RX_tokn += t

        try:
            this_rank = int(generation.content[0:1])
        except ValueError:
            this_rank = 0
            print("Book: ", book_vs[i][0], " Invalid Response ERROR:  ", generation.content)

        print("Book: ", book_vs[i][0], " has ranking: ", this_rank)
        top_books.append(this_rank)

    sorted_books = np.argsort(top_books)
    print("top books, most relevant first:")
    for i in range(num_books, num_books - MAX_BOOKS, -1):
        print(book_vs[sorted_books[i - 1]][0])

    return {
        "book_0_index": int(sorted_books[num_books - 1]),
        "book_1_index": int(sorted_books[num_books - 2]),
        "book_2_index": int(sorted_books[num_books - 3]),
        "book_3_index": int(sorted_books[num_books - 4]),
        "book_4_index": int(sorted_books[num_books - 5]),
    }


def make_expert_node(expert_id: int, llm: ChatOllama, tokenizer: PreTrainedTokenizerFast, book_vs):
    """
    Factory to create expert nodes X_0function ... X_4function with identical behavior
    to the original script, including logging to Expert_{id}_Response.txt.
    """

    def expert_node(state: State):
        global TX_char, TX_tokn, RX_char, RX_tokn

        index_key = f"book_{expert_id}_index"
        i = state[index_key]
        book_name = book_vs[i][0]

        retriever = book_vs[i][1].as_retriever(search_kwargs={"k": K})
        question = state["revised_question"] + " "
        docs = retriever.invoke(question)

        # To re-enable reranking, uncomment below and comment out the lines above:
        # from .reranker import rerank_documents
        # k_docs = retriever.invoke(question)
        # docs = rerank_documents(question, k_docs, embeddings, TOP_N)

        docs_txt = format_docs(docs)
        rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
        generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

        c, t = token_len(tokenizer, rag_prompt_formatted)
        if t > NUM_CTX:
            print(
                f"\n-->WARNING Token count {t} of query to {book_name} "
                f"exceeded max context token limit of {NUM_CTX}"
            )
        TX_char += c
        TX_tokn += t
        c, t = token_len(tokenizer, generation.content)
        RX_char += c
        RX_tokn += t

        answer_slot = f"book_{expert_id}_answer"
        print("\nExpert ", book_vs[i][0], " provided a response\n")
        filename = f"Expert_{expert_id}_Response.txt"
        with open(filename, "w") as f:
            f.write(
                f"\nExpert {answer_slot} using texbook {book_name} provided this response: \n"
                f"{generation.content}"
            )

        return {
            answer_slot: generation.content,
        }

    return expert_node


def no_book_function(state: State, llm: ChatOllama, tokenizer: PreTrainedTokenizerFast):
    """
    Expert that relies only on the LLM without any textbook context.
    Behavior matches the original `no_book_function`.
    """
    global TX_char, TX_tokn, RX_char, RX_tokn

    question = state["revised_question"]
    generation = llm.invoke([HumanMessage(content=question)])

    c, t = token_len(tokenizer, question)
    if t > NUM_CTX:
        print(
            f"\n-->WARNING Token count {t} of query to no_book_answer "
            f"exceeded max context token limit of {NUM_CTX}"
        )
    TX_char += c
    TX_tokn += t
    c, t = token_len(tokenizer, generation.content)
    RX_char += c
    RX_tokn += t

    answer_slot = "no_book_answer"
    print("\nExpert ", answer_slot, " provided a response\n")
    with open("Expert_NO BOOK_Response.txt", "w") as f:
        f.write(
            f"\nExpert {answer_slot} using no texbook provided this response: \n"
            f"{generation.content}"
        )

    return {
        answer_slot: generation.content,
    }


def critique_of_experts(
    state: State,
    llm: ChatOllama,
    tokenizer: PreTrainedTokenizerFast,
    book_vs,
):
    """
    Node that removes expert responses deemed 'bad' by the critic prompt.
    """
    global TX_char, TX_tokn, RX_char, RX_tokn

    context = " "
    question = state["original_human_question"]
    answer_slot_list = []
    answer_list = []

    for i in range(MAX_BOOKS):
        answer_slot = f"book_{i}_answer"
        answer_slot_list.append(answer_slot)
        answer = state[answer_slot]

        critique_prompt_formatted = critique_prompt.format(
            question=question,
            answer=answer,
        )
        generation = llm.invoke([HumanMessage(content=critique_prompt_formatted)])

        c, t = token_len(tokenizer, critique_prompt_formatted)
        if t > NUM_CTX:
            print(
                f"\n-->WARNING Token count {t} of query to EXPERT CRITIC "
                f"exceeded max context token limit of {NUM_CTX}"
            )
        TX_char += c
        TX_tokn += t
        c, t = token_len(tokenizer, generation.content)
        RX_char += c
        RX_tokn += t

        if generation.content == "bad":
            book_index_hook = f"book_{i}_index"
            book_index = state[book_index_hook]
            book_name = book_vs[int(book_index)][0]
            print(f"\n removing expert answer from {book_name}\n")
            answer = " "
        answer_list.append(answer)
        context += state[answer_slot]

    # Maintain behavior if MAX_BOOKS < 5 (although in this script it's always 5)
    if MAX_BOOKS < 5:
        for i in range(MAX_BOOKS, 5):
            answer_slot_list.append("NO EXPERT")
            answer_list.append(" ")

    return {
        answer_slot_list[0]: answer_list[0],
        answer_slot_list[1]: answer_list[1],
        answer_slot_list[2]: answer_list[2],
        answer_slot_list[3]: answer_list[3],
        answer_slot_list[4]: answer_list[4],
    }


def combined_answer_node(
    state: State,
    llm: ChatOllama,
    tokenizer: PreTrainedTokenizerFast,
):
    """
    Combine all (possibly filtered) expert answers into a single final answer
    using the final_answer_prompt.
    """
    global TX_char, TX_tokn, RX_char, RX_tokn

    context = " "
    question = state["original_human_question"]
    print("\nOriginal human question was : ", question)
    revised_question = state["revised_question"]
    if question != revised_question:
        print("\nRevised question        : ", revised_question)

    for i in range(MAX_BOOKS):
        answer_slot = f"book_{i}_answer"
        context += state[answer_slot]

    # Skip no_book_expert, as in the latest script (commented out)
    rag_prompt_formatted = final_answer_prompt.format(context=context, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

    c, t = token_len(tokenizer, rag_prompt_formatted)
    if t > NUM_CTX:
        print(
            f"\n-->WARNING Token count {t} of query to FINAL_ANSWER "
            f"exceeded max context token limit of {NUM_CTX}"
        )
    TX_char += c
    TX_tokn += t
    c, t = token_len(tokenizer, generation.content)
    RX_char += c
    RX_tokn += t

    return {
        "combined_answer": generation.content,
    }


def revise_question(
    state: State,
    llm: ChatOllama,
    tokenizer: PreTrainedTokenizerFast,
) -> Literal["reword_question", "final_answer"]:
    """
    Conditional node that decides whether the question should be reworded
    based on the critic's judgment of the combined_answer.
    """
    global TX_char, TX_tokn, RX_char, RX_tokn

    question = state["original_human_question"]
    print("\nDo I need to revise the original question?")

    answer = state["combined_answer"]
    critique_prompt_formatted = critique_prompt.format(question=question, answer=answer)
    generation = llm.invoke([HumanMessage(content=critique_prompt_formatted)])

    c, t = token_len(tokenizer, critique_prompt_formatted)
    if t > NUM_CTX:
        print(
            f"\n-->WARNING Token count {t} of query to CRITIC CHOICE "
            f"exceeded max context token limit of {NUM_CTX}"
        )
    TX_char += c
    TX_tokn += t
    c, t = token_len(tokenizer, generation.content)
    RX_char += c
    RX_tokn += t

    if generation.content == "bad":
        print("---> YES!")
        return "reword_question"
    else:
        print("---> No, the question was OK.")
        return "final_answer"


def reword_question(
    state: State,
    llm: ChatOllama,
    tokenizer: PreTrainedTokenizerFast,
):
    """
    Node that produces a revised version of the human question, up to MAX_REWRITE times.
    """
    global TX_char, TX_tokn, RX_char, RX_tokn
    global rewrite_count

    rewrite_count += 1
    if rewrite_count > MAX_REWRITE:
        raise SystemExit("QUESTION REWRITE MAXIMUM REACHED - SOLUTION NOT FOUND")

    original_human_question = state["original_human_question"]
    revised_question = state["revised_question"]

    reword_prompt_formatted = reword_prompt.format(
        original_human_question=original_human_question,
        revised_question=revised_question,
    )
    generation = llm.invoke([HumanMessage(content=reword_prompt_formatted)])

    c, t = token_len(tokenizer, reword_prompt_formatted)
    if t > NUM_CTX:
        print(
            f"\n-->WARNING Token count {t} of query to REWORD_QUESTION "
            f"exceeded max context token limit of {NUM_CTX}"
        )
    TX_char += c
    TX_tokn += t
    c, t = token_len(tokenizer, generation.content)
    RX_char += c
    RX_tokn += t

    print("\n*******************************************")
    print("\nOriginal Question: ", original_human_question)
    print("\nPrevious revised Question: ", revised_question)
    print("\nNewly Created Revised Question: ", generation.content)

    return {
        "revised_question": generation.content,
    }


def generate_final_answer(state: State):
    """
    Final node that simply returns the already-computed combined_answer.
    """
    return {
        "combined_answer": state["combined_answer"],
    }


#############################
# Graph construction
#############################

def save_graph_image(graph, filename: str) -> None:
    """
    Save a PNG image of the LangGraph diagram using Mermaid.

    (Mermaid support may require additional dependencies, and was commented
    in the original script.)
    """
    with open(filename, "wb") as file:
        file.write(Image(graph.get_graph().draw_mermaid_png()).data)


def build_graph(
    llm: ChatOllama,
    tokenizer: PreTrainedTokenizerFast,
    book_descriptions: list[str],
    book_vs,
):
    """
    Construct and compile the LangGraph workflow.
    """

    workflow = StateGraph(State)

    # Node wrappers that close over llm/tokenizer/book_descriptions/book_vs
    def get_question(state: State):
        return get_question_node(
            state=state,
            llm=llm,
            tokenizer=tokenizer,
            book_descriptions=book_descriptions,
            book_vs=book_vs,
        )

    def critique_node(state: State):
        return critique_of_experts(
            state=state,
            llm=llm,
            tokenizer=tokenizer,
            book_vs=book_vs,
        )

    def combined_node(state: State):
        return combined_answer_node(
            state=state,
            llm=llm,
            tokenizer=tokenizer,
        )

    def revise_node(state: State) -> Literal["reword_question", "final_answer"]:
        return revise_question(
            state=state,
            llm=llm,
            tokenizer=tokenizer,
        )

    def reword_node(state: State):
        return reword_question(
            state=state,
            llm=llm,
            tokenizer=tokenizer,
        )

    workflow.add_node("get_question", get_question)
    workflow.add_node("final_answer", generate_final_answer)

    # Expert nodes (book_0 .. book_4)
    for i in range(MAX_BOOKS):
        node_name = f"Expert {i}"
        print("Adding Node: ", node_name)
        expert_fn = make_expert_node(
            expert_id=i,
            llm=llm,
            tokenizer=tokenizer,
            book_vs=book_vs,
        )
        workflow.add_node(node_name, expert_fn)

    # Critique + combined nodes
    workflow.add_node("critique_of_experts", critique_node)
    workflow.add_node("combined_answer_node", combined_node)

    # Question revision nodes
    workflow.add_node("revise_question", revise_node)
    workflow.add_node("reword_question", reword_node)

    # Edges
    workflow.set_entry_point("get_question")

    for i in range(MAX_BOOKS):
        node_name = f"Expert {i}"
        workflow.add_edge("get_question", node_name)
        workflow.add_edge(node_name, "critique_of_experts")

    workflow.add_edge("critique_of_experts", "combined_answer_node")

    workflow.add_conditional_edges("combined_answer_node", revise_node)
    workflow.add_edge("reword_question", "get_question")
    workflow.set_finish_point("final_answer")

    graph = workflow.compile()
    return graph


#############################
# CLI entrypoint
#############################

def run_cli() -> None:
    """
    Run the interactive CLI loop, preserving the behavior of the original script.
    """
    global TX_char, TX_tokn, RX_char, RX_tokn

    print(
        """
#############################
#### SETUP INFERENCE AND ENVIRONMENT HYPERPARAMETERS
#############################
"""
    )

    print(f"# MOST NUMBER OF EXPERTS (BOOKS) TO QUERY - MAX IS 5 {MAX_BOOKS}")
    print(
        f"# MOST NUMBER OF RAG RETRIEVED CONTENT {K} CHUNKS -R RERANKED DOWN TO {TOP_N}"
    )
    print(f"# CHUNKING  {CHUNK_SIZE}, AND OVERLAP {CHUNK_OVERLAP}")
    print(
        f"# running count of token TX/RX {TX_tokn}/{RX_tokn} and characters TX to LLM, "
        f"RX received back from LLM {TX_char }/ {RX_char}"
    )
    print(f"# Maximum number of times to re-write the question {MAX_BOOKS}")

    print(
        f"# Context token count maximum (contexts+questions can get truncated if longer {NUM_CTX})"
    )

    print(f"# Model Selection - Ollama serving {LOCAL_LLM_NAME}")

    print(
        """
#############################
#### SETUP LLM(s)
#############################
"""
    )

    # LLMs
    llm = ChatOllama(model=LOCAL_LLM_NAME, temperature=0, num_ctx=NUM_CTX)
    # Kept for parity, though not currently used
    llm_json_mode = ChatOllama(
        model=LOCAL_LLM_NAME,
        temperature=0,
        format="json",
        num_ctx=NUM_CTX,
    )
    smarter_llm = ChatOllama(
        model=SMARTER_LLM_NAME,
        temperature=0,
        num_ctx=NUM_CTX,
    )

    # Embeddings
    embeddings = create_embeddings(EMBEDDING_MODEL_NAME)

    # Tokenizer
    tokenizer = build_tokenizer()

    print(
        """
#############################
#### SET UP A RAG VECTOR STORE FOR EACH BOOK
#############################
"""
    )

    book_vs, created_new_faiss = load_or_build_vectorstores(
        FILE_ROOTS,
        embeddings,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    num_books = len(book_vs)

    print(
        """
#############################
#### SUMMARIZING EACH BOOK, AND THEREFORE EACH AVAILABLE AGENT EXPERT
#############################
"""
    )

    book_descriptions = load_or_build_book_descriptions(
        book_vs=book_vs,
        llm=llm,
        tokenizer=tokenizer,
        created_new_faiss=created_new_faiss,
    )

    print("#############################")
    print(f"#### CREATING STATE VARIABLE WITH UP TO {MAX_BOOKS} EXPERTS")
    print("#############################")
    print("Define state variable for up to 5 books")

    # Build graph
    graph = build_graph(
        llm=llm,
        tokenizer=tokenizer,
        book_descriptions=book_descriptions,
        book_vs=book_vs,
    )

    # Optionally save mermaid diagram (commented as in original)
    # save_graph_image(graph, "parallel_book_vs_research.png")

    # Using native ASCII drawing
    # Skip drawing the ASCII graph of Agent Graph
    # print(graph.get_graph().print_ascii())

    print("Type in the Human question or instructions to AI, or leave blank to exit.\n")
    while True:
        try:
            human_msg = input("\n[Human]:")
            if not human_msg.strip():
                break
            result = graph.invoke(
                {
                    "original_human_question": human_msg,
                    "revised_question": human_msg,
                }
            )
            print("\n\n\nResults", result["combined_answer"])
            print(
                " TX_char ",
                TX_char,
                " TX_tokn ",
                TX_tokn,
                " RX_char ",
                RX_char,
                " RX_tokn ",
                RX_tokn,
            )
        except KeyboardInterrupt:
            print("KeyboardInterrupt --> Immediate program termination!")
            raise SystemExit

    print("Break here for debugging")
    raise SystemExit

