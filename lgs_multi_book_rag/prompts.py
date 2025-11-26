"""
Prompt templates used by the LGS multi-book RAG demo.
"""

rag_prompt = """
You are an AI Agentic expert equipped with the vector database of a textbook and your task is to help a Coordinating AI Agent answer a Human question. The Human posed a question to the The Coordinating AI Agent who then in turn is asking you for expert assistance in answering the question. Prior to responding, a vector database semantic similarity search was performed, extracting a number of pages from the textbook that were semantically similar to the question.
Those extracted pages will be your expert context.
Use the expert context to assist the Coordinating AI Agent with answering the Human question by providing unique information that is contained in the expert context. Since the Coordinating AI Agent was trained on the same data that you were, give preference to direct quotes from the expert context and other information you found new and/or you feel will help to answer the Human question. 
Here is the expert context to use to answer the question:
{context} 
Think carefully about the above context. 
Now, review the Human question:
{question}
Keep the answer on-topic.
Answer:
"""

final_answer_prompt = """
You are an agentic AI controller.
You were asked a question from the Human. 
Prior to responding, you created a number of expert agents who have been using different textbooks for information to best answer the question. Those expert agents responded to you with their answer, and they gave preference to direct quotes from their textbook, and indicating what portions of the answer are not from the book, but rather from their own knowledge.
You must read their cumulative (concatenated) answers and thoughtfully combine them to create a final and complete answer. 
Here are the answers from the expert agents for you to use to answer the question:
{context} 
Think carefully about the above answers. 
Now, review the user question:
{question}
Provide an answer to this questions using the above context. 
Keep the answer on-topic, and feel free to include all important details from the expert agents.
Answer:
"""

book_description_prompt = """
You are to describe the contents of this book. It may be contained in the preamble, the introduction, the abstract, or other overview section of the book that explains the contents in chapters and sections that make up the subject matter of the book. In your description, highlight what topics are covered in the book. Also, what step-by-step instructions, steps, milestones, nodes, edges, paths, roads, or procedures are covered?
"""

# ORIGINAL RANKING PROMPT (last definition in the original script)
ranking_prompt = """
I will provide you a book description as well as a question.
You are to rank, from 1 to 10, how well the contents of a book with such a book description is likely to be able to be helpful in answering the Human question or provide valuable and/or unique guidance towards answering the Human question. Provide this rank as a single integer response from 1 to 10, with 10 being the best.
Do not provide any other introductory or explanatory text.
Only reply with the integer response.
Here is the Human question: {question}
Here is the book description: {context}
"""

critique_prompt = """
You are a critic analyzing if an response (provided by an Expert AI agent using a textbook as context) is good or bad at helping a Coordinator AI Agent in answering a Human question. The Expert AI agent gave preference to direct quotes from their textbook, as well as information that is unique, new, and/or helpful to the Coordinator AI agent in answering the Human question. 
You should carefully read the Human question as well as the Expert AI agent's response and reply whether the answer is good or bad.
Provide only a single word answer, either "good" or "bad" depending on your analysis.
The Human question is: {question}
The answer the Expert AI Agent using their textbook provided is: {answer}
"""

reword_prompt = """
You are a helpful agent trying to get the best answers to the human provided question.
You discovered that better results may be achieved if you reword the question slightly differently.
You may have to repeat this process, revising the quesiton more than once.
You should carefully read the original human question as well as the the most recently created revised version of that question (which also failed to produce the desired response).
Your response should be a suggested wording of the original human question that will yield hopefully better results.
Do not include introductory or explanatory wording. Do not ask questions. Simply provide your best effort at a newly suggested revised question only.
The original human question was: {original_human_question}
The most recently revised version of the question was: {revised_question}
 
"""
