"""
Configuration and hyperparameters for the LGS multi-book RAG demo.
"""

# Maximum number of book experts to query
MAX_BOOKS: int = 5  # was 5

# Vector store retrieval parameters
K: int = 25  # was 5, increased for reranker; good results with 10, 5 for 4k token limits
TOP_N: int = 3  # reranker limit (currently not used, but preserved)

# Text chunking parameters
CHUNK_SIZE: int = 6000  # was 100
CHUNK_OVERLAP: int = 3000  # was 50

# Maximum number of times to re-write a question
MAX_REWRITE: int = 3

# Context token count maximum
NUM_CTX: int = 16384  # original comment: best seemed 4096, but script used 16384

# LLM model names served by Ollama
LOCAL_LLM_NAME: str = "llama3.2:3b"        # Pretty fast. Bad book ranking, good answer.
SMARTER_LLM_NAME: str = "gemma3:12b-it-qat"  # Excellent responses, slower.

# Embedding model name
EMBEDDING_MODEL_NAME: str = "llama3"

# Book / PDF roots (without .pdf suffix)
FILE_ROOTS = [
    "./BOOK_IntroductionToBusiness",
    "./BOOK_Entrepreneurship",
    "./BOOK_Marketing",
    "./BOOK_Financial_Accounting",
    "./BOOK_Managerial_Accounting",
    "./BOOK_Business_Law",
    "./CULINARY-calculations-simplified-math-for-culinary-professionals",
    "./CULINARY-nutrition-for-foodservice-and-culinary-professionals",
    "./CULINARY-real-restaurant-recipes-food-that-built-a-business",
    "./CULINARY-recipes-americas-favorite-restaurant-recipes-vol-1",
    "./CULINARY-restaurant-calorie-counter-for-dummies",
    "./CULINARY-restaurant-financial-basics",
    "./CULINARY-restaurant-law-basics",
    "./CULINARY-restaurant-management-career-starter",
    "./CULINARY-restaurant-service-basics",
    "./CULINARY-the-restaurant-from-concept-to-operation-sixth-edition",
]

# File used to cache book descriptions
BOOK_DESC_FILE: str = "book_desc.json"
