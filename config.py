import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GROQ_MODEL = "llama-3.1-70b-versatile"
MAX_SEARCH_RESULTS = 5
CHROMA_PERSIST_DIR = "./chroma_db"
SESSION_MEMORY_LIMIT = 20  # max messages kept in session memory
