# Live AI Assistant

A production-grade AI assistant that searches the web in real time, verifies facts, and remembers conversations across sessions. Built entirely on free infrastructure.

## What it does

You ask a question, it decides whether to search the web or answer directly, pulls live results from Tavily, synthesizes a response using LLaMA 3.3 70B running on Groq, and stores the conversation in memory. Toggle persistent memory on and it remembers past sessions using ChromaDB vector search.

## Stack

| Layer | Tool |
|---|---|
| LLM | LLaMA 3.3 70B via Groq (free) |
| Agent framework | LangGraph |
| Web search | Tavily API (free) |
| Memory | Session dict + ChromaDB |
| UI | Gradio |
| Hosting | Hugging Face Spaces (free) |

## Architecture
## Author

Mohan Savendra Tikkireddy — AI/ML Engineer
[LinkedIn](https://www.linkedin.com/in/mohan-savendra-tikkireddy-b59897220/) · [Portfolio](https://mohan-ai-portfolio.vercel.app/)
