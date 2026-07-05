# Live AI Assistant

Real-time AI assistant with web search, fact verification, and persistent memory.
Built with LangGraph + Groq + Tavily + ChromaDB + Gradio. **100% free to run.**

## Stack

| Layer | Tool | Cost |
|---|---|---|
| LLM | Groq (LLaMA 3.1 70B) | Free tier |
| Web Search | Tavily | Free (1000 searches/month) |
| Agent | LangGraph | Open source |
| Memory | ChromaDB + session dict | Free / local |
| UI | Gradio | Open source |
| Deploy | Hugging Face Spaces | Free |

## Project Structure

```
live-ai-assistant/
├── app.py          # Gradio UI + entry point
├── agent.py        # LangGraph agent + routing logic
├── tools.py        # Tavily search + verify tools
├── memory.py       # Session + ChromaDB memory manager
├── config.py       # API key loading
├── requirements.txt
├── .env.example
└── README.md
```

## Local Setup

### 1. Get Free API Keys

- **Groq:** https://console.groq.com → Sign up → Create API key (free, no card)
- **Tavily:** https://app.tavily.com → Sign up → Get API key (free, 1000/month)

### 2. Clone and Install

```bash
git clone <your-repo>
cd live-ai-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
cp .env.example .env
# Edit .env and paste your API keys
```

### 4. Run Locally

```bash
python app.py
# Open http://localhost:7860
```

## Deploy to Hugging Face Spaces (Free)

### 1. Create a Space

- Go to https://huggingface.co/spaces
- Click "Create new Space"
- SDK: **Gradio**
- Visibility: Public or Private

### 2. Push Your Code

```bash
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git add .
git commit -m "initial commit"
git push origin main
```

### 3. Add Secrets in HF Settings

In your Space → Settings → Repository secrets:
```
GROQ_API_KEY = your_key
TAVILY_API_KEY = your_key
```

That's it. HF Spaces runs Gradio apps natively.

## How It Works

```
User Input
    ↓
LangGraph Agent (LLaMA 3.1 70B on Groq)
    ↓
Router: needs search? → yes/no
    ↓              ↓
Tavily Search   Direct Answer
    ↓
Synthesize + cite sources
    ↓
Session Memory (always on)
ChromaDB Memory (toggle on/off in UI)
    ↓
Gradio Response
```

## Memory Modes

- **Session only (default):** Remembers context within the current conversation. Resets on restart.
- **Persistent (toggle on):** Stores all Q&A pairs in ChromaDB locally. Retrieves semantically similar past conversations on every new query. Survives restarts.

## Agent Tools

| Tool | When Used |
|---|---|
| `web_search` | Current events, live data, recent news, prices, anything time-sensitive |
| `verify_fact` | User asks to fact-check or verify a specific claim |

The agent decides which tool to call (or whether to call any) based on the query type.
