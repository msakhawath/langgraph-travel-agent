# AI Travel Booking Agent — LangGraph Multi-Agent System

A production-grade multi-agent AI application that plans complete trips using a **LangGraph** agentic pipeline with **persistent long-term memory** backed by PostgreSQL.

> Built by [Sakhawat Hossain](https://sakhawatanalytics.com/) · [GitHub](https://github.com/msakhawath)

---

## What It Does

Type a natural-language travel request like *"Plan a 7-day Japan trip under $3000 including flights and hotels"* and the system:

1. **Flight Agent** — queries live flight data via AviationStack API
2. **Hotel Agent** — searches real-time hotel options via Tavily web search
3. **Itinerary Agent** — generates a day-by-day itinerary using LLaMA 3.3 70B (Groq)
4. **Final Agent** — synthesises everything into a polished travel plan
5. **Memory** — every conversation is saved to PostgreSQL so the agent remembers your past trips across sessions

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Flight Agent│───▶│ Hotel Agent │───▶│ Itinerary Agent  │───▶│ Final Agent │
│ AviationStack│   │ Tavily Search│   │ LLaMA 3.3 70B   │    │ LLaMA 3.3  │
└─────────────┘    └─────────────┘    └──────────────────┘    └─────────────┘
                                                                      │
                                              PostgreSQL Checkpointer ◀┘
                                              (persistent memory per user)
```

**Stack:** LangGraph · LangChain · Groq (LLaMA 3.3 70B) · Tavily · AviationStack · PostgreSQL · Streamlit · psycopg3

---

## Key Technical Highlights

- **Multi-agent orchestration** with LangGraph `StateGraph` — each agent is an independent node with typed state
- **Persistent memory** using `PostgresSaver` — conversations persist across browser sessions per `thread_id`
- **Real API integrations** — live flight data (AviationStack) and web search (Tavily)
- **LLM-powered planning** — Groq-hosted LLaMA 3.3 70B for fast, high-quality itinerary generation
- **Streamlit frontend** — clean UI with quick-fill prompts, conversation history, and per-user session isolation

---

## Setup

### Prerequisites
- Python 3.10+
- PostgreSQL running locally

### 1. Clone & install
```bash
git clone https://github.com/msakhawath/langgraph-travel-agent.git
cd langgraph-travel-agent
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_key
TAVILY_API_KEY=your_tavily_key
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/langgraph_memory
```

Get free API keys:
- Groq: https://console.groq.com
- Tavily: https://tavily.com
- AviationStack: https://aviationstack.com

### 3. Create the database
```sql
CREATE DATABASE langgraph_memory;
```

### 4. Run
```bash
streamlit run frontend.py
```

---

## Project Structure

```
├── main.py              # LangGraph graph definition & PostgreSQL checkpointer
├── frontend.py          # Streamlit UI
├── tools/
│   ├── flight_tool.py   # AviationStack flight search
│   └── tavily_tool.py   # Tavily web search for hotels
├── requirements.txt
└── .env                 # (not committed — see above)
```

---

## Skills Demonstrated

`LangGraph` `Multi-Agent Systems` `LangChain` `Groq API` `LLaMA 3.3` `PostgreSQL` `psycopg3` `Streamlit` `Tavily` `Python` `REST APIs` `Agentic AI`
