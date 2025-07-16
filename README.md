# Agentic AI Chatbot for **USMLE Step‑1**



## 1  Overview

This project is an end‑to‑end, **retrieval‑augmented‑generation (RAG)** system that answers United States Medical Licensing Examination (USMLE) Step‑1 questions. It couples a custom medical knowledge base stored in Pinecone with an agentic reasoning graph built with LangGraph. The chatbot is exposed through a lightweight **Flask** API and an interactive **Streamlit** front‑end, and is fully containerised with separate Docker images for the back‑end and front‑end plus a `docker‑compose` orchestrator.



## 2  Motivation

Studying for Step‑1 requires rapid recall of thousands of facts spread across diverse sources. Traditional flash‑card or static Q‑bank apps cannot personalise explanations in real time. The goal of this project is to provide an **on‑demand tutor** that can:

* surface the most relevant authoritative content for any high‑yield question;
* justify each answer with concise clinical reasoning; and
* improve continuously through automated tracing & evaluation.



## 3  Goals & Objectives

1. **High‑precision retrieval.** Embed curated medical texts with `MedEmbed‑large‑v0.1` and store them in Pinecone for millisecond‑level semantic search.
2. **Agentic reasoning.** Orchestrate document retrieval, query rewriting, web search fall‑back, answer generation, and grading through a LangGraph state machine.
3. **Observability out of the box.** Collect structured traces in **LangSmith** and score every response for faithfulness & relevancy via **JudgmentLabs (Judgeval)**.
4. **Seamless deployment.** Provide a one‑command `docker‑compose` stack that spins up the API and UI with sensible defaults.



## 4  Solution Approach

### 4.1  RAG Pipeline

```
User ➜ Streamlit ➜ Flask /chat ➜ LangGraph Agent
          │                    │
          │                    ├─▶ RetrieverTool (Pinecone)
          │                    ├─▶ Query‑Rewriter (LLM)
          │                    ├─▶ Tavily Web Search
          │                    └─▶ Answer Generator (LLM)
          ▼
        JSON Response ➜ Streamlit renderer
```
<!-- Architecture diagram -->
<p align="center">
          <img width="520" alt="image" src="https://github.com/user-attachments/assets/dbc01165-83bb-4069-80ea-f07202ab81ef" />
</p>


1. **Embedding & Storage :** All source documents are embedded offline with `MedEmbed‑large‑v0.1` and pushed to a dedicated Pinecone index.
2. **Conversation Flow :** The LangGraph graph routes each user message through specialised nodes. If retrieval fails a grader node triggers query rewriting or real‑time web search.
3. **Answer Format :** The assistant returns a 2‑4 sentence rationale followed by the single best answer on a new line (e.g. `**Answer: Tamsulosin**`).

### 4.2  Tracing & Evaluation

* **LangSmith**: Every run records prompts, tool calls, latencies, and token usage. This enables offline inspection and prompt tuning.
* **JudgmentLabs**: Each response is automatically scored with *Answer Relevancy* and *Faithfulness* scorers (threshold ≥ 0.8). Scores and execution graphs are saved back to LangSmith for unified observability.

### 4.3  Front‑end

The Streamlit UI mimics a chat interface, persists history with `shelve`, and calls the back‑end REST endpoint. A custom CSS layer (`style/chatbot_style.py`) applies a dark mode theme and brand imagery.



## 5  Tech Stack

| Layer           | Technology                                           | Purpose                              |
| --------------- | ---------------------------------------------------- | ------------------------------------ |
| LLM / Reasoning | **OpenAI GPT‑4.1‑mini**                              | Primary language model               |
| Embeddings      | **MedEmbed‑large‑v0.1 (HF)**                         | Domain‑specific text embeddings      |
| Vector Store    | **Pinecone**                                         | Approximate‑nearest‑neighbour search |
| Orchestration   | **LangChain + LangGraph**                            | Tool binding & state graph           |
| Evaluation      | **JudgmentLabs (Judgeval)**                          | Automated scoring                    |
| Tracing         | **LangSmith**                                        | End‑to‑end run tracing               |
| Back‑end        | **Flask + uvicorn via uv**                           | REST API & health checks             |
| Front‑end       | **Streamlit**                                        | Chat UI                              |
| Infrastructure  | Docker (backend & frontend images), `docker‑compose` | Local / edge deployment              |



## 6  Repository Layout

```
.
├─ .streamlit/            # Streamlit config
├─ artifacts/             # Persisted chat history & logs
├─ notebook/              # Exploration / dataset prep
├─ src/
│   ├─ Agentic_RAG_Evaluation.py  # Core LangGraph workflow
│   ├─ logger.py         # Structured logging helper
│   └─ custome_exception.py
├─ utils/                 # Misc helpers
├─ style/                 # CSS & images
├─ app.py                 # Flask entry‑point
├─ streamlit_app.py       # Streamlit UI entry‑point
├─ Dockerfile.backend     # Build backend image
├─ Dockerfile.frontend    # Build frontend image
├─ docker‑compose.yml     # Orchestrate both services
└─ requirements.txt / pyproject.toml
```



## 7  Getting Started

### 7.1  Local Development (Python ≥ 3.11)

```bash
# Clone & enter
$ git clone https://github.com/Karthikmh2510/Agentic-AI-ChatBot-for-USMLE-Step1.git
$ cd Agentic-AI-ChatBot-for-USMLE-Step1

# Install deps (uv recommended)
$ pip install uv
$ uv pip install -r requirements.txt

# Environment
$ cp .env.example .env  # edit API keys & Pinecone index name

# Run services
$ python app.py          # starts backend on :8080
$ streamlit run streamlit_app.py  # starts UI on :8501
```

### 7.2  Containerised Deployment

1. **Build images** (optional – Compose will build if they’re missing):

   ```bash
   docker build -f Dockerfile.backend -t usmle-rag-backend:latest .
   docker build -f Dockerfile.frontend -t usmle-rag-frontend:latest .
   ```
2. **Start the stack**:

   ```bash
   docker-compose up --build -d
   ```

   * Back‑end: `http://localhost:8080`  \* Front‑end: `http://localhost:8501`
3. **Shut down**:

   ```bash
   docker-compose down
   ```

### 7.3  Environment Variables

Minimum keys required in `.env`:

```env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX=
HUGGINGFACE_API_KEY=...
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
JUDGMENT_API_KEY=...
JUDGMENT_ORG_ID=...
```

## 8  Usage

Open the Streamlit UI, type any Step‑1 style prompt (e.g. *“Compare Crohn disease with ulcerative colitis.”*) and press Enter. The UI renders the dialogue with markdown support, tables, and inline LaTeX where relevant. Click **Clear Chat** in the sidebar to reset the session.


## 9  Observability & Quality Checks

* **Trace Viewer:** Each request appears in your \[LangSmith] dashboard with time‑stamped nodes and token‑level costs.
* **Judgeval Reports:** After the answer node executes, two scorers evaluate relevancy & faithfulness using the retrieved context. Failing scores are logged for prompt refinement.



## 10  Roadmap

* 🔍 Add Per‑question *explain‑why‑score* feedback
* 🏥 Integrate UMLS & PubMed abstracts to widen knowledge base
* ⚡ Serve models through GPU‑enabled inference API (e.g. Groq or Together.ai) for faster answers
* 📈 Grafana / Prometheus metrics for latency & throughput monitoring



## 11  License & Disclaimer

This work is released under the **MIT License**. All content is for **educational purposes only** and must **not** be used to make clinical decisions. Always consult a qualified physician for medical advice.
