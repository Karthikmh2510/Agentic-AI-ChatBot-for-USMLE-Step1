import os
import sys
import warnings
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools.retriever import create_retriever_tool
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
import torch

from logger import get_logger
from custome_exception import CustomException

logger = get_logger(__name__)
logger.info("Agentic RAG Started")

# ==============================================================================================
# 1. ENVIRONMENT                                                                               |
# ==============================================================================================

load_dotenv()
warnings.filterwarnings("ignore")

# Primary secrets
OPENAI_KEY      = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_KEY    = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")

# Optional LangSmith tracing
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT

if not OPENAI_KEY:
    raise CustomException("OPENAI_API_KEY is missing", sys)
if not PINECONE_KEY:
    raise CustomException("PINECONE_API_KEY is missing", sys)
if not TAVILY_API_KEY:
    raise CustomException("TAVILY_API_KEY is missing", sys)
if not HUGGINGFACE_API:
    raise CustomException("HUGGINGFACE_API_KEY is missing", sys)

logger.info("Environment variables loaded")

# ==============================================================================================
# 2. LLM INITIALISATION                                                                        |
# ==============================================================================================

def init_llm(api_key: str):
    try:
        llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", api_key=api_key)
        logger.info("OpenAI LLM initialised")
        return llm
    except Exception as exc:
        logger.error("Failed to initialise LLM")
        raise CustomException(str(exc), sys) from exc

LLM = init_llm(OPENAI_KEY)

# ==============================================================================================
# 3. EMBEDDINGS                                                                                |
# ==============================================================================================

def init_embedding_model(model_name: str, hf_key: str | None):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_key)
        model = AutoModel.from_pretrained(model_name, token=hf_key)
        logger.info("Embedding model loaded: %s", model_name)

        def generate_embeddings(text: str):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooling the token embeddings
            return embeddings.squeeze().tolist()  # Ensure the output is a flat list

        class Embed:
            def embed_query(self, query: str):  # noqa: D401, N802
                return generate_embeddings(query)

        return Embed()
    except Exception as exc:
        logger.error("Failed to load embedding model")
        raise CustomException(str(exc), sys) from exc

EMBEDDINGS = init_embedding_model("abhinand/MedEmbed-large-v0.1", HUGGINGFACE_API)

# ==============================================================================================
# 4. VECTOR STORE & RETRIEVER                                                                  |
# ==============================================================================================

def init_vectorstore(pinecone_key: str):
    try:
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("personal-test-1")
        vs = PineconeVectorStore(index=index, embedding=EMBEDDINGS, text_key="text", namespace=None)
        logger.info("Vector store initialised")
        return vs.as_retriever()
    except Exception as exc:
        logger.error("Vector store initialisation failed")
        raise CustomException(str(exc), sys) from exc

RETRIEVER = init_vectorstore(PINECONE_KEY)

# Build retriever tool -------------------------------------------------------------------------
RETRIEVER_TOOL = create_retriever_tool(
    retriever=RETRIEVER,
    name="retrieve_medical_knowledge",
    description=(
        "Search and return authoritative medical information from the Pinecone vector store. "
        "The index contains peer-reviewed clinical studies, treatment guidelines, drug monographs, "
        "and other curated healthcare content. Invoke this tool **only** when the user explicitly asks a medical-related "
        "question requiring sourced data from the index (e.g., disease mechanisms, therapy options, statistics, references). "
        "For any non-medical query, casual greeting (‘hi’, ‘hello’, ‘how are you’), or general conversation, respond directly "
        "without using this tool."
    ),
)

# ==============================================================================================
# 5. PROMPTS                                                                                   |
# ==============================================================================================

SYSTEM_PROMPT = (
    "You are a senior USMLE professor. Answer only with correct medical information. "
    "Cite your reasoning step by step before giving the final answer."
)

ASSISTANT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

RAG_PROMPT = PromptTemplate(
    template=(
        SYSTEM_PROMPT
        + "\n\n<context>\n{context}\n</context>\n\n"
        + "Question: {question}\nAnswer step‑by‑step:"
    ),
    input_variables=["context", "question"],
)

GRADER_PROMPT = PromptTemplate(
    template=(
        "You are a grader deciding if a retrieval context answers the question.\n"
        "Context: {context}\n\nQuestion: {question}\n\nRespond with exactly 'yes' or 'no'."
    ),
    input_variables=["context", "question"],
)

logger.info("Core prompts prepared")

# ==============================================================================================
# 6. GRAPH NODES                                                                               |
# ==============================================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Tavily ---------------------------------------------------------------------------------------
TAVILY = TavilySearchResults(api_key=TAVILY_API_KEY)

def tavily_search(state: AgentState):
    logger.info("Tavily search invoked")
    question = state["messages"][0].content
    snippets = TAVILY.invoke(question) if TAVILY_API_KEY else []
    joined = "\n\n---\n\n".join(snippets)
    return {"messages": [AIMessage(content=joined or "No web results found.")]}  # fallback msg

# Assistant ------------------------------------------------------------------------------------

def ai_assistant(state: AgentState):
    messages = state["messages"]
    if len(messages) > 1:
        question = messages[-1].content
        response = (ASSISTANT_PROMPT | LLM).invoke({"question": question})
        return {"messages": [response]}

    response = LLM.bind_tools([RETRIEVER_TOOL]).invoke(messages)
    return {"messages": [response]}

# Grade schema ---------------------------------------------------------------------------------
class Grade(BaseModel):
    binary_score: str = Field(description="'yes' if relevant else 'no'")

def grade_documents(state: AgentState) -> Literal["Output_Generator", "Query_Rewriter", "Tavily_Search"]:
    docs = state["messages"][-1].content
    if not docs.strip():
        return "Tavily_Search"
    
    chain = GRADER_PROMPT | LLM.with_structured_output(Grade)
    result = chain.invoke({
        "question": state["messages"][0].content,
        "context": docs,
    })
    return "Output_Generator" if result.binary_score.lower() == "yes" else "Query_Rewriter"

# Output generator -----------------------------------------------------------------------------

def generate_answer(state: AgentState):
    logger.info("Generating final answer")
    chain = RAG_PROMPT | LLM
    response = chain.invoke({
        "context": state["messages"][-1].content,
        "question": state["messages"][0].content,
    })
    return {"messages": [response]}

# Query rewriter -------------------------------------------------------------------------------

def rewrite_query(state: AgentState):
    logger.info("Rewriting query for retrieval optimisation")
    prompt = (
        "Rewrite this question so that a retrieval system can fetch the most relevant clinical information.\n"
        f"Original: {state['messages'][0].content}\nRewritten:"
    )
    response = LLM.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# ==============================================================================================
# 7. WORKFLOW COMPILATION                                                                      |
# ==============================================================================================

def build_graph():
    logger.info("Compiling LangGraph workflow")
    retrieve_node = ToolNode([RETRIEVER_TOOL])

    wf = StateGraph(AgentState)
    wf.add_node("AI_Assistant", ai_assistant)
    wf.add_node("Vector_Retriever", retrieve_node)
    wf.add_node("Output_Generator", generate_answer)
    wf.add_node("Query_Rewriter", rewrite_query)
    wf.add_node("Tavily_Search", tavily_search)

    wf.add_edge(START, "AI_Assistant")

    wf.add_conditional_edges("AI_Assistant", 
                             tools_condition, 
                             {"tools": "Vector_Retriever", 
                              END: END})

    wf.add_conditional_edges("Vector_Retriever",
                            grade_documents,
                            {"Output_Generator": "Output_Generator",
                            "Query_Rewriter": "Query_Rewriter",
                            "Tavily_Search": "Tavily_Search"})
    
    wf.add_edge("Tavily_Search", "Output_Generator")
    wf.add_edge("Output_Generator", END)
    wf.add_edge("Query_Rewriter", "AI_Assistant")

    return wf.compile()

GRAPH = build_graph()

# ==============================================================================================
# 8. PUBLIC API                                                                                |
# ==============================================================================================

def ask(question: str) -> str:
    """High‑level helper that returns the assistant's answer string."""
    try:
        logger.info("Question received: %s", question.replace("\\n", " ")[:100])
        result = GRAPH.invoke({"messages": [question]})
        answer = result["messages"][-1].content
        logger.info("Answer generated")
        return answer
    except Exception as exc:
        logger.error("Error during ask()")
        raise CustomException(str(exc), sys) from exc

# ==============================================================================================
# 9. CLI / Quick demo                                                                          |
# ==============================================================================================

if __name__ == "__main__":
    demo_q = "Give the table of lysosomal storage disorders"
    print(ask(demo_q))
