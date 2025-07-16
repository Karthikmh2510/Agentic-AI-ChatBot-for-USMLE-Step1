import os
import sys
import warnings
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_tavily.tavily_search import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools.retriever import create_retriever_tool
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
import torch

from src.logger import get_logger
from src.custome_exception import CustomException

from judgeval.integrations.langgraph import JudgevalCallbackHandler
from judgeval.common.tracer import Tracer
from judgeval.scorers import AnswerRelevancyScorer,FaithfulnessScorer

# Initialize logger
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
JUDGMENT_API_KEY = os.getenv("JUDGMENT_API_KEY")
JUDGMENT_ORG_ID = os.getenv("JUDGMENT_ORG_ID")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

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
if not JUDGMENT_API_KEY:
    raise CustomException("JUDGMENT_API_KEY is missing", sys)
if not JUDGMENT_ORG_ID:
    raise CustomException("JUDGMENT_ORG_ID is missing", sys)
if not LANGSMITH_API_KEY:
    raise CustomException("LANGSMITH_API_KEY is missing", sys)
if not LANGSMITH_TRACING:
    raise CustomException("LANGSMITH_TRACING is missing", sys)
if not LANGSMITH_ENDPOINT:
    raise CustomException("LANGSMITH_ENDPOINT is missing", sys)
if not PINECONE_INDEX:
    raise CustomException("PINECONE_INDEX is missing", sys)

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
        index = pc.Index(PINECONE_INDEX)
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
    "You are a senior USMLE Step 1 tutor. "
    "Respond conversationally: give a concise clinical rationale (2–4 short sentences) "
    "and then state the single best answer clearly on its own line, e.g. **Answer: Tamsulosin**. "
    "Avoid numbered 'Step‑1 / Step‑2' formats or long bullet lists. "
    "► If the learner explicitly asks for a *detailed* explanation, expand your rationale to a fuller paragraph before the final answer."
)


ASSISTANT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

RAG_PROMPT = PromptTemplate(
    template=(
        SYSTEM_PROMPT
        + "\n\n<context>\n{context}\n</context>\n\n"
        + "Question: {question}\nAnswer:"
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
# Tavily Tool
tavily_tool = TavilySearch(max_results=5, topic="general", language="en",
                           include_answer=True, include_raw_content=True,
                           description=("Use this tool for real-time news or updates. "
                                    "Only call when question requires up-to-date info."))

def tavily_search(state: AgentState):
    logger.info("Tavily search invoked")
    question = state["messages"][0].content
    snippets = tavily_tool.invoke(question) if TAVILY_API_KEY else []
    joined = "\n\n---\n\n".join(snippets)
    return {"messages": [AIMessage(content=joined or "No web results found.")]}  # fallback msg

# Assistant ------------------------------------------------------------------------------------

def ai_assistant(state: AgentState):
    messages = state["messages"]
    if len(messages) > 1:
        question = messages[-1].content
        response = (ASSISTANT_PROMPT | LLM).invoke({"question": question})
        return {"messages": [response]}

    response = LLM.bind_tools([RETRIEVER_TOOL, tavily_tool]).invoke(messages)
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

# JudgmentLabs Evaluation for our LLM ----------------------------------------------------------
def judgment_eval(state:AgentState):
    logger.info("Evaluation for Relevancy and Faithfulness begins.....")
    judgment = Tracer(project_name="USMLE_Step1", api_key= JUDGMENT_API_KEY,organization_id=JUDGMENT_ORG_ID)

    user_input = str(state["messages"][0].content)
    llm_output = str(state["messages"][-1].content)
    model_name="gpt-4.1-mini-2025-04-14"

    retrieve_from_vectorstores = []
    for i in RETRIEVER.invoke(user_input):
        retrieve_from_vectorstores.append(i.page_content)
    retrieval_context = list(retrieve_from_vectorstores[0])

    judgment.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.8),FaithfulnessScorer(threshold=0.8)],
        input=user_input,
        actual_output=llm_output,
        model=model_name,
        retrieval_context=retrieval_context
    )

    return state


# ==============================================================================================
# 7. WORKFLOW COMPILATION                                                                      |
# ==============================================================================================

def build_graph():
    logger.info("Compiling LangGraph workflow")
    retrieve_node = ToolNode([RETRIEVER_TOOL, tavily_tool])

    wf = StateGraph(AgentState)
    wf.add_node("AI_Assistant", ai_assistant)
    wf.add_node("Vector_Retriever", retrieve_node)
    wf.add_node("Output_Generator", generate_answer)
    wf.add_node("Query_Rewriter", rewrite_query)
    wf.add_node("Tavily_Search", tavily_search)
    wf.add_node("judgment_eval", judgment_eval)

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
    
    wf.add_edge("Tavily_Search", "AI_Assistant")
    wf.add_edge("Tavily_Search", "Output_Generator")
    wf.add_edge("Output_Generator", "judgment_eval")
    wf.add_edge("judgment_eval", END)
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

        judgment = Tracer(project_name="USMLE_Step1", api_key= JUDGMENT_API_KEY,organization_id=JUDGMENT_ORG_ID)
        handler = JudgevalCallbackHandler(judgment)
        initial_state = {"messages": [question]}
        config_with_callbacks = {"callbacks": [handler]}

        result = GRAPH.invoke(initial_state, config=config_with_callbacks)
        answer = result["messages"][-1].content
        logger.info("Answer generated")

        logger.info("Executed Nodes: %s", handler.executed_nodes)
        logger.info("Executed Tools: %s", handler.executed_tools)
        logger.info("Node/Tool Flow: %s", handler.executed_node_tools)

        return answer
    except Exception as exc:
        logger.error("Error during ask()")
        raise CustomException(str(exc), sys) from exc
    
# ==============================================================================================
# 9. CLI / Quick demo                                                                          |
# ==============================================================================================

if __name__ == "__main__":
    demo_q = "Give the table of lysosomal storage disorders"
    demo_q1 = "What is the latest news for USMLE Step1 examination?"

    question1=''' 
        A 76-year-old African American man presents to his primary care 
        provider complaining of urinary frequency. He wakes up 3-4 times per night 
        to urinate while he previously only had to wake up once per night. He also 
        complains of post-void dribbling and difficulty initiating a stream of 
        urine. He denies any difficulty maintaining an erection. His past medical 
        history is notable for non-alcoholic fatty liver disease, hypertension, 
        hyperlipidemia, and gout. He takes aspirin, atorvastatin, enalapril, and 
        allopurinol. His family history is notable for prostate cancer in his 
        father and lung cancer in his mother. He has a 15-pack-year smoking 
        history and drinks alcohol socially. On digital rectal exam, his prostate 
        is enlarged, smooth, and non-tender. Which of the following medications is 
        indicated in this patient? 
        Options: 
        A: Clonidine, 
        B: Hydrochlorothiazide, 
        C: Midodrine, 
        D: Oxybutynin, 
        E: Tamsulosin 
        '''
    
    # print(ask(demo_q))
    print(ask(demo_q1))
