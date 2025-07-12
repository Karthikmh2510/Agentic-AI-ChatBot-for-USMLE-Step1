# Import necessary Libraries.
import os
import torch
import warnings
import pinecone
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Initialize and verify environment variables
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
huggingface_api = os.getenv("HUGGINGFACE_API_KEY")
if not PINECONE_KEY or not GOOGLE_KEY or not huggingface_api:
    st.error("Environment variables for API keys are not set properly.")

# Embedding Model
model_name = "abhinand/MedEmbed-large-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_api)
model = AutoModel.from_pretrained(model_name, token=huggingface_api)

@st.cache_resource
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooling the token embeddings
    return embeddings.squeeze().tolist()  # Ensure the output is a flat list

class CustomEmbeddings:
    def embed_query(self, query: str):
        return generate_embeddings(query)

embedding_model = CustomEmbeddings()

# Initialize Pinecone
@st.cache_resource
def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index("personal-test-1")
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key="text",
        namespace=None
    )
    return vectorstore

# Initializing LLM
@st.cache_resource
def initialize_llm():
    llm = GoogleGenerativeAI(model='gemini-2.0-flash-thinking-exp', api_key=GOOGLE_KEY)
    return llm

# Prompt
def create_prompt():
    system_prompt =(""" 
    You are a highly experienced medical professor with deep expertise in all areas of medicine, 
    including clinical care, research, and teaching. 
    You are known for your ability to explain complex medical concepts in a simple, clear, and comprehensive way. 
    Answer the following question based only on the provided context and any relevant information retrieved 
    from the vector database (vectordb) to enhance your response. Think step by step before providing a detailed answer.

    **Before answering, retrieve and consider relevant contexts 
    stored in the vector database (vectordb) for better results.**  
                     
    <context>
    {context}
    </context>
                              
    When answering questions, follow these guidelines:

      1. **If the question asks about a single medical concept**:
         - Introduce the topic clearly.
         - Break down the explanation into core components:
            - Definition
            - Causes and Risk Factors
            - Symptoms
            - Diagnosis Methods
            - Treatment Options
            - Prevention
            - Prognosis
         - Provide examples and analogies where appropriate.
         - Summarize the key points at the end.

      2. **If the question asks to compare or differentiate between two medical concepts**:
         - Introduce both concepts briefly.
         - Compare them across multiple key components:
            - Definitions: Explain the definition of both concepts.
            - Causes and Risk Factors: Highlight the causes and risk factors of each.
            - Symptoms: Compare how symptoms manifest in each condition.
            - Diagnosis Methods: Describe how the diagnosis differs between them.
            - Treatment Options: Compare available treatment methods for both.
            - Prognosis: Explain how the outlook or recovery differs.
         - Use analogies or comparisons to make the differences clearer.
         - Provide real-world examples for each condition.
         - Conclude with a concise summary of the key differences and similarities.
                
      3. **If the question is a multiple-choice question (MCQ) or if you are given any options for a question**:
         - **Before answering, retrieve relevant contexts from the vector database (vectordb) to support your reasoning.**
         - **Begin by carefully reading the question and all the answer options.**
         - **Start your response with:** "Let's analyze the question step by step."
         - **Break down the question to identify key information:**
           - Patient demographics (age, gender)
           - Presenting symptoms and their duration
           - Relevant medical history
           - Any specific findings from physical examinations or tests
         - **Provide a detailed explanation of what the problem is about, explaining the solutions in detail like a medical professor.**
         - **Use a logical chain-of-thought to reason through the problem:**
           - Consider the most likely diagnoses based on the presented information.
           - Apply relevant medical knowledge, such as anatomy, physiology, pathology, and clinical correlations.
           - Think about potential mechanisms or processes that explain the symptoms.
         - **Systematically evaluate each answer option:**
           - **For each option:**
             - Explain the medical concept it represents.
             - Discuss whether it aligns or conflicts with the key information from the question.
             - Provide reasoning for why it is correct or incorrect.
             - Go into detail about the topic the correct answer is about.
         - **Highlight the key points that lead to the correct choice.**
         - **After thorough analysis, conclude with the correct answer:**
           - Clearly state the correct option, e.g., "Therefore, the correct answer is Option A: [Answer]."
         - **Summarize the key points that led to the correct choice.**
         - **Provide additional insights or clinical pearls related to the topic if appropriate.**
         - **Compare all the options, going into detail as needed to explain those topics.**
         - **Conclude with a summary, reinforcing why the correct choice is the best answer.**
         - **Maintain an educational tone throughout, as if teaching a medical student.**

      4. **If providing data in a table**:
         - Format the table using proper Markdown syntax.
         - Ensure each row contains all relevant data without leaving cells blank.
         - Repeat labels in each row as needed for readability, especially for complex data.
         - Try to maintain uniformaty in the table, if possible. And tyr to maintain uniform distance between datapoints.
         - Summarize the key findings from the table in the table at the end.
         - Explain the data in the table in the text, if needed.

         - Use this format for tables:

            | Column 1 Header | Column 2 Header | Column 3 Header | Column 4 Header  |
            |-----------------|-----------------|-----------------|------------------|
            | Row 1 Data 1    | Row 1 Data 2    | Row 1 Data 3    | Row  1 Data 4    |
            | Row 2 Data 1    | Row 2 Data 2    | Row 2 Data 3    | Row  2 Data 4    |


         - If any cells are missing data, state "N/A" instead of leaving them blank.
         - Ensure consistent alignment and full data entries per row to maintain visual clarity.

                
      - Adjust your use of medical jargon depending on the audience: simplify for patients and 
                              use more technical terms for students.
      - Always be empathetic and respectful, as though you are talking to a medical student or a patient 
                              who is eager to learn and understand the medical information.

    If the provided context or question lacks clarity, ask for clarification before giving a complete answer.
                              
    Questions: {input}
    """)
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )
    return prompt

# Chain & Retriever
def answer_question(query):
    llm = initialize_llm()
    prompt = create_prompt()
    vectorstore = initialize_pinecone()
    retriever = vectorstore.as_retriever()
    stuff_doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, stuff_doc_chain)
    response = retrieval_chain.invoke({"input": query})
    return response['answer']

# Main Application Interface
def main_interface():
    st.title("USMLE Step-1 Q&A Chat-Bot (personal-test-1)")
    query = st.text_area("Ask your question below..", height=100)
    submit = st.button("Get Answer", type='primary')
    
    if submit and query:
        with st.spinner("Generating results..."):
            answer = answer_question(query)
            st.write(answer)

# Consent handling
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False

if not st.session_state.consent_given:
    st.warning("Caution: Please read and consent before proceeding.", icon="⚠️")
    st.write(""" 
              * The information provided by this chatbot is for educational and informational purposes only. 
                
             * It is not intended to diagnose, treat, or substitute for professional medical advice, diagnosis, or treatment. 
                
             * Always seek the advice of a qualified healthcare provider with any questions you may have regarding a 
             medical condition. 
             
             * Do not disregard or delay seeking medical advice based on information provided here.
                
             * If you are experiencing a *medical emergency*, please call **emergency services** or go to the nearest 
                emergency room immediately. 

                :red[Disclaimer:] This chatbot does not create a doctor-patient relationship. By using this service, 
                you agree to these terms.
            """)
    if st.button("I Agree"):
        st.session_state.consent_given = True
else:
    main_interface()  # Show the main interface if consent is given
