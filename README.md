# RAG-Powered GenAI Application for USMLE Step-1 Query Answering

## Overview
This project is a **RAG-Powered GenAI Application for USMLE Step-1 Query Answering** powered by **Google Gemini 2.0**, **Pinecone Vector Database**, and **Hugging Face Medical Embeddings**. It allows users to ask medical-related questions and receive detailed, context-aware answers based on medical documents stored in a vector database.

## Features
- **Medical Query Answering**: Provides detailed answers to USMLE Step-1 related questions.
- **Vector Database Integration**: Uses **Pinecone** for efficient retrieval of medical knowledge.
- **Hugging Face Medical Embeddings**: Generates embeddings using **MedEmbed-large-v0.1** for better context understanding.
- **Google Gemini 2.0 Model**: Utilizes the **Gemini-2.0-flash-thinking-exp** model for generating responses.
- **Streamlit UI**: A simple and interactive web interface for users.
- **Multi-Format Answering**: Provides structured responses including text, tables, and medical comparisons.
- **Consent Management**: Ensures users acknowledge a medical disclaimer before using the chatbot.

## Tech Stack
- **Python**
- **Streamlit** (Frontend UI)
- **Pinecone** (Vector Database)
- **LangChain** (LLM Orchestration)
- **Google Generative AI** (Gemini LLM)
- **Transformers** (Hugging Face Models)
- **Torch** (Deep Learning Backend)
- **dotenv** (Environment Variables Management)

## Installation
### Prerequisites
- Python 3.8+
- API keys for **Pinecone**, **Google Generative AI**, and **Hugging Face**

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Kathikmh2510/USMLE-RAG.git
   cd USMLE-RAG
   ```
2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Environment Variables**
   Create a `.env` file and add the required API keys:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   GOOGLE_API_KEY=your_google_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```
5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the **Streamlit UI** in your browser.
2. Enter a **medical question** related to USMLE Step-1.
3. Click **Get Answer**.
4. The chatbot retrieves relevant medical information and provides an **accurate, structured response**.

## Project Structure
```
|-- app.py                  # Main Streamlit application
|-- requirements.txt        # Required dependencies
|-- .env.example            # Example environment file
|-- utils/                  # Helper functions
|-- models/                 # Embedding and LLM initialization
|-- data/                   # Optional dataset storage
```

## API Workflow
1. **User Input**: A question is entered in the UI.
2. **Text Embeddings**: The question is embedded using `MedEmbed-large-v0.1`.
3. **Vector Search**: Pinecone retrieves relevant documents.
4. **LLM Processing**: Gemini 2.0 generates a response based on retrieved context.
5. **Answer Delivery**: The chatbot presents the response in a structured format.

## Example Queries
- "What are the risk factors for Type 2 Diabetes?"
- "Compare and contrast Crohn’s Disease and Ulcerative Colitis."
- "Which of the following is the most likely diagnosis based on these symptoms? (MCQ format)"

## Disclaimer
- **Educational Purposes Only**: This chatbot is not a substitute for professional medical advice.
- **Consult a Medical Professional**: Always consult a healthcare provider for medical conditions.
- **Emergency Notice**: If you experience a medical emergency, seek immediate medical attention.

## Future Enhancements
- Improve response accuracy with **advanced prompt engineering**.
- Enhance UI with **interactive elements**.
- Expand **vector database** with additional medical resources.
- Implement **real-time API logging** for better debugging.

## Contributors
- **Karthik Manjunath Hadagali** – Developer & AI Engineer at MealMatch AI

## License
This project is licensed under the **MIT License**. Feel free to contribute and enhance it!

---

**🌟 If you found this project helpful, consider giving it a star ⭐ on GitHub!**

