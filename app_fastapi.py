import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Import functions from your main.py
from main import (
    load_documents,
    split_documents,
    get_embedding_function,
    get_vector_store,
    index_documents,
    create_rag_chain,
    CHROMA_PATH
)

load_dotenv()

app = FastAPI(
    title="Portfolio RAG API",
    description="API for managing and querying a RAG (Retrieval-Augmented Generation) system for a portfolio.",
    version="1.0.0"
)

rag_chain = None
vector_store = None
embedding_function = None

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For simplicity during testing, allow all origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for the request body
class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    """Initializes RAG components (embedding, vector store, chain) on application startup."""
    global rag_chain, vector_store, embedding_function

    print("Starting up RAG system...")
    embedding_function = get_embedding_function(model_name="models/text-embedding-004")

    if not os.path.exists(CHROMA_PATH):
        print("Vector store not found. Creating new index...")
        docs = load_documents()
        chunks = split_documents(docs)
        vector_store = index_documents(chunks, embedding_function)
    else:
        print("Loading existing vector store...")
        vector_store = get_vector_store(embedding_function)
    
    rag_chain = create_rag_chain(vector_store, llm_model_name="gemini-2.0-flash")
    print("RAG components initialized successfully.")


@app.get("/")
async def read_root():
    return {"message": "Personal RAG API is running! Go to /docs for API documentation."}


@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    """
    Endpoint to ask a question about the person's professional background.
    """
    question = request.question
    print(f"Received question: {question}")

    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain is not initialized.")

    try:
        response = rag_chain.invoke(question)
        print(f"Response: {response}")
        return {"answer": response}
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
