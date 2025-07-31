import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json # Only for json_main.py

load_dotenv() # Load environment variables at the very beginning

DATA_PATH = "data/"
PDF_FILENAME = "personal_data.json"
CHROMA_PATH = "chroma_db"
# best option so far is chroma_db_fixed

# ... (load_documents and create_documents_from_json functions - no change) ...
def load_documents():
    json_path = os.path.join(DATA_PATH, PDF_FILENAME)
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        json_content = json.load(f)
    documents = create_documents_from_json(json_content, json_path)
    print(f"Loaded JSON document from {json_path} and converted to {len(documents)} structured sections")
    return documents

def create_documents_from_json(json_content, source_path):
    from langchain_core.documents import Document
    import json
    documents = []
    for section_name, section_content in json_content.items():
        if isinstance(section_content, list):
            for i, item in enumerate(section_content):
                if isinstance(item, dict):
                    content = json.dumps(item, indent=2)
                else:
                    content = str(item)
                doc = Document(page_content=content, metadata={"source": source_path, "section": section_name, "item_index": i, "content_type": "list_item"})
                documents.append(doc)
        elif isinstance(section_content, dict):
            content = json.dumps(section_content, indent=2)
            doc = Document(page_content=content, metadata={"source": source_path, "section": section_name, "content_type": "section"})
            documents.append(doc)
        else:
            doc = Document(page_content=str(section_content), metadata={"source": source_path, "section": section_name, "content_type": "text"})
            documents.append(doc)
    return documents

# ... (split_documents function - no change) ...
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, length_function=len)
    final_chunks = []
    for doc in documents:
        if len(doc.page_content) > 1500:
            chunks = text_splitter.split_documents([doc])
            final_chunks.extend(chunks)
        else:
            final_chunks.append(doc)
    print(f"Processed {len(documents)} structured sections into {len(final_chunks)} final chunks")
    for i, chunk in enumerate(final_chunks[:3]):
        print(f"Chunk {i+1} ({chunk.metadata.get('section', 'Unknown')}):")
        print(f"  Content: {chunk.page_content[:100]}...")
        print(f"  Metadata: {chunk.metadata}")
        print("---")
    return final_chunks


# Step 3: Choose and Configure Embedding Model
def get_embedding_function(model_name="models/text-embedding-004"):
    """Initializes the Google Generative AI embedding function."""
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    print(f"Initialized Google Generative AI embeddings with model: {model_name}")
    return embeddings

# Step 4: Set Up Local Vector Store (ChromaDB)
def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    """Initializes or loads the Chroma vector store."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print(f"Vector store initialized/loaded from: {persist_directory}")
    return vectorstore

# Step 5: Index Documents (Embed and Store)
def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """Indexes document chunks into the Chroma vector store."""
    print(f"Indexing {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vectorstore

# Step 6: Build the RAG Chain
def create_rag_chain(vector_store, llm_model_name="gemini-2.0-flash", context_window=8192): # CHANGE THIS LINE
    """Creates the RAG chain."""
    llm = ChatGoogleGenerativeAI(
        model=llm_model_name,
        temperature=0,
    )
    print(f"Initialized ChatGoogleGenerativeAI with model: {llm_model_name}")

    retriever = vector_store.as_retriever(
        search_type="mmr",  # Using MMR for better diversity
        search_kwargs={'k': 15, 'fetch_k': 20}  # Fetch more, return top 15
    )
    print("Retriever initialized.")

    template = """You are answering questions about a person's professional background. Use ALL the provided context to answer questions in first person. When asked about work experience, make sure to include ALL job positions mentioned in the context, in chronological order (most recent first).

    Context:
    {context}

    Question: {question}

    Answer (as the person described in the context, including ALL relevant information):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created.")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain created.")
    return rag_chain

# Step 7: Query Your Documents
def query_rag(chain, question):
    """Queries the RAG chain and prints the response."""
    print("\nQuerying RAG chain...")
    print(f"Question: {question}")
    # Get the retriever context directly to check if it's empty
    # This assumes the retriever is the first step in the chain
    try:
        context = chain.steps[0]["context"].invoke(question)
    except Exception:
        context = None
    # If context is empty or not relevant, print fallback
    if not context or (isinstance(context, list) and all((not doc.page_content.strip()) for doc in context)):
        print("\nResponse:")
        print("I'm sorry, I don't answer questions about this topic. Please ask about my professional background, skills, experience, or related areas.")
        return
    response = chain.invoke(question)
    print("\nResponse:")
    print(response)
    return response

def debug_retrieval(vector_store, question):
    """Debug function to see what chunks are being retrieved."""
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 15, 'fetch_k': 20})
    docs = retriever.invoke(question)
    print(f"\nDEBUG: Retrieved {len(docs)} chunks for question: '{question}'")
    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}:")
        print(f"  Section: {doc.metadata.get('section', 'Unknown')}")
        print(f"  Content: {doc.page_content[:200]}...")
        print("---")

if __name__ == "__main__":
    load_dotenv() # Ensure .env is loaded

    embedding_function = get_embedding_function(model_name="models/text-embedding-004")
    
    # Check if vector store already exists
    if not os.path.exists(CHROMA_PATH):
        print("Vector store not found. Creating new index...")
        docs = load_documents()
        chunks = split_documents(docs)
        vector_store = index_documents(chunks, embedding_function)
    else:
        print("Loading existing vector store...")
        vector_store = get_vector_store(embedding_function)

    # Create RAG Chain
    rag_chain = create_rag_chain(vector_store, llm_model_name="gemini-2.0-flash")

    # Test with debug first
    test_question = "What is your work experience?"
    debug_retrieval(vector_store, test_question)
    query_rag(rag_chain, test_question)

    # Interactive query loop
    print("\nReady to answer questions! (Type 'exit' to quit)")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
        query_rag(rag_chain, question)